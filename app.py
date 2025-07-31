import re
import os
import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from collections import deque
from sympy import pretty_print
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from helper import calculate_box_iou
from flask import Flask, request, jsonify, Response, render_template, redirect, url_for

app = Flask(__name__)

# app.config['DEBUG_PREPROCESSING'] = True

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_path = 'models/od/best.pt'

yolo_model = YOLO(model_path)
reader = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available(),
    recog_network='english_g2',
    model_storage_directory='models/ocr',
    download_enabled=True,
    quantize=False,
    cudnn_benchmark=True
)
nep_reader = easyocr.Reader(
    ['ne'],
    gpu=torch.cuda.is_available(),
    # recog_network='devanagari.pth',
    model_storage_directory='models/ocr',
    download_enabled=True,
    quantize=False,
    cudnn_benchmark=True
)

PLATE_CONFIDENCE_THRESHOLD = 0.7
TEXT_CONFIDENCE_THRESHOLD = 0.4
ALLOWED_EN_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')

PLATE_PATTERN = re.compile(
    r'^'
    r'([A-Z]+\s+)?'          # Optional state
    r'([A-Z])\s*'            # L
    r'([A-Z]{2})\s*'         # LL
    r'(\d{4})'               # NNNN
    r'$',
    re.IGNORECASE
)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_plate_text(text):
    cleaned = ''.join(c.upper() for c in text if c.upper() in ALLOWED_EN_CHARS)
    cleaned = ' '.join(cleaned.split())
    match = PLATE_PATTERN.match(cleaned)
    if match:
        state, first_letter, two_letters, numbers = match.groups()
        if state:
            formatted = f"{state.strip()} {first_letter} {two_letters} {numbers}"
        else:
            formatted = f"{first_letter} {two_letters} {numbers}"
        return formatted
    letters = ''.join(c for c in cleaned if c.isalpha())
    numbers = ''.join(c for c in cleaned if c.isdigit())
    if len(letters) >= 3 and len(numbers) >= 4:
        state_part = ''
        letter_part = letters[-3:]
        number_part = numbers[-4:]
        if len(letters) > 3:
            state_part = letters[:-3] + ' '
        return f"{state_part}{letter_part[0]} {letter_part[1:]} {number_part}"
    return cleaned

def validate_nepali_plate_text(text):
    # Nepali plates typically contain Devanagari digits (०१२३४५६७८९)
    devanagari_digits = set('०१२३४५६७८९')
    has_devanagari = any(c in devanagari_digits for c in text)
    if has_devanagari and len(text) >= 6:
        return text
    return ""

def preprocess_plate_image(plate_img):
    height, width = plate_img.shape[:2]
    resize_factor = 2
    plate_img = cv2.resize(plate_img, (width * resize_factor, height * resize_factor))
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(dilated) * 255
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter small contours
            cv2.drawContours(mask, [contour], -1, 0, -1)
    final = cv2.bitwise_and(dilated, mask)
    if app.config.get('DEBUG_PREPROCESSING', False):
        cv2.imshow('original', plate_img)
        cv2.imshow('gray', gray)
        cv2.imshow('filtered', filtered)
        cv2.imshow('enhanced', enhanced)
        cv2.imshow('thresholded', thresh)
        cv2.imshow('cleaned', cleaned)
        cv2.imshow('dilated', dilated)
        cv2.imshow('final', final)
        cv2.waitKey(0)
    return final

def process_english_plate(results):
    if not results:
        return ""
    results.sort(key=lambda x: x['confidence'], reverse=True)
    state_name = None
    for item in results:
        text = item['text']
        if len(text) > 4 and text.isalpha() and item['confidence'] > 0.9:
            state_name = text
            break
    all_letters = []
    all_numbers = []
    for item in results:
        text = item['text']
        if state_name and text == state_name:
            continue
        letters = ''.join(c for c in text if c.isalpha())
        numbers = ''.join(c for c in text if c.isdigit())
        if letters:
            all_letters.append(letters)
        if numbers:
            all_numbers.append(numbers)
    combined_letters = ''.join(all_letters)
    combined_numbers = ''.join(all_numbers)
    plate_text = ""
    if state_name:
        plate_text += f"{state_name} "
    if len(combined_letters) >= 3:
        plate_text += f"{combined_letters[0]} {combined_letters[1:3]} "
    elif combined_letters:
        plate_text += f"{combined_letters} "
    if combined_numbers:
        if len(combined_numbers) > 4:
            plate_text += combined_numbers[-4:]
        else:
            plate_text += combined_numbers
    return validate_plate_text(plate_text)

def process_nepali_plate(results):
    if not results:
        return ""
    results.sort(key=lambda x: x['confidence'], reverse=True)
    text_votes = {}
    for item in results:
        text = item['text'].strip()
        if text:
            if text in text_votes:
                text_votes[text] += item['confidence']
            else:
                text_votes[text] = item['confidence']
    if not text_votes:
        return ""
    best_text = max(text_votes.items(), key=lambda x: x[1])[0]
    validated = validate_nepali_plate_text(best_text)
    return validated if validated else best_text

def process_image(image):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_model(image)
    recognized_plates = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf > PLATE_CONFIDENCE_THRESHOLD and box.cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image[y1:y2, x1:x2]
                padding = 10
                padded_plate = cv2.copyMakeBorder(plate_img, padding, padding, padding, padding,
                                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
                processed = preprocess_plate_image(padded_plate)
                equalized = cv2.equalizeHist(cv2.cvtColor(padded_plate, cv2.COLOR_BGR2GRAY))
                english_results = []
                nepali_results = []
                for img, method_name in [
                    (padded_plate, "original"),
                    (processed, "processed"),
                    (equalized, "equalized")
                ]:
                    try:
                        eng_texts = reader.readtext(
                            img,
                            decoder='beamsearch',
                            beamWidth=10,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                            paragraph=False
                        )
                        for result_item in eng_texts:
                            if len(result_item) >= 3 and result_item[2] >= TEXT_CONFIDENCE_THRESHOLD:
                                coords, text, conf = result_item
                                clean_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip().upper()
                                if clean_text:
                                    english_results.append({
                                        'text': clean_text,
                                        'confidence': conf,
                                        'method': method_name
                                    })
                        nep_texts = nep_reader.readtext(
                            img,
                            decoder='beamsearch',
                            beamWidth=10,
                            paragraph=False
                        )
                        for result_item in nep_texts:
                            if len(result_item) >= 3 and result_item[2] >= TEXT_CONFIDENCE_THRESHOLD:
                                coords, text, conf = result_item
                                if text.strip():
                                    nepali_results.append({
                                        'text': text.strip(),
                                        'confidence': conf,
                                        'method': method_name
                                    })
                    except Exception as e:
                        print(f"Error with {method_name}: {e}")
                eng_avg_conf = sum(r['confidence'] for r in english_results) / len(english_results) if english_results else 0
                nep_avg_conf = sum(r['confidence'] for r in nepali_results) / len(nepali_results) if nepali_results else 0
                eng_score = eng_avg_conf * len(english_results) if english_results else 0
                nep_score = nep_avg_conf * len(nepali_results) if nepali_results else 0
                is_english = eng_score >= nep_score
                if is_english:
                    final_text = process_english_plate(english_results)
                else:
                    final_text = process_nepali_plate(nepali_results)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    image,
                    final_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 180, 255),
                    2
                )
                recognized_plates.append({
                    'coordinates': (x1, y1, x2, y2),
                    'text': final_text,
                    'confidence': float(box.conf),
                    'language': 'english' if is_english else 'nepali'
                })
    return image, recognized_plates

MAX_TRACKING_AGE = 30   # Maximum number of frames to keep tracking a disappeared object
MIN_HITS = 3            # Minimum number of detections before considering a plate valid
IOU_THRESHOLD = 0.3     # IoU threshold to match detections between frames
tracked_plates = {}     # Dict to store tracked plates
next_plate_id = 0       # ID counter for new plate detections

def process_video_frame(frame, frame_count):
    global next_plate_id, tracked_plates
    processed_frame, current_detections = process_image(frame)
    active_ids = []
    unmatched_detections = []
    for detection in current_detections:
        x1, y1, x2, y2 = detection['coordinates']
        detection_box = [x1, y1, x2, y2]
        best_iou = IOU_THRESHOLD
        best_id = None
        for track_id, track_info in tracked_plates.items():
            if track_info['active']:
                track_box = track_info['last_box']
                iou = calculate_box_iou(detection_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
        if best_id is not None:
            tracked_plates[best_id]['last_box'] = detection_box
            tracked_plates[best_id]['text'] = detection['text']
            tracked_plates[best_id]['last_seen'] = frame_count
            tracked_plates[best_id]['hits'] += 1
            tracked_plates[best_id]['text_history'].append(detection['text'])
            tracked_plates[best_id]['language'] = detection.get('language', 'unknown')
            active_ids.append(best_id)
        else:
            unmatched_detections.append(detection)
    for detection in unmatched_detections:
        x1, y1, x2, y2 = detection['coordinates']
        tracked_plates[next_plate_id] = {
            'last_box': [x1, y1, x2, y2],
            'text': detection['text'],
            'first_seen': frame_count,
            'last_seen': frame_count,
            'hits': 1,
            'active': True,
            'text_history': deque(maxlen=10),  # Store last 10 readings
            'confidence': detection['confidence'],
            'language': detection.get('language', 'unknown')
        }
        tracked_plates[next_plate_id]['text_history'].append(detection['text'])
        active_ids.append(next_plate_id)
        next_plate_id += 1
    for track_id, track_info in tracked_plates.items():
        if track_info['active'] and (frame_count - track_info['last_seen'] > MAX_TRACKING_AGE):
            track_info['active'] = False
        if track_info['active'] and track_id in active_ids and track_info['hits'] >= MIN_HITS:
            x1, y1, x2, y2 = track_info['last_box']
            if track_info['text_history']:
                text_counts = {}
                for text in track_info['text_history']:
                    if text in text_counts:
                        text_counts[text] += 1
                    else:
                        text_counts[text] = 1
                most_common_text = max(text_counts.items(), key=lambda x: x[1])[0]
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
                cv2.putText(
                    processed_frame,
                    f"ID:{track_id} {most_common_text}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (77, 77, 255),
                    2
                )
    return processed_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = Image.open(filepath)
        processed_img, plates = process_image(img)
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, processed_img)
        return render_template('image_result.html',
                               original_image=filename,
                               processed_image=processed_filename,
                               plates=plates)
    return redirect(request.url)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        cap = cv2.VideoCapture(filepath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        out = cv2.VideoWriter(processed_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        recognized_plates = []
        frame_count = 0
        global next_plate_id, tracked_plates
        next_plate_id = 0
        tracked_plates = {}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_video_frame(frame, frame_count)
            out.write(processed_frame)
            frame_count += 1
        for track_id, track_info in tracked_plates.items():
            if track_info['hits'] >= MIN_HITS:
                if track_info['text_history']:
                    text_counts = {}
                    for text in track_info['text_history']:
                        if text in text_counts:
                            text_counts[text] += 1
                        else:
                            text_counts[text] = 1
                    most_common_text = max(text_counts.items(), key=lambda x: x[1])[0]
                    recognized_plates.append({
                        'track_id': track_id,
                        'text': most_common_text,
                        'confidence': track_info['confidence'],
                        'hits': track_info['hits'],
                        'language': track_info.get('language', 'unknown')
                    })
        cap.release()
        out.release()
        return render_template('video_result.html',
                               original_video=filename,
                               processed_video=processed_filename,
                               plates=recognized_plates)
    return redirect(request.url)

@app.route('/realtime')
def realtime():
    return render_template('realtime_feed.html')

@app.route('/api/realtime_feed')
def realtime_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    global next_plate_id, tracked_plates
    next_plate_id = 0
    tracked_plates = {}
    while True:
        success, frame = cap.read()
        if not success:
            break
        processed_frame = process_video_frame(frame, frame_count)
        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)