class PlateInfo:
    def __init__(self, plate_bbox, plate_conf, plate_color, text, text_conf, lang, source, original_image_path, result_image_path, timestamp):
        self.plate_bbox = plate_bbox
        self.plate_conf = plate_conf
        self.plate_color = plate_color
        self.text = text
        self.text_conf = text_conf
        self.lang = lang
        self.source = source
        self.original_image_path = original_image_path
        self.result_image_path = result_image_path
        self.timestamp = timestamp

    def save_plate_info(self):
        import csv
        import os

        filename = 'plate_db.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)

            # Write header only if file is new
            if not file_exists:
                writer.writerow([
                    'Plate BBox', 'Plate Confidence', 'Plate Color',
                    'Text', 'Text Confidence', 'Language',
                    'Source', 'Original Image Path', 'Result Image Path', 'Timestamp'
                ])

            writer.writerow([
                self.plate_bbox,
                self.plate_conf,
                self.plate_color,
                self.text,
                self.text_conf,
                self.lang,
                self.source,
                self.original_image_path,
                self.result_image_path,
                self.timestamp
            ])
def get_all_plate_info(source = None):
    import csv
    plate_info_list = []
    try:
        with open('plate_db.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if source and row[6] != source: # If source is specified, filter by source
                    continue
                if len(row) == 10:  # Ensure the row has all required fields
                    plate_info = PlateInfo(
                        plate_bbox=row[0],
                        plate_conf=float(row[1]),
                        plate_color=row[2],
                        text=row[3],
                        text_conf=float(row[4]),
                        lang=row[5],
                        source=row[6],
                        original_image_path=row[7],
                        result_image_path=row[8],
                        timestamp=row[9]
                    )
                    plate_info_list.append(plate_info)
    except FileNotFoundError:
        print("Database file not found. Returning an empty list.")
    return plate_info_list
