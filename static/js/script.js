// --- Tab Switcher ---
function openTab(tabId, event) {
  const tabs = document.querySelectorAll('.tab');
  const contents = document.querySelectorAll('.tab-content');
  tabs.forEach(tab => tab.classList.remove('border-b-2', 'border-blue-600', 'text-blue-600', 'font-semibold'));
  contents.forEach(content => content.classList.add('hidden'));
  document.getElementById(tabId).classList.remove('hidden');
  event.target.classList.add('border-b-2', 'border-blue-600', 'text-blue-600', 'font-semibold');
}

// --- Upload Area Creator ---
function createUploadArea({ targetId, mediaType = 'image', apiEndpoint }) {
  const acceptTypes = mediaType === 'image' ? 'image/*' : 'video/*';
  const supportsText = mediaType === 'image' ? 'JPEG, PNG, GIF (Max 10MB)' : 'MP4, AVI, MOV (Max 50MB)';

  const container = document.getElementById(targetId);
  container.innerHTML = `
    <form method="POST" action="${apiEndpoint}" enctype="multipart/form-data" id="upload-form-${mediaType}" class="border-2 border-dashed border-gray-300 rounded-lg p-6 bg-white/40 text-center">
      <label for="file-${mediaType}" class="cursor-pointer block">
        <div class="space-y-3">
          <svg class="w-12 h-12 text-gray-400 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1" />
            <path d="M12 12V4m0 0L8 8m4-4l4 4" />
          </svg>
          <p class="text-lg font-medium text-gray-700">Drag and drop your file here</p>
          <p class="text-sm text-gray-500">or click to browse files</p>
          <p class="text-xs text-gray-400">Supports: ${supportsText}</p>
        </div>
        <input id="file-${mediaType}" name="file" type="file" accept="${acceptTypes}" class="hidden" />
      </label>
      <div class="mt-4">
        <button id="submit-btn-${mediaType}" type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50" disabled>
          Detect Number Plate
        </button>
      </div>
      <div id="preview-${mediaType}" class="mt-6"></div>
    </form>
  `;

  const fileInput = document.getElementById(`file-${mediaType}`);
  const submitBtn = document.getElementById(`submit-btn-${mediaType}`);
  const previewDiv = document.getElementById(`preview-${mediaType}`);

  fileInput.addEventListener('change', () => {
    previewDiv.innerHTML = '';
    submitBtn.disabled = !fileInput.files.length;

    if (fileInput.files.length) {
      const file = fileInput.files[0];
      const url = URL.createObjectURL(file);

      if (mediaType === 'image') {
        previewDiv.innerHTML = `<img src="${url}" alt="Preview" class="mx-auto max-h-60 rounded shadow" />`;
      } else {
        previewDiv.innerHTML = `
          <video controls class="mx-auto max-h-60 rounded shadow">
            <source src="${url}" type="${file.type}">
            Your browser does not support the video tag.
          </video>`;
      }
    }
  });

  // Let the form submit normally to Flask — no JavaScript submission here!
}

// --- Feature Rendering ---
function renderFeatures() {
  const features = [
    {
      icon: "📷",
      title: "AI-Powered Recognition",
      description: "Advanced neural networks trained specifically for Nepali number plates with high accuracy."
    },
    {
      icon: "🌐",
      title: "Dual Language Support",
      description: "Recognizes both Nepali (देवनागरी) and English characters on license plates."
    },
    {
      icon: "⚡",
      title: "Fast Processing",
      description: "Get results in seconds with our optimized recognition algorithms."
    },
    {
      icon: "🛡️",
      title: "Secure & Private",
      description: "Your uploaded images are processed securely and not stored permanently."
    },
  ];

  const container = document.getElementById("features-container");
  container.innerHTML = features.map(f => `
    <div class="bg-white text-center border rounded-lg p-6 hover:shadow-lg transition-shadow">
      <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4 text-2xl">${f.icon}</div>
      <h4 class="text-lg font-bold mb-2">${f.title}</h4>
      <p class="text-sm text-gray-600">${f.description}</p>
    </div>
  `).join("");
}
// --- Language Toggle ---
let isNepali = true;

function toggleLanguage() {
  const title = document.getElementById("hero-title");
  const label = document.getElementById("lang-label");
  const flag = document.getElementById("lang-flag");

  if (isNepali) {
    title.textContent = "Nepali Number Plate Recognition System";
    label.textContent = "Nepal";
    flag.textContent = "🇳🇵";
  } else {
    title.textContent = "नेपाली नम्बर प्लेट पहिचान प्रणाली";
    label.textContent = "Nepal";
    flag.textContent = "🇳🇵";
  }

  isNepali = !isNepali;
}

// --- Initialize ---
document.addEventListener("DOMContentLoaded", () => {
  renderFeatures();
  createUploadArea({ targetId: 'upload-card', mediaType: 'image', apiEndpoint: '/upload_image' });
  createUploadArea({ targetId: 'video-tab', mediaType: 'video', apiEndpoint: '/upload_video' });
});