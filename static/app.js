const PREDICT_URL = "/predict"; // changeable
const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15 MB
const CLASS_MAP = {
    0: {
        label: "No DR",
        info: "No abnormalities were detected. This indicates no visible signs of diabetic retinopathy. It is essential to continue with regular annual screenings to monitor eye health."
    },
    1: {
        label: "Mild",
        info: "The earliest stage. Small areas of swelling in the retina's blood vessels, known as microaneurysms, are present. At this stage, vision is typically not affected. Routine follow-up is recommended."
    },
    2: {
        label: "Moderate",
        info: "Progression of the condition with more significant damage to blood vessels, which may lead to blockages. This stage requires closer monitoring and a consultation with an ophthalmologist is advised."
    },
    3: {
        label: "Severe",
        info: "Many blood vessels are blocked, depriving several areas of the retina of their blood supply. These areas signal the retina to grow new blood vessels. Clinical advice should be sought soon."
    },
    4: {
        label: "Proliferative DR",
        info: "The most advanced stage. The retina triggers the growth of new, fragile blood vessels (neovascularization). These can leak blood, causing severe vision loss and even blindness. Specialist care is required promptly."
    }
};

let els = {}; // This will be populated after the DOM is ready
let selectedFile = null;
let previewDataUrl = "";
let inFlight = false;
const history = [];

function initTheme() {
    const saved = localStorage.getItem("dr-theme") || "light";
    applyTheme(saved);
}

function applyTheme(theme) {
    els.body.dataset.theme = theme;
    localStorage.setItem("dr-theme", theme);
    const isDark = theme === "dark";
    els.themeToggle.setAttribute("aria-pressed", String(isDark));
    els.themeToggle.dataset.theme = theme;
    els.themeToggle.querySelector('.toggle-thumb').style.transform = isDark ? 'translateX(2rem)' : 'translateX(0)';
}

function toggleTheme() {
    const next = els.body.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
}


function wireEvents() {
    els.themeToggle.addEventListener("click", toggleTheme);

    ["dragover", "dragenter"].forEach(evt =>
        els.dropZone.addEventListener(evt, e => {
            e.preventDefault();
            els.dropZone.classList.add("drag-over");
        })
    );

    ["dragleave", "dragend"].forEach(evt =>
        els.dropZone.addEventListener(evt, () => {
            els.dropZone.classList.remove("drag-over");
        })
    );

    els.dropZone.addEventListener("drop", handleDrop);
    els.dropZone.addEventListener("click", () => els.fileInput.click());
    els.dropZone.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            els.fileInput.click();
        }
    });

    els.chooseImage.addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", () => {
        if (els.fileInput.files && els.fileInput.files[0]) {
            handleFile(els.fileInput.files[0]);
        }
    });

    els.analyzeBtn.addEventListener("click", analyzeImage);
    els.retryBtn.addEventListener("click", analyzeImage);
    els.uploadAnother.addEventListener("click", handleReset);
}

function handleDrop(e) {
    e.preventDefault();
    els.dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const isValidType = ["image/jpeg", "image/png", "image/jpg"].includes(file.type);
    if (!isValidType) {
        showError("Please upload a JPG or PNG image.");
        els.statusText.textContent = "Only JPG and PNG files are supported.";
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        showError("File is too large. Please choose an image under 15 MB.");
        els.statusText.textContent = "Selected file exceeds 15 MB.";
        return;
    }

    selectedFile = file;
    els.uploadError.textContent = "";
    enableAnalyze(true);
    els.statusText.textContent = "Image ready. Click Analyze.";

    const reader = new FileReader();
    reader.onload = evt => {
        previewDataUrl = String(evt.target?.result || "");
        renderPreview(previewDataUrl);
    };
    reader.readAsDataURL(file);
}

function renderPreview(src) {
    els.preview.innerHTML = src
        ? `<img src="${src}" alt="Selected retinal image preview">`
        : "";
}

function enableAnalyze(state) {
    els.analyzeBtn.disabled = !state || inFlight;
}

async function analyzeImage() {
    if (!selectedFile || inFlight) {
        return;
    }
    inFlight = true;
    enableAnalyze(false);
    els.retryBtn.hidden = true;
    els.statusText.textContent = "Analyzing image…";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch(PREDICT_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const payload = await response.json();
        applyPrediction(payload);
    } catch (err) {
        console.error(err);
        showError("Something went wrong with the prediction. Please try again.");
        els.retryBtn.hidden = false;
        els.statusText.textContent = "Prediction failed.";
    } finally {
        inFlight = false;
        enableAnalyze(Boolean(selectedFile));
    }
}

function applyPrediction(payload) {
    const classId = Number(payload.class_id);
    const info = CLASS_MAP[classId];
    if (!info) {
        showError("Received an unexpected class from the model.");
        els.statusText.textContent = "Prediction returned an unknown class.";
        return;
    }

    const confidence = Number(payload.confidence) || 0;
    els.classBadge.textContent = `${info.label} (${classId})`;
    els.confidenceLabel.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    els.resultsCard.hidden = false;

    updateSliderHighlight(classId, confidence);
    updateDetailsText(classId);
    els.statusText.textContent = "Prediction complete.";

    addHistoryEntry({
        classId,
        label: info.label,
        confidence,
        timestamp: new Date(),
        thumb: previewDataUrl || payload.image_url || ""
    });
}

function initSliderTicks() {
    els.sliderRail.innerHTML = "";
    for (let i = 0; i <= 4; i += 1) {
        const tickButton = document.createElement("button");
        tickButton.className = "slider-tick";
        tickButton.type = "button";
        tickButton.style.left = `${(i / 4) * 100}%`;
        tickButton.dataset.classId = String(i);
        tickButton.title = `${CLASS_MAP[i].label} (${i})`;
        tickButton.addEventListener("click", () => {
            updateDetailsText(i);
        });
        tickButton.addEventListener("keyup", event => {
            if (event.key === "Enter" || event.key === " ") {
                updateDetailsText(i);
            }
        });

        const label = document.createElement("span");
        label.className = "slider-tick-label";
        label.textContent = i.toString();

        const wrapper = document.createElement("div");
        wrapper.className = "slider-node";
        wrapper.style.left = `${(i / 4) * 100}%`;
        wrapper.appendChild(tickButton);
        wrapper.appendChild(label);

        els.sliderRail.appendChild(wrapper);
    }
}

function updateSliderHighlight(activeClass, confidence) {
    const ticks = els.sliderRail.querySelectorAll(".slider-tick");
    ticks.forEach(tick => {
        tick.classList.toggle("active", Number(tick.dataset.classId) === activeClass);
    });

    const glowStrength = Math.min(Math.max(confidence, 0.2), 1);
    els.sliderRail.style.boxShadow = `0 0 20px rgba(37, 99, 235, ${glowStrength / 2})`;
    els.classSlider.setAttribute("aria-valuenow", String(activeClass));
    els.classSlider.setAttribute("aria-valuetext", CLASS_MAP[activeClass].label);
}

function updateDetailsText(classId) {
    const info = CLASS_MAP[classId];
    els.classInfo.textContent = info ? info.info : "";
    els.detailsPanel.style.borderColor = `var(--border)`; // Reset border
}

function showError(message) {
    els.uploadError.textContent = message;
}

function addHistoryEntry(entry) {
    history.unshift(entry);
    if (history.length > 5) {
        history.pop();
    }
    renderHistory();
}

function renderHistory() {
    if (!history.length) {
        els.historyList.innerHTML = '<li class="helper-text">No predictions yet.</li>';
        return;
    }
    els.historyList.innerHTML = history
        .map(item => {
            const confidencePercent = (item.confidence * 100).toFixed(0);
            const timestamp = item.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
            const thumbSrc = item.thumb ? item.thumb : "";
            const thumbMarkup = thumbSrc
                ? `<img class="history-thumb" src="${thumbSrc}" alt="${item.label} thumbnail">`
                : `<div class="history-thumb history-thumb--placeholder" aria-hidden="true"></div>`;
            return `
                <li class="history-item">
                    ${thumbMarkup}
                    <div class="history-meta">
                        <strong>${item.label} (${item.classId})</strong>
                        <span>${confidencePercent}% • ${timestamp}</span>
                    </div>
                </li>
            `;
        })
        .join("");
}

function handleReset() {
    selectedFile = null;
    previewDataUrl = "";
    els.fileInput.value = "";
    renderPreview("");
    els.resultsCard.hidden = true;
    els.statusText.textContent = "Select an image to begin.";
    els.uploadError.textContent = "";
    enableAnalyze(false);
    els.dropZone.focus();
}

function cacheDOMElements() {
    els = {
        body: document.body,
        dropZone: document.getElementById("dropZone"),
        fileInput: document.getElementById("fileInput"),
        chooseImage: document.getElementById("chooseImage"),
        analyzeBtn: document.getElementById("analyzeBtn"),
        retryBtn: document.getElementById("retryBtn"),
        statusText: document.getElementById("statusText"),
        uploadError: document.getElementById("uploadError"),
        preview: document.getElementById("imagePreview"),
        resultsCard: document.getElementById("resultsCard"),
        classBadge: document.getElementById("classBadge"),
        confidenceLabel: document.getElementById("confidenceLabel"),
        classSlider: document.getElementById("classSlider"),
        sliderRail: document.getElementById("sliderRail"),
        classInfo: document.getElementById("classInfo"),
        themeToggle: document.getElementById("themeToggle"),
        detailsPanel: document.getElementById("detailsPanel"),
        historyList: document.getElementById("historyList"),
        uploadAnother: document.getElementById("uploadAnother")
    };
}

document.addEventListener("DOMContentLoaded", () => {
    cacheDOMElements(); // Find all elements now that the page is loaded
    initTheme();
    initSliderTicks();
    wireEvents(); // Attach event listeners
    updateDetailsText(0);
    renderHistory();
});
