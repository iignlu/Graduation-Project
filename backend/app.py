from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import timm
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
import os
import requests

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
# The web client lives in ../frontend, so serve it from there. This lets the
# repo run as one piece locally; in production the client can still be hosted
# separately and call /predict cross-origin (CORS is enabled below).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(__name__,
            template_folder=FRONTEND_DIR,
            static_folder=os.path.join(FRONTEND_DIR, 'static'),
            static_url_path='/static')
CORS(app)

# -------------------------------------------------
# Device configuration
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Auto-download the model on first run
# -------------------------------------------------
MODEL_PATH = "swinv2_small_window16_256_epoch_19.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1h-DvV6gZIrxFMMnMM_UNLkBV00K5sBE-"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... This may take a minute.")
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)
        print("Model downloaded successfully!")

download_model()

# -------------------------------------------------
# Load model from checkpoint
# -------------------------------------------------
def load_model_from_checkpoint(checkpoint_path, model_class, device):
    model = timm.create_model('swinv2_small_window16_256', pretrained=False, num_classes=model_class)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model_class = 5
model = load_model_from_checkpoint(MODEL_PATH, model_class, device)

# -------------------------------------------------
# Preprocessing helpers
# -------------------------------------------------
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

train_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

level_to_category = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_cropped = crop_image_from_gray(img)
    if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        img_resized = cv2.resize(img, (256, 256))
    else:
        img_resized = cv2.resize(img_cropped, (256, 256))

    # CLAHE
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    pil_image = Image.fromarray(final_img)
    img_tensor = train_transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, 1)

    class_id = predicted_class_id.item()
    confidence_score = confidence.item()
    label = level_to_category.get(class_id, "Unknown")

    return jsonify({
        "class_id": class_id,
        "label": label,
        "confidence": confidence_score
    })

# -------------------------------------------------
# IMPORTANT: Do NOT run app.run()
# Railway/Gunicorn will run the server.
# -------------------------------------------------
