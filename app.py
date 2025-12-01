from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import timm
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
import os

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__, template_folder='.')
CORS(app)

# -------------------------------------------------
# Device configuration
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load model from checkpoint (same as notebook)
# -------------------------------------------------
def load_model_from_checkpoint(checkpoint_path, model_class, device):
    model = timm.create_model('swinv2_small_window16_256', pretrained=False, num_classes=model_class)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

checkpoint_path = "swinv2_small_window16_256_epoch_19.pt"
model_class = 5
model = load_model_from_checkpoint(checkpoint_path, model_class, device)

# -------------------------------------------------
# Preprocessing functions (exactly like training)
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

# Transformation identical to notebook testing
train_transforms_DeiT_base_patch16 = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# Label mapping
level_to_category = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# -------------------------------------------------
# Flask routes
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

    # ---- Preprocessing identical to notebook ----
    img_cropped = crop_image_from_gray(img)
    if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        # Handle cases where cropping removes the entire image
        img_resized = cv2.resize(img, (256, 256))
    else:
        img_resized = cv2.resize(img_cropped, (256, 256))
    
    # Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    pil_image = Image.fromarray(final_img)
    img_tensor = train_transforms_DeiT_base_patch16(pil_image).unsqueeze(0).to(device)

    # ---- Inference ----
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, 1)

    # ---- Prepare JSON response ----
    class_id = predicted_class_id.item()
    confidence_score = confidence.item()
    label = level_to_category.get(class_id, "Unknown")

    return jsonify({
        "class_id": class_id,
        "label": label,
        "confidence": confidence_score
    })

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
