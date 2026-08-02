"""
Diabetic Retinopathy Detection — Hugging Face Space (Gradio + ZeroGPU).

Same preprocessing and model as backend/app.py, wrapped in Gradio so it can
run on ZeroGPU, which free personal accounts get two of.

ZeroGPU rules this file follows:
  - Gradio SDK (ZeroGPU supports no other SDK)
  - the model is moved to cuda at module level, not inside the GPU function
  - the inference function is decorated with @spaces.GPU
"""

import os

import cv2
import gradio as gr
import numpy as np
import spaces
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# Point these at the model repo holding the checkpoint. Keeping weights in a
# model repo rather than in the Space keeps the Space small and rebuilds fast.
REPO_ID = os.getenv("MODEL_REPO_ID", "iignlu/swinv2-dr-aptos2019")
FILENAME = os.getenv("MODEL_FILENAME", "swinv2_small_window16_256_epoch_19.pt")
MODEL_NAME = "swinv2_small_window16_256"
NUM_CLASSES = 5

checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()
# Must happen at module level — ZeroGPU emulates CUDA outside @spaces.GPU so
# this is allowed, and placing it here is far faster than transferring inside.
model.to("cuda")

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

NOTES = {
    0: "No abnormalities detected. Routine annual screening should continue.",
    1: "Microaneurysms only. Vision is typically unaffected at this stage.",
    2: "Multiple microaneurysms, scattered haemorrhages and/or soft exudates.",
    3: "Extensive haemorrhages or vascular blockage; progression looks imminent.",
    4: "Neovascularisation — new fragile vessels. Highest risk of vision loss.",
}


# --------------------------------------------------------------------------
# Preprocessing — identical to the Flask backend
# --------------------------------------------------------------------------
def crop_image_from_gray(img, tol=7):
    """Crop the black surround so the retina fills the frame."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
        return img
    return np.stack(
        [img[:, :, i][np.ix_(mask.any(1), mask.any(0))] for i in range(3)],
        axis=-1,
    )


def preprocess(rgb):
    cropped = crop_image_from_gray(rgb)
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = rgb
    resized = cv2.resize(cropped, (256, 256))

    # CLAHE on lightness only, so local contrast lifts without shifting colour
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@spaces.GPU(duration=30)
def predict(image):
    if image is None:
        return {}, "Upload a retinal fundus photograph to begin."

    processed = preprocess(np.array(image.convert("RGB")))
    tensor = transform(Image.fromarray(processed)).unsqueeze(0).to("cuda")

    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]

    scores = {LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    top = int(torch.argmax(probs))
    summary = f"**{LABELS[top]}** — {probs[top]:.1%} confidence\n\n{NOTES[top]}"
    return scores, summary


DISCLAIMER = """
> ⚠️ **Not a medical device.** This is a student research project. It has not been
> clinically validated or approved by any regulatory body, and nothing it outputs is a
> diagnosis. Screening for diabetic retinopathy must be done by a qualified ophthalmologist.
"""

with gr.Blocks(title="Diabetic Retinopathy Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Diabetic Retinopathy Detection")
    gr.Markdown(
        "Grades diabetic retinopathy severity (0–4) from a retinal fundus photograph, "
        "using a Swin Transformer V2 fine-tuned on APTOS 2019."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Retinal image", height=340)
            run = gr.Button("Analyse", variant="primary")
        with gr.Column():
            label_out = gr.Label(num_top_classes=5, label="Severity")
            text_out = gr.Markdown()

    run.click(predict, inputs=image_in, outputs=[label_out, text_out])
    image_in.change(predict, inputs=image_in, outputs=[label_out, text_out])

    gr.Markdown(
        "Source: [github.com/iignlu/Graduation-Project]"
        "(https://github.com/iignlu/Graduation-Project)"
    )

demo.queue().launch()
