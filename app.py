"""
Hugging Face Spaces - Chest X-Ray Classifier with non-X-ray rejection
Classes: COVID / NORMAL / PNEUMONIA

Place the trusted notebook exports in models/ before starting the app.
"""

import os
import json
import warnings

import cv2
import gradio as gr
import matplotlib
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "efficientnet_b0.pt"))]
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# A closed-set classifier will always choose one of its known classes.
# These gates stop obvious non-CXR images before classification.
MIN_CXR_SCORE = 5
MIN_CONFIDENCE = 0.55
ENTROPY_REJECT = 1.05  # close to ln(3)=1.099 means very uncertain among 3 classes

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    idx_to_class = {int(k): v for k, v in config.get("idx_to_class", {}).items()}
    CLASS_NAMES = [idx_to_class.get(i, str(i)) for i in range(NUM_CLASSES)]
else:
    CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA"]

CLASS_INFO = {
    "COVID": "COVID-19 pattern detected.",
    "COVID19": "COVID-19 pattern detected.",
    "NORMAL": "No obvious abnormality detected by the model.",
    "PNEUMONIA": "Pneumonia pattern detected.",
}


def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    found_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            found_path = path
            break
    if found_path is None:
        raise FileNotFoundError(
            f"No model weights found. Expected one of: {MODEL_PATHS}. Current files: {os.listdir('.')}"
        )

    state = torch.load(found_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    print(f"Loaded model weights from {found_path}")
    print(f"Class names: {CLASS_NAMES}")
    return model


model = load_model()


def remove_black_borders(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return rgb
    x, y, w, h = cv2.boundingRect(coords)
    H, W = rgb.shape[:2]
    if w < 0.45 * W or h < 0.45 * H:
        return rgb
    return rgb[y : y + h, x : x + w]


def enhance_xray(gray):
    p1, p99 = np.percentile(gray, (1, 99))
    if p99 > p1:
        gray = np.clip(gray, p1, p99)
        gray = ((gray - p1) / (p99 - p1) * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def chest_xray_gate(pil_img):
    """
    Lightweight non-X-ray rejection without needing a second model.
    Returns: is_chest_xray, reasons, score
    """
    rgb = np.array(pil_img.convert("RGB"))
    H, W = rgb.shape[:2]
    reasons = []
    score = 0

    if H < 128 or W < 128:
        return False, ["Image is too small. Please upload a clear chest X-ray."], score

    rgb = remove_black_borders(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mean = float(gray.mean())
    std = float(gray.std())
    saturation_mean = float(hsv[:, :, 1].mean())
    saturation_p90 = float(np.percentile(hsv[:, :, 1], 90))

    # 1) Chest X-rays are usually grayscale/low-saturation.
    if saturation_mean < 35 and saturation_p90 < 70:
        score += 2
    else:
        reasons.append("The image appears too colourful for a standard chest X-ray.")

    # 2) Reject blank, very flat, or extremely dark/bright images.
    if 20 <= mean <= 235 and std >= 25:
        score += 1
    else:
        reasons.append("The brightness/contrast does not look like a usable X-ray.")

    # 3) Edge density: natural photos often have too many colourful/local edges;
    # blank documents have too few. CXR usually sits in a middle band.
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    if 0.015 <= edge_density <= 0.25:
        score += 1
    else:
        reasons.append("The image structure does not match a chest radiograph.")

    # 4) Lung-field style symmetry check: front CXRs have two broad dark regions.
    small = cv2.resize(gray, (224, 224))
    small = cv2.GaussianBlur(small, (5, 5), 0)
    central = small[35:190, 25:199]
    threshold = np.percentile(central, 45)
    dark = (central < threshold).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h > 35 and w > 18:
            valid_regions.append((x, y, w, h, area))
    if len(valid_regions) >= 2:
        score += 2
    else:
        reasons.append("Could not detect the expected two-lung chest X-ray structure.")

    # 5) Aspect ratio check. Do not reject only on this, because some CXRs are square.
    aspect = rgb.shape[1] / max(rgb.shape[0], 1)
    if 0.70 <= aspect <= 1.60:
        score += 1
    else:
        reasons.append("The image shape is unusual for a frontal chest X-ray.")

    return score >= MIN_CXR_SCORE, reasons, score


def preprocess_for_model(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    rgb = remove_black_borders(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enhanced = enhance_xray(gray)
    rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    processed_pil = Image.fromarray(rgb_enhanced)

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tensor = transform(processed_pil).unsqueeze(0).to(DEVICE)
    return tensor, processed_pil


tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda img: transforms.functional.rotate(img, 5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]


def predict_probs(processed_pil):
    probs_all = []
    with torch.no_grad():
        for tfm in tta_transforms:
            tensor = tfm(processed_pil).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            probs_all.append(probs)
    return np.mean(probs_all, axis=0)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


grad_cam = GradCAM(model, model.conv_head)


def make_gradcam_overlay(original_pil, tensor, class_idx):
    try:
        tensor = tensor.clone().detach().requires_grad_(True)
        cam = grad_cam.generate(tensor, class_idx)
        base = np.array(original_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
        heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        coloured = plt.cm.jet(heatmap)[..., :3]
        overlay = np.clip(0.60 * base + 0.40 * coloured, 0, 1)
        return Image.fromarray((overlay * 255).astype(np.uint8))
    except Exception as e:
        print("Grad-CAM failed:", e)
        return None


def empty_probs():
    return {c: 0.0 for c in CLASS_NAMES}


def classify_image(image):
    if image is None:
        return None, empty_probs(), "## Please upload a chest X-ray image.", ""

    try:
        is_cxr, reasons, score = chest_xray_gate(image)
        if not is_cxr:
            msg = (
                "## ❌ Invalid input: this does not look like a chest X-ray\n\n"
                "The uploaded image was rejected before classification, so the model will not force it into COVID, Normal, or Pneumonia.\n\n"
                f"**Chest X-ray validation score:** {score}/7 (threshold: {MIN_CXR_SCORE})\n\n"
                "**Reason:** " + (" ".join(reasons[:2]) if reasons else "Image failed the chest X-ray validation checks.") + "\n\n"
                "Please upload a frontal PA/AP chest X-ray where both lungs are visible."
            )
            return None, empty_probs(), msg, ""

        tensor, processed_pil = preprocess_for_model(image)
        probs = predict_probs(processed_pil)
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())

        # Second safety layer: if the classifier itself is too uncertain, do not report diagnosis.
        if confidence < MIN_CONFIDENCE or entropy > ENTROPY_REJECT:
            msg = (
                "## ⚠️ Uncertain / unsupported image\n\n"
                "The image passed the basic X-ray check, but the model is not confident enough to provide a reliable class.\n\n"
                f"**Top confidence:** {confidence*100:.1f}%\n\n"
                "Please try a clearer frontal chest X-ray."
            )
            return None, {c: float(p) for c, p in zip(CLASS_NAMES, probs)}, msg, ""

        overlay = make_gradcam_overlay(processed_pil, tensor, pred_idx)
        conf_dict = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
        desc = CLASS_INFO.get(pred_class, "Model prediction completed.")

        bars = "\n".join(
            [f"- **{c}:** {p*100:.1f}%" for c, p in zip(CLASS_NAMES, probs)]
        )
        result = (
            f"## Prediction: **{pred_class}**\n\n"
            f"**Confidence:** {confidence*100:.1f}%\n\n"
            f"**Status:** {desc}\n\n"
            "### Probability breakdown\n"
            f"{bars}\n\n"
            "---\n"
            "Input was first checked as a likely chest X-ray, then classified using EfficientNet-B0 with TTA."
        )
        disclaimer = (
            "⚕️ **Medical disclaimer:** This application is for educational/research use only. "
            "It is not a clinical diagnostic tool and must not be used for medical decisions."
        )
        return overlay, conf_dict, result, disclaimer

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, empty_probs(), f"## ❌ Error\n\n{str(e)}", ""


with gr.Blocks(title="Chest X-Ray Classifier", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
# Chest X-Ray Classifier
### COVID-19 | Normal | Pneumonia
Educational research prototype. Not for medical decisions.
Upload a **frontal chest X-ray**. Basic heuristic checks may reject unsuitable images; they are not a validated X-ray detector.
"""
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload chest X-ray", height=320)
            btn = gr.Button("Analyse X-Ray", variant="primary")
            gr.Markdown(
                f"""
### Model details
| Item | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Classes | {', '.join(CLASS_NAMES)} |
| Input validation | Non-X-ray rejection gate |
| Explainability | Grad-CAM |

**Important:** CT scans, MRI scans, ordinary photos, animals, documents, and non-medical images are not supported.
"""
            )

        with gr.Column():
            gradcam_output = gr.Image(label="Grad-CAM on processed input (single pass; blank if unavailable)", height=320)
            confidence_output = gr.Label(label="Class probabilities", num_top_classes=3)
            result_text = gr.Markdown()
            disclaimer_text = gr.Markdown()

    btn.click(
        fn=classify_image,
        inputs=input_image,
        outputs=[gradcam_output, confidence_output, result_text, disclaimer_text],
    )

if __name__ == "__main__":
    demo.launch()
