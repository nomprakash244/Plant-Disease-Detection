# streamlit_app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os

# ---------- Configuration ----------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
MODEL_PATH = os.environ.get("MODEL_PATH", "plant_disease_model_1.pt")  # keep model alongside app
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Replace with the actual class names used during training (39 classes from PlantVillage)
# If the original repo provides a classes list/json, paste it here to keep label order consistent.
CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
    "Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy","Background"
]

# ---------- Model loader ----------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    # Model architecture must match training. If the original repo defines a Net class,
    # recreate it here or torch.load a scripted model. For a typical state_dict, we need the class.
    # Attempt to load an entire torch-saved model first; fallback to state_dict if needed.
    model = None
    try:
        model = torch.load(model_path, map_location=DEVICE)
        # If torch.load returns a dict (state_dict), this will not be a nn.Module.
        if not hasattr(model, "eval"):
            raise ValueError("Loaded object is not a torch.nn.Module. Expecting state_dict; define model class.")
    except Exception:
        # Example: define a simple torchvision model matching training (adjust to the repo’s architecture).
        # If the original architecture is ResNet18 fine-tuned to 39 classes, uncomment below:
        # from torchvision.models import resnet18
        # model = resnet18(weights=None)
        # model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        # state = torch.load(model_path, map_location=DEVICE)
        # model.load_state_dict(state)
        # If unsure, prompt for exact architecture.
        raise RuntimeError(
            "Could not load the model with the current architecture. "
            "Please paste the model definition used in training or switch to a scripted/torch.jit model."
        )
    model.to(DEVICE)
    model.eval()
    return model

# ---------- Preprocessing ----------
# Adjust to the training pipeline (check original transforms used)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize if used during training, e.g. ImageNet stats:
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(model, image: Image.Image):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        top_prob, top_idx = probs.max(dim=0)
    return CLASS_NAMES[top_idx.item()], float(top_prob), probs.tolist()

# ---------- UI ----------
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to identify disease using a PyTorch CNN model trained on PlantVillage classes. Ensure the .pt model file is available.", )

with st.sidebar:
    st.header("Settings")
    model_path_input = st.text_input("Model path", MODEL_PATH)
    if model_path_input != MODEL_PATH:
        MODEL_PATH = model_path_input

# Load model lazily
model = None
load_btn = st.sidebar.button("Load / Reload Model")

if load_btn or "model_loaded" not in st.session_state:
    try:
        model = load_model(MODEL_PATH)
        st.session_state["model_loaded"] = True
        st.success(f"Model loaded from: {MODEL_PATH}")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()
else:
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

uploaded = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png","webp"])
if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)
    with st.spinner("Analyzing..."):
        label, confidence, probs = predict_image(model, img)
    st.subheader("Prediction")
    st.write(f"Class: {label}")
    st.write(f"Confidence: {confidence:.2%}")

    # Show top-5 probabilities
    top5 = sorted(list(zip(CLASS_NAMES, probs)), key=lambda x: x[1], reverse=True)[:5]
    st.markdown("Top-5 classes:")
    for cls, p in top5:
        st.write(f"- {cls}: {p:.2%}")

st.caption("Note: Class list and image transforms must match those used during training. If predictions look off, please update CLASS_NAMES and normalization to mirror training.")
