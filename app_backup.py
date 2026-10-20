
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "brain_mri_resnet50_v0.1.0.pt"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor",
]

DEVICE = torch.device("cpu")


# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = models.resnet50(weights=None)

    # The checkpoint uses fc.1, so reproduce its original head.
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 4),
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ---------------- IMAGE PREPROCESSING ----------------

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- DASHBOARD ----------------

st.title("🧠 Brain MRI Analysis")
st.caption(
    "AI-assisted brain MRI classification | Local CPU inference"
)



uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.markdown("### Analyze an MRI image")
    st.write(
        "Upload an image to view the model prediction "
        "and its confidence scores."
    )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1.2], gap="large")

        with left:
            st.subheader("Uploaded MRI")
            st.image(
                image,
                caption="Input image",
                use_container_width=True,
            )
            st.caption(f"Image size: {image.width} × {image.height}")

        with right:
            st.subheader("Classification Result")

            if st.button("Analyze MRI", type="primary"):
                try:
                    with st.spinner("Loading model and analyzing image..."):
                        model = load_model()
                        input_tensor = preprocess(image).unsqueeze(0)

                        with torch.inference_mode():
                            logits = model(input_tensor)
                            probabilities = torch.softmax(logits, dim=1)[0]

                        predicted_index = int(
                            torch.argmax(probabilities).item()
                        )
                        predicted_class = CLASS_NAMES[predicted_index]
                        confidence = float(
                            probabilities[predicted_index].item()
                        )

                    st.success("Inference completed")

                    st.metric(
                        label="Predicted Class",
                        value=predicted_class,
                        delta=f"{confidence:.1%} model confidence",
                    )

                    st.markdown("#### Class confidence scores")

                    for name, score in zip(
                        CLASS_NAMES, probabilities.tolist()
                    ):
                        st.write(f"**{name}** — {score:.1%}")
                        st.progress(float(score))

                    st.caption(
                        "Scores are model outputs, not calibrated "
                        "clinical probabilities."
                    )

                except Exception as exc:
                    st.error(f"Model inference failed: {exc}")

    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")

st.divider()
st.caption(
    "Prototype: ResNet-50 | PyTorch | Streamlit | CPU"
)
