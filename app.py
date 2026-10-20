import numpy as np
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroScan | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CONFIGURATION — KEEP YOUR EXISTING MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "brain_mri_resnet50_v0.1.0.pt"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary Tumor",
]

DEVICE = torch.device("cpu")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #14243A;
        --blue: #3578F6;
        --text: #1D2939;
        --muted: #667085;
        --border: #E5EAF1;
        --background: #F5F7FB;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--background);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.95);
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #25374F;
    }

    [data-testid="stSidebar"] * {
        color: #E8EEF8;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #B8C6DA;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #FFFFFF;
        margin-top: 12px;
    }

    .brand span {
        color: #78A7FF;
    }

    .sidebar-label {
        color: #91A4BF;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 10px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 12px;
    }

    .sidebar-item {
        padding: 12px 14px;
        border-radius: 9px;
        margin: 5px 0;
        font-size: 14px;
        color: #D5DFED;
    }

    .sidebar-active {
        background: #263D5C;
        border: 1px solid #36557E;
        color: white;
        font-weight: 600;
    }

    .sidebar-footer {
        margin-top: 35px;
        padding: 14px;
        background: #1C304A;
        border: 1px solid #304560;
        border-radius: 12px;
        font-size: 12px;
        line-height: 1.7;
        color: #C6D3E4;
    }

    .page-eyebrow {
        color: var(--blue);
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.8px;
        margin-bottom: 8px;
    }

    .page-title {
        font-family: 'Manrope', sans-serif;
        color: var(--navy);
        font-size: clamp(27px, 3vw, 36px);
        font-weight: 800;
        letter-spacing: -1.2px;
        line-height: 1.2;
        margin: 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 14px;
        margin-top: 10px;
        line-height: 1.7;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 17px;
        font-weight: 800;
        color: var(--navy);
        margin-bottom: 3px;
    }

    .section-description {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 18px;
        line-height: 1.6;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 23px;
        box-shadow: 0 3px 14px rgba(22, 39, 64, 0.035);
        margin-bottom: 18px;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 20px;
        background: #E9F8F0;
        color: #18794E;
        font-size: 11px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 7px;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .metric-note {
        font-size: 11px;
        color: var(--muted);
        margin-top: 6px;
        line-height: 1.5;
    }

    .result-card {
        background: #F3F7FF;
        border: 1px solid #D9E6FF;
        border-radius: 13px;
        padding: 18px;
        margin-bottom: 17px;
    }

    .result-card-label {
        color: #41618E;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-card-value {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        color: #1D4FA3;
        overflow-wrap: anywhere;
    }

    .confidence-note {
        font-size: 12px;
        color: var(--muted);
        line-height: 1.6;
        margin-top: 12px;
    }

    .empty-state {
        background: #FFFFFF;
        border: 1px dashed #CAD5E3;
        border-radius: 14px;
        padding: 34px 20px;
        text-align: center;
        color: var(--muted);
        line-height: 1.8;
    }

    .empty-icon {
        font-size: 35px;
        margin-bottom: 10px;
    }

    .footer {
        border-top: 1px solid var(--border);
        padding-top: 17px;
        margin-top: 25px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.8;
    }

    div.stButton > button[kind="primary"] {
        background: #3578F6;
        border: 1px solid #3578F6;
        color: white;
        border-radius: 9px;
        min-height: 45px;
        font-weight: 700;
        width: 100%;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #2464D9;
        border-color: #2464D9;
        box-shadow: 0 4px 12px rgba(53, 120, 246, 0.2);
    }

    [data-testid="stFileUploader"] {
        background: #F8FAFD;
        border: 1px dashed #B8C9E0;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #3578F6;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .panel {
            padding: 16px;
        }

        .page-title {
            font-size: 27px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MODEL LOADING — SAME RESNET-50 CHECKPOINT
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = models.resnet50(weights=None)

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


# --------------------------------------------------
# IMAGE PREPROCESSING — UNCHANGED
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def generate_gradcam(model, input_tensor, original_image, target_class=None):
    """
    Generate a Grad-CAM overlay using the final ResNet-50
    convolutional block (layer4).
    """
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

        def save_gradient(gradient):
            gradients["value"] = gradient

        output.register_hook(save_gradient)

    target_layer = model.layer4
    hook = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        # Grad-CAM requires gradients, so do not use inference_mode here.
        with torch.enable_grad():
            logits = model(input_tensor)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            score = logits[0, target_class]
            score.backward()

        feature_maps = activations["value"].detach()
        feature_gradients = gradients["value"].detach()

        # Average gradients spatially to obtain channel weights.
        weights = feature_gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(original_image.height, original_image.width),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()

        # Normalize the heatmap to [0, 1].
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Create a simple blue-green-yellow-red heatmap.
        red = np.clip(1.5 * cam, 0, 1)
        green = np.clip(1.5 - np.abs(2 * cam - 1) * 1.5, 0, 1)
        blue = np.clip(1.5 * (1 - cam), 0, 1)

        heatmap = np.stack([red, green, blue], axis=-1)
        heatmap = (heatmap * 255).astype(np.uint8)

        heatmap_image = Image.fromarray(heatmap).convert("RGB")
        original = original_image.convert("RGB")

        # Blend the heatmap with the original MRI.
        overlay = Image.blend(original, heatmap_image, alpha=0.42)

        return overlay

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">Neuro<span>Scan</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#91A4BF;font-size:12px;margin-top:4px;">'
        'Brain MRI Classification Prototype</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Workspace</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-item sidebar-active">'
        '🧠 &nbsp; MRI Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item">📊 &nbsp; Classification Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">System</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-item">⚙️ &nbsp; Model Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>MODEL STATUS</strong><br>
            ResNet-50 classifier<br>
            <span style="color:#83D8AB;">●</span> Local inference<br>
            CPU execution
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="color:#91A4BF;font-size:11px;margin-top:25px;">
            NEUROSCAN PROTOTYPE<br>
            Built with PyTorch and Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="page-eyebrow">MEDICAL AI · RESEARCH PROTOTYPE</div>
    <div class="page-title">Brain MRI Analysis</div>
    <div class="page-subtitle">
        Upload a brain MRI image to explore the classification output
        from your locally running deep learning model.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")




# --------------------------------------------------
# UPLOAD + RESULTS LAYOUT
# --------------------------------------------------

upload_col, result_col = st.columns(
    [1, 1.12],
    gap="large",
)


with upload_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-heading">MRI Image</div>
            <div class="section-description">
                Upload a PNG or JPEG image to begin your analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose MRI image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="mri_uploader",
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            st.image(
                image,
                caption="Uploaded MRI image",
                use_container_width=True,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="metric-label">FILE DETAILS</div>
                    <div style="font-size:14px;font-weight:700;color:#1D2939;
                                overflow-wrap:anywhere;">
                        {uploaded_file.name}
                    </div>
                    <div class="metric-note">
                        Image dimensions: {image.width} × {image.height} px
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Image preview only. Image quality and format may affect "
                "model output."
            )

        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            image = None
    else:
        image = None

        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🧠</div>
                <strong style="color:#1D2939;">No MRI image selected</strong><br>
                Choose an image using the upload control above.
                <br><br>
                <span style="font-size:12px;">
                    Supported formats: PNG, JPG, JPEG
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


with result_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-heading">Classification Results</div>
            <div class="section-description">
                Model prediction and class-wise confidence scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "Analyze MRI",
        type="primary",
        disabled=uploaded_file is None or image is None,
        use_container_width=True,
    )

    if uploaded_file is None:
        st.session_state.pop("mri_result", None)

    
if analyze_clicked and image is not None:
    try:
        with st.spinner("Analyzing MRI and generating Grad-CAM..."):
            model = load_model()
            input_tensor = preprocess(image).unsqueeze(0)

            # Generate the Grad-CAM overlay and obtain the predicted class.
            overlay = generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                original_image=image,
            )

            # Calculate the displayed class confidence scores.
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

            st.session_state["mri_result"] = {
                "filename": uploaded_file.name,
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": float(
                    probabilities[predicted_index].item()
                ),
                "probabilities": probabilities.tolist(),
                "gradcam_overlay": overlay,
            }

    except Exception as exc:
        st.error(f"Model inference failed: {exc}")
        st.session_state.pop("mri_result", None)

    result = st.session_state.get("mri_result")

    if result is not None:
        if result["filename"] != uploaded_file.name:
            st.session_state.pop("mri_result", None)
            result = None

    if result is not None:
        st.markdown(
            '<div class="status-pill">✓ &nbsp; Inference completed</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-card-label">Predicted class</div>
                <div class="result-card-value">
                    {result["predicted_class"]}
                </div>
                <div class="metric-note">
                    Highest-scoring category produced by the model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Top class score",
                f'{result["confidence"]:.1%}',
            )

        with metric_col2:
            st.metric(
                "Categories evaluated",
                len(CLASS_NAMES),
            )

        st.write("")

        st.markdown(
            '<div class="section-heading">Class confidence scores</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-description">
                Scores are softmax outputs; they are not necessarily
                calibrated probabilities of a medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for name, score in zip(
            CLASS_NAMES,
            result["probabilities"],
        ):
            label_col, score_col = st.columns([3, 1])

            with label_col:
                st.markdown(
                    f'<div style="font-size:13px;font-weight:600;'
                    f'color:#344054;margin-top:7px;">{name}</div>',
                    unsafe_allow_html=True,
                )

            with score_col:
                st.markdown(
                    f'<div style="text-align:right;font-size:13px;'
                    f'font-weight:700;color:#1D2939;margin-top:7px;">'
                    f'{score:.1%}</div>',
                    unsafe_allow_html=True,
                )

            st.progress(float(score))

        st.markdown(
            """
            <div class="confidence-note">
                A high score does not establish the presence or absence
                of a tumor. Results require independent validation and
                professional medical interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="empty-state" style="margin-top:18px;">
                <div class="empty-icon">📈</div>
                <strong style="color:#1D2939;">Your results will appear here</strong>
                <br>
                Upload an MRI image, then select
                <strong>Analyze MRI</strong>.
                <br><br>
                <span style="font-size:12px;">
                    Inference runs locally on the CPU.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# GRAD-CAM VISUALIZATION
# --------------------------------------------------

result = st.session_state.get("mri_result")

if result is not None:
    if (
        uploaded_file is not None
        and result["filename"] == uploaded_file.name
    ):
        st.write("")
        st.markdown(
            """
            <div class="panel">
                <div class="section-heading">
                    Model Attention (Grad-CAM)
                </div>
                <div class="section-description">
                    A visualization of image regions that influenced
                    the selected class score.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        original_col, heatmap_col = st.columns(2, gap="large")

        with original_col:
            st.markdown("#### Original MRI")
            st.image(
                image,
                use_container_width=True,
            )

        with heatmap_col:
            st.markdown("#### Grad-CAM Overlay")
            st.image(
                result["gradcam_overlay"],
                use_container_width=True,
            )

        st.caption(
            "Interpretation note: Grad-CAM is a coarse model-explanation "
            "visualization. Highlighted areas are not a tumor segmentation "
            "or a validated medical explanation."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <strong>NeuroScan</strong> · Brain MRI Classification Prototype
        <br>
        ResNet-50 · PyTorch · Streamlit · Local CPU inference
        <br><br>
        For educational and research prototyping only. Not intended
        for clinical diagnosis or treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
