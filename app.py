import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown
import zipfile

st.set_page_config(
    page_title="AI Dental Diagnosis",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

background_image_url = "https://raw.githubusercontent.com/Reem-Albadwy/Dental_Diagnosis_App/main/background2.png"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url('{background_image_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

h1, h2, h4, p {{
    color: #1c1c3b;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
}}

.result-card {{
    background: linear-gradient(135deg, #92c6f9 0%, #d6e6f2 100%);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    text-align: center;
    margin-top: 30px;
    color: #1c1c3b;
}}


.other-predictions {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-top: 30px;
    color: #1c1c3b;
}}

div.stButton > button {{
    background-color: #1c1c3b;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.6em 1.4em;
    font-size: 16px;
    border: none;
    transition: all 0.3s ease-in-out;
}}

div.stButton > button:hover {{
    background-color: #003366;
    transform: scale(1.03);
}}

section[data-testid="stFileUploader"] {{
    border: 2px dashed #1c1c3b;
    border-radius: 12px;
    background-color: rgba(255, 255, 255, 0.75);
    padding: 1.5em;
}}

section[data-testid="stFileUploader"] label {{
    color: #1c1c3b;
    font-size: 18px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1>🦷 AI Dental Diagnosis</h1>", unsafe_allow_html=True)
st.markdown("<h4>Upload a dental image to receive an instant prediction</h4>", unsafe_allow_html=True)

model_url = "https://drive.google.com/uc?export=download&id=1Ddqk-r3RJjh2-hBKqtQjNyJmIb0JP7vY"
model_zip_path = "Xception_FineTuned_Model.zip"
model_folder = "Xception_FineTuned_Model"

@st.cache_resource
def load_model():
    model_dir = "Xception_FineTuned_Model"
    zip_path = "Xception_FineTuned_Model.zip"

    if not os.path.exists(model_dir):
        url = "https://drive.google.com/uc?id=1Ddqk-r3RJjh2-hBKqtQjNyJmIb0JP7vY"
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

    inner_dir = os.path.join(model_dir, "Xception_FineTuned_Model")
    if os.path.exists(os.path.join(inner_dir, "saved_model.pb")):
        return tf.keras.models.load_model(inner_dir)
    else:
        return tf.keras.models.load_model(model_dir)

model = load_model()
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

advice_dict = {
    'CaS': "Calcium buildup requires regular brushing and a dental cleaning.",
    'CoS': "See your dentist for signs of cysts and monitor any swelling.",
    'Gum': "Gum issues need gentle flossing and possibly an antibacterial rinse.",
    'MC': "Maintain good oral hygiene and avoid sugary foods to prevent cavities.",
    'OC': "Oral cancer needs immediate professional evaluation. Please consult a specialist.",
    'OLP': "Oral Lichen Planus requires monitoring and stress management.",
    'OT': "Oral trauma should be managed by avoiding further irritation and visiting a dentist."
}

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    
    infer = model.signatures["serving_default"]
    img = image.convert("RGB").resize((128, 128))  
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)

    output = infer(img_tensor)
    preds = list(output.values())[0].numpy()[0]
    sorted_preds = sorted(zip(class_names, preds), key=lambda x: x[1], reverse=True)
    main_label, main_prob = sorted_preds[0]

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h2>🦷 Primary Diagnosis: <span style='color:#003049'>{main_label}</span></h2>
            <p style="font-size:18px;">Confidence: {main_prob*100:.2f}%</p>
            <p style="font-size:16px;"><strong>Advice:</strong> {advice_dict.get(main_label, "Keep good oral hygiene and visit your dentist regularly.")}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        with st.expander("Other Probalities", expanded=True):
            for label, prob in sorted_preds[1:]:
                st.markdown(f"<p style='text-align:center;'><strong>{label}</strong>: {prob*100:.2f}%</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
   
