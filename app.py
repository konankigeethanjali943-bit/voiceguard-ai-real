import streamlit as st
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

st.set_page_config(page_title="VoiceGuard AI", layout="centered")
st.title("🛡️ VoiceGuard AI - SIH26104")
st.caption("Real AI Model - Wav2Vec2 Deepfake Detection")

@st.cache_resource
def load_model():
    model_id = "mo-thecreator/Deepfake-audio-detection"
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    return extractor, model

extractor, model = load_model()

uploaded = st.file_uploader("Upload Voice File", type=["wav","mp3","m4a","ogg"])

if uploaded:
    st.audio(uploaded)
    if st.button("🔍 REAL AI ANALYSIS"):
        with st.spinner("AI Analyzing..."):
            y, sr = librosa.load(uploaded, sr=16000)
            inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                fake_prob = float(probs[0]) if 'fake' in model.config.id2label[0].lower() else float(probs[1])
                if fake_prob < 0.3: # safety fix
                    fake_prob = 1 - fake_prob if probs[0] > 0.5 else fake_prob

            score = int(fake_prob*100)
            st.metric("CLONED RISK", f"{score}%")
            if score > 60:
                st.error(f"🚨 CLONED VOICE - {score}%")
            else:
                st.success(f"✅ ORIGINAL VOICE - {100-score}% Real")
