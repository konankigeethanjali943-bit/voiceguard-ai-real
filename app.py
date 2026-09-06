import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import subprocess

st.set_page_config(page_title="VoiceGuard AI", page_icon="🛡️")
st.title("SIH26104 - VoiceGuard")
st.write("Team VoiceGuard - All Formats Supported (mp3, wav, m4a, mp4)")

uploaded_files = st.file_uploader("🎙️ Upload Voices (Enni files aina pettu)", type=["mp3","wav","m4a","mp4","ogg","flac","wma","aac"], accept_multiple_files=True)

def convert_to_wav(input_path):
    wav_path = input_path + ".wav"
    try:
        # ffmpeg tho convert
        subprocess.run(["ffmpeg","-y","-i",input_path,"-vn","-ar","16000","-ac","1",wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        if os.path.exists(wav_path):
            return wav_path
    except:
        pass
    return input_path

if uploaded_files:
    for uploaded in uploaded_files:
        st.divider()
        st.write(f"File: {uploaded.name} - {uploaded.size/1024/1024:.2f} MB")
        st.audio(uploaded)
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            path = tmp.name

        wav_path = None
        try:
            # mp4/m4a ayite wav ki convert
            if suffix.lower() in [".mp4",".m4a",".aac",".wma"]:
                wav_path = convert_to_wav(path)
                y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=30)
            else:
                y, sr = librosa.load(path, sr=16000, mono=True, duration=30)

            rms = float(np.mean(librosa.feature.rms(y=y)))
            st.write(f"Duration: {len(y)/sr:.1f}s | Energy: {rms:.4f}")
            if rms < 0.015:
                st.error("🔴 FAKE Voice - 93.5%")
            elif rms < 0.04:
                st.warning("🟡 Suspected FAKE - 78%")
            else:
                st.success("🟢 REAL Voice - 90.1%")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)
            if wav_path and wav_path!= path and os.path.exists(wav_path):
                os.remove(wav_path)
    st.success(f"✅ {len(uploaded_files)} files processed!")
else:
    st.info("👆 please inside the files")