import streamlit as st
import requests

# ----------------------------
# 🔑 Hugging Face API Key
# ----------------------------
HF_TOKEN = "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"   # ⬅️ replace with your key
MODEL_ID = "ibm-granite/granite-speech-3.3-2b"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ----------------------------
# 🎤 Streamlit UI
# ----------------------------
st.title("🎧 Granite Speech-to-Text (Hugging Face API)")
st.write("Upload a WAV/MP3 audio file and get transcription using IBM Granite model.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    # Show audio player in Streamlit
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Transcribe"):
        with st.spinner("Transcribing... ⏳"):
            # Send audio to Hugging Face Inference API
            response = requests.post(API_URL, headers=HEADERS, data=uploaded_file.read())

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "text" in result[0]:
                    st.success("✅ Transcription Complete!")
                    st.write(result[0]["text"])
                else:
                    st.error("Unexpected response format.")
                    st.json(result)
            else:
                st.error(f"❌ Error {response.status_code}")
                st.text(response.text)





import os
import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import asyncio
import tempfile
from pathlib import Path
import edge_tts

# ---------------------
# API KEY HANDLING
# ---------------------
st.title("📚 Multilingual Audiobook Generator")

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    hf_key = st.text_input("🔑 Enter Hugging Face API key:", type="password")
    if hf_key:
        os.environ["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"] = hf_key

HF_API_KEY = os.getenv("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", None)
if not HF_API_KEY:
    st.error("❌ Missing Hugging Face API key. Please enter it above.")
    st.stop()

# ---------------------
# MODELS
# ---------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token="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")

# MarianMT translation models (English <-> many langs)
MARIAN_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
}

# Edge voices (per language)
SUPPORTED_LANGS = {
    "en": ("English", "en-US-AriaNeural", "en-US-GuyNeural"),
    "fr": ("French", "fr-FR-DeniseNeural", "fr-FR-HenriNeural"),
    "de": ("German", "de-DE-KatjaNeural", "de-DE-ConradNeural"),
    "es": ("Spanish", "es-ES-ElviraNeural", "es-ES-AlvaroNeural"),
    "hi": ("Hindi", "hi-IN-SwaraNeural", "hi-IN-MadhurNeural"),
}

# ---------------------
# TRANSLATION HELPER
# ---------------------
def translate_text(text, src_lang, tgt_lang):
    if tgt_lang == "en":
        return text
    model_name = MARIAN_MODELS.get(tgt_lang)
    if not model_name:
        return text
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ---------------------
# TTS HELPER
# ---------------------
async def synthesize_edge_tts(text, voice, out_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)

# ---------------------
# STREAMLIT UI
# ---------------------
text_input = st.text_area("Paste your book text here:", height=200)

col1, col2 = st.columns(2)
with col1:
    do_summary = st.checkbox("Summarize before audiobook", value=True)
with col2:
    target_lang = st.selectbox("Target Language", list(SUPPORTED_LANGS.keys()), format_func=lambda x: SUPPORTED_LANGS[x][0])

voice_style = st.radio("Voice Style", ["Female", "Male"], horizontal=True)

if st.button("🎧 Generate Audiobook"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
        st.stop()

    st.info("Summarizing text..." if do_summary else "Using full text...")

    if do_summary:
        summary = summarizer(text_input, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]
    else:
        summary = text_input

    st.success("Summary/Content ready ✅")

    st.info(f"Translating to {SUPPORTED_LANGS[target_lang][0]}...")
    final_text = translate_text(summary, "en", target_lang)

    st.success("Translation ready ✅")
    st.write(final_text)

    _, female_voice, male_voice = SUPPORTED_LANGS[target_lang]
    voice = female_voice if voice_style == "Female" else male_voice

    st.info("Generating speech...")
    with tempfile.TemporaryDirectory() as td:
        out_mp3 = os.path.join(td, "audiobook.mp3")

        try:
            asyncio.run(synthesize_edge_tts(final_text, voice, out_mp3))
        except RuntimeError:
            # Fix for nested event loop (e.g. in Jupyter/Streamlit Cloud)
            loop = asyncio.get_event_loop()
            task = loop.create_task(synthesize_edge_tts(final_text, voice, out_mp3))
            loop.run_until_complete(task)

        audio_bytes = Path(out_mp3).read_bytes()
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button("💾 Download Audiobook", data=audio_bytes, file_name="audiobook.mp3", mime="audio/mpeg")

st.caption("Powered by Hugging Face Transformers + Edge TTS")
