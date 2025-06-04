import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import json
from datetime import timedelta
import torch
from huggingface_hub import login

from transcriber import transcribe_audio
from diarizer import diarize_audio
from audio_utils import convert_audio

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Log in to Hugging Face
if HUGGINGFACE_TOKEN:
    try:
        login(token=HUGGINGFACE_TOKEN)
    except Exception as e:
        st.sidebar.error(f"Hugging Face login failed: {e}")
else:
    st.sidebar.warning("Hugging Face token not found.")

st.title("Audio Transcription and Diarization")

def process_audio(audio_file, language=None, model_size="tiny", device="cpu"):
    """Processes the audio file for transcription and diarization."""
    if not HUGGINGFACE_TOKEN:
        st.error("Hugging Face token not found. Diarization requires authentication.")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        original_audio_path = tmp_file.name

    with tempfile.TemporaryDirectory() as tmp_dir:
        converted_audio_path = convert_audio(original_audio_path, tmp_dir)

        if not converted_audio_path:
            st.error("Audio conversion failed.")
            os.unlink(original_audio_path)
            return None

        st.subheader("Transcription")
        transcript_segments = transcribe_audio(converted_audio_path, language, model_size=model_size, device=device)

        if not transcript_segments:
            st.warning("Transcription did not return any segments.")
            os.unlink(original_audio_path)
            return None

        st.subheader("Speaker Diarization")
        diarization_segments = diarize_audio(converted_audio_path, HUGGINGFACE_TOKEN)

        os.unlink(original_audio_path)

        if not diarization_segments:
            st.warning("Diarization did not return any segments.")
            st.subheader("Transcript (Diarization Failed)")
            for segment in transcript_segments:
                 st.write(f"[{str(timedelta(seconds=segment['start'])):<12} - {str(timedelta(seconds=segment['end'])):<12}] {segment['text']}")
            return None

        st.subheader("Aligned Transcript")
        aligned_transcript = []
        diarization_index = 0

        for transcript_segment in transcript_segments:
            t_start = transcript_segment['start']
            t_end = transcript_segment['end']
            t_text = transcript_segment['text'].strip()

            current_speakers = set()

            for i in range(diarization_index, len(diarization_segments)):
                di_start = diarization_segments[i]['start']
                di_end = diarization_segments[i]['end']
                speaker = diarization_segments[i]['speaker']

                if max(t_start, di_start) < min(t_end, di_end):
                    current_speakers.add(speaker)
                    if di_end > t_end:
                        diarization_index = i
                        break
                    else:
                        diarization_index = i + 1
                elif di_start >= t_end:
                    break

            speaker_label = ", ".join(sorted(list(current_speakers))) if current_speakers else "Unknown Speaker"

            aligned_transcript.append({
                "speaker": speaker_label,
                "start": str(timedelta(seconds=t_start)),
                "end": str(timedelta(seconds=t_end)),
                "text": t_text
            })
            st.write(f"**{speaker_label}**: [{str(timedelta(seconds=t_start)):<12} - {str(timedelta(seconds=t_end)):<12}] {t_text}")

        st.subheader("JSON Output")
        json_output = json.dumps(aligned_transcript, indent=2)
        st.json(json_output)

        st.download_button(
            label="Download JSON",
            data=json_output,
            file_name="transcript.json",
            mime="application/json"
        )


# Main Streamlit app logic
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    languages = {
        "Auto Detect": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Russian": "ru",
        "Arabic": "ar",
        "Portuguese": "pt"
    }
    selected_language_name = st.selectbox("Select Language", list(languages.keys()))
    selected_language_code = languages[selected_language_name]

    # Defaulting to 'tiny' and 'cpu'
    selected_model_size = "tiny"
    selected_device = "cpu"

    if st.button("Process Audio"):
        with st.spinner("Processing audio..."):
            process_audio(uploaded_file, selected_language_code, model_size=selected_model_size, device=selected_device) 