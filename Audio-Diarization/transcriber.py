import whisper
import torch

def transcribe_audio(audio_path: str, language: str = None, model_size: str = "medium", device: str = None):
    """Transcribes the audio file using OpenAI Whisper."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for transcription")

    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Error loading Whisper model '{model_size}' on device '{device}': {e}")
        return None

    print(f"Starting transcription for {audio_path} using model '{model_size}'.")
    use_fp16 = device == "cuda"
    if language and language != "Auto Detect":
        result = model.transcribe(audio_path, language=language, fp16=use_fp16)
    else:
        result = model.transcribe(audio_path, fp16=use_fp16)

    print("Transcription complete.")
    return result["segments"] # Return list of segment dictionaries 