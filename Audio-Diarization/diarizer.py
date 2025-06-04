from pyannote.audio import Pipeline
import os

def diarize_audio(audio_path: str, hf_token: str):
    """Performs speaker diarization on the audio file."""
    if not hf_token:
        print("Hugging Face token not found.")
        return []

    print(f"Loading diarization pipeline...")
    pipeline = None
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1.1",
                                            use_auth_token=hf_token)
        print("Pipeline loaded.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Check Hugging Face token and model access.")
        return []

    if pipeline is None:
         print("Pipeline failed to load.")
         return []

    print(f"Starting diarization for {audio_path}")
    try:
        diarization = pipeline(audio_path)
        print("Diarization complete.")

        diarization_segments = []
        print("Processing results...")
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end
            })
        print(f"Found {len(diarization_segments)} segments.")
        return diarization_segments

    except Exception as e:
        print(f"Error during diarization: {e}")
        return [] 