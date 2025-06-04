from pydub import AudioSegment
import os

def convert_audio(input_path: str, output_dir: str) -> str:
    """Converts audio to 16kHz mono WAV format."""
    output_path = os.path.join(output_dir, "converted_audio.wav")

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio.export(output_path, format="wav")
        print(f"Converted audio to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None 