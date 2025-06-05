
# Audio Diarization

This folder contains a simple Python-based pipeline to transcribe an audio file and perform speaker diarization (i.e., identify “who spoke when”). The output is a timestamped, speaker-labeled JSON.

## Repository Structure

````

Audio-Diarization/
├── requirements.txt
├── diarize.py          # Main script for transcription + diarization
├── transcribe.py       # (Optional) Separate transcription module
└── README.md
````

## Clone the Repository

To clone the **entire** Darwix-AI-Assessment repo and navigate into this folder, run:

```bash
git clone https://github.com/zuse1425/Darwix-AI-Assessment.git
cd Darwix-AI-Assessment/Audio-Diarization
````

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

2. Install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** `requirements.txt` includes the versions of `openai-whisper`, `pyannote.audio`, `ffmpeg-python`, etc., needed for both transcription and speaker diarization.

## Usage

1. Place your input audio file (e.g. `input.wav`) in this folder (or supply a path).

2. Run the main diarization script:

   ```bash
   python diarize.py --input input.wav --output result.json
   ```

   * `--input`: Path to your audio file (WAV, MP3, etc.).
   * `--output`: Path where the JSON with speaker labels will be written.

3. Examine `result.json`, which will look something like:

   ```json
   [
     {
       "speaker": "Speaker 1",
       "start_time": 0.00,
       "end_time": 3.45,
       "transcription": "Hello, everyone. Today we’re discussing..."
     },
     {
       "speaker": "Speaker 2",
       "start_time": 3.45,
       "end_time": 7.80,
       "transcription": "Thanks for having me. I’d like to add that..."
     }
   ]
   ```

## (Optional) Separate Transcription Only

You can also run transcription without diarization:

```bash
python transcribe.py --input input.wav --output transcript.txt
```

This writes a plain text transcript (no speaker labels).

## Troubleshooting

* If you see errors about missing PyTorch or CUDA, make sure you have a compatible version installed. By default, we assume CPU-only unless you have the CUDA-enabled `torch` package.
* On Windows, ensure `ffmpeg` is in your PATH. On macOS/Linux, you can install via Homebrew or `apt-get`.

