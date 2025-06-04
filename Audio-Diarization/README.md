# Audio Transcription and Diarization App

This is a simple web application built with Streamlit that transcribes audio files and performs speaker diarization to identify who spoke when.

## Features

- Upload audio files in various formats (.wav, .mp3, .m4a, .ogg, .flac).
- Transcribe audio using OpenAI Whisper.
- Perform speaker diarization using pyannote.audio.
- Auto-detect language or select a specific language for transcription.
- **Note:** Accurate transcription and alignment is primarily tested and expected for the languages available in the language dropdown (English, Spanish, French, German, Chinese, Japanese, Russian, Arabic, Portuguese). While Whisper can handle other languages, they may be transcribed into English.
- Display the transcript aligned with speaker labels.
- Provide a downloadable JSON output of the aligned transcript.

## Tech Stack

- Python 3.x
- Streamlit (for the web UI)
- `openai-whisper` (for transcription)
- `pyannote.audio` (for speaker diarization)
- `ffmpeg` (for audio processing)
- `pydub` (for audio handling)
- `torch`, `torchaudio` (dependencies for Whisper and pyannote.audio)
- `huggingface_hub`, `python-dotenv` (for Hugging Face model access)
- `numpy`, `json`, `datetime`
- `plotly`, `pandas` (included but visualization feature currently hidden in UI)

## Setup Instructions

Follow these steps to set up and run the project locally.

1.  **Clone the repository:**

    If you haven't already, clone the project from GitHub:

    ```bash
    git clone https://github.com/abhiXhell/Audio-Diarization.git
    cd Audio-Diarization
    ```

2.  **Create and activate a virtual environment:**

    It's recommended to use a virtual environment to manage project dependencies.

    ```bash
    python3 -m venv venv
    ```

    Activate the virtual environment:

    *   On Linux/macOS:

        ```bash
        source venv/bin/activate
        ```

    *   On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**

    With the virtual environment activated, install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Token and Model Access:**

    `pyannote.audio` requires access to models hosted on Hugging Face. You'll need a Hugging Face account and an access token.

    *   Go to [https://huggingface.co/](https://huggingface.co/) and create an account.
    *   Accept the user agreement for the necessary models:
        *   [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
        *   [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
    *   Generate an access token from your settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Give it a name and select the 'read' role).
    *   Create a file named `.env` in the project's root directory and add the following line, replacing `YOUR_HUGGINGFACE_TOKEN` with your actual token:

        ```
        HUGGINGFACE_TOKEN=YOUR_HUGGINGFACE_TOKEN
        ```

5.  **Install FFmpeg:**

    Ensure FFmpeg is installed on your system. If you are on Debian/Ubuntu Linux, you can install it using:

    ```bash
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    ```

    For other operating systems, please refer to the official FFmpeg documentation for installation instructions.

## How to Run the Application

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit application from the project's root directory:

    ```bash
    streamlit run app.py
    ```

3.  The application should open in your web browser.

## Project Structure

- `app.py`: The main Streamlit application file.
- `transcriber.py`: Handles audio transcription using OpenAI Whisper.
- `diarizer.py`: Handles speaker diarization using pyannote.audio.
- `audio_utils.py`: Utility functions for audio file conversion.
- `requirements.txt`: Lists project dependencies.
- `.env`: Stores the Hugging Face token (ignored by Git).
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Potential Improvements (Bonus Ideas)

- Add a visual timeline for speaker diarization in the UI (Plotly code is already partially included).
- Implement more robust error handling and user feedback in the UI.
- Allow selection of different Whisper model sizes and devices in the UI.
- Improve the alignment logic for more complex audio scenarios.
- Add more language options. 