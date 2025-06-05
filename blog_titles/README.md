# AI-Powered Blog Title Suggester

A Django + DRF application that takes raw blog content (HTML or plain text), splits it into overlapping chunks, retrieves the most relevant pieces via FAISS embeddings, summarizes them using a Hugging Face model (on GPU if available), combines those summaries into a global overview via an LLM (Ollama), and finally generates three catchy title suggestions (also via Ollama).

---

## Features

* **Recursive Character Splitting** into 1 024-character windows (with overlap) so each chunk fits a typical 1 024-token limit.
* **FAISS Retrieval**: Embed and index chunks with `OllamaEmbeddings`, then retrieve the top 20 most relevant chunks for summarization.
* **Fast Summarization**: Use Hugging Face’s DistilBART on GPU (if available) to summarize each chunk.
* **Global Summary & Title Generation**:

  * Combine chunk summaries into a single overall summary using `llama3:instruct` (Ollama).
  * Generate three distinct, engaging titles from that summary via `llama3:instruct` (Ollama).
* **GPU-Aware**: Summarization automatically uses CUDA if `torch.cuda.is_available()`; Ollama will use GPU (fp16) if the Ollama server is launched with `--gpu` and environment variables are set.

---

## Repository Structure

```
blog_titles/
├── blog_titles/               # Django project (settings, URLs)
├── suggestions/               # DRF app (“suggest-titles/” endpoint)
│   ├── serializers.py         # BlogContentSerializer
│   ├── urls.py                # Route for TitleSuggestionAPIView
│   ├── utils.py               # Core pipeline (clean → chunk → embed → summarize → titles)
│   └── views.py               # TitleSuggestionAPIView implementation
├── manage.py
├── requirements.txt
└── README.md
```

---

## Prerequisites

1. **Python 3.8+**
2. **Git** (to clone the repository)
3. **CUDA-capable NVIDIA GPU** (optional, for faster summarization)
4. **Ollama** (for `llama3:instruct` inference)

   * Install via Homebrew (`brew install ollama`) on macOS or follow [https://ollama.com](https://ollama.com) for Windows/Linux.
   * Pull and serve the model:

     ```bash
     ollama pull llama3:instruct
     ollama serve --gpu
     ```

     If you do not have a GPU or prefer CPU inference, omit `--gpu`:

     ```bash
     ollama serve
     ```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/blog-title-suggester.git
   cd blog-title-suggester
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Install PyTorch with CUDA support**
   If you have CUDA 11.8 drivers, for GPU summarization run:

   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Verify in Python:

   ```python
   import torch
   print(torch.cuda.is_available())  # should be True if GPU is enabled
   ```

5. **Start the Ollama server**

   ```bash
   ollama pull llama3:instruct       # if not already pulled
   ollama serve --gpu                # for GPU inference
   ```

   If you want CPU‐only Ollama:

   ```bash
   ollama serve
   ```

6. **Set environment variables** (create a `.env` or export directly):

   ```
   USE_OLLAMA_GPU=1
   OLLAMA_DEFAULT_DEVICE=cuda
   OLLAMA_DEFAULT_PRECISION=fp16
   ```

---

## Running the Django API

1. **Apply migrations**

   ```bash
   python manage.py migrate
   ```

2. **Create a superuser** (optional)

   ```bash
   python manage.py createsuperuser
   ```

3. **Run the development server**

   ```bash
   python manage.py runserver
   ```

   You should see:

   ```
   Watching for file changes with StatReloader
   Performing system checks...

   System check identified no issues (0 silenced).
   Starting development server at http://127.0.0.1:8000/
   ```

---

## Environment Variables

* **USE\_OLLAMA\_GPU**

  * Set to `1` → instructs ChatOllama to run on GPU (requires Ollama server with `--gpu`).
  * Set to `0` or unset → ChatOllama uses CPU.

* **OLLAMA\_DEFAULT\_DEVICE** / **OLLAMA\_DEFAULT\_PRECISION**

  * Typically set to `cuda` and `fp16`, respectively, so that Ollama‐hosted models run in fp16 on GPU. Only effective if Ollama is serving on GPU.

---

## API Usage

### Endpoint

```
POST http://127.0.0.1:8000/api/suggest-titles/
Content-Type: application/json

{
  "content": "<raw blog post HTML or plain text>"
}
```

* **Request Body**

  * `content` (string, required): Full blog post text or HTML.

* **Response (200 OK)**

  ```json
  {
    "titles": [
      "First Suggested Title",
      "Second Suggested Title",
      "Third Suggested Title"
    ]
  }
  ```

* **Error Responses**

  * `400 Bad Request` if `content` is missing or invalid.
  * `500 Internal Server Error` if any pipeline step fails—response body will be:

    ```json
    {
      "error": "Detailed error message"
    }
    ```

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/suggest-titles/ \
     -H "Content-Type: application/json" \
     -d '{ "content": "<h1>My Blog</h1><p>This is my blog post content…</p>" }'
```

---

## Troubleshooting

* **“Torch not compiled with CUDA enabled”**

  1. Uninstall CPU‐only PyTorch:

     ```bash
     pip uninstall torch torchvision torchaudio
     ```
  2. Reinstall GPU‐enabled PyTorch (e.g., for CUDA 11.8):

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
  3. Verify in Python:

     ```python
     import torch
     print(torch.cuda.is_available())  # should be True
     ```

* **Ollama Connection Errors**

  * Ensure you started Ollama with `ollama serve --gpu` (if `USE_OLLAMA_GPU=1`).
  * If you only have CPU Ollama, set `USE_OLLAMA_GPU=0` (or unset) in your environment and restart Django.

* **Chunk Too Long / Summarizer Errors**

  * If you see errors about token length, reduce `chunk_size` (e.g. to 800) in `get_recursive_chunks()` so each chunk is well under 1 024 characters.

* **404 on `/api/suggest-titles/`**

  * Confirm that `suggestions/urls.py` is included in the project’s `urls.py`:

    ```python
    path("api/", include("suggestions.urls")),
    ```

---

## License

This project is MIT-licensed. See [LICENSE](LICENSE) for details.
