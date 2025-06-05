# AI-Powered Blog Title Suggester

A Django + DRF service that generates three catchy blog post title suggestions from raw content (HTML or plain text). The pipeline:

1. **Clean & Preprocess**: Strip HTML, remove URLs/markdown fences, normalize whitespace.
2. **Chunking**: Split cleaned text into overlapping 1 024-character windows.
3. **FAISS Retrieval**: Embed each chunk with OllamaEmbeddings, index via FAISS, retrieve the top 20 most relevant chunks.
4. **Chunk Summarization**: Run each selected chunk through Hugging Face’s DistilBART (on GPU if available).
5. **Global Summary**: Combine chunk summaries into one paragraph using a `llama3:instruct` model (via Ollama).
6. **Title Generation**: Produce exactly three distinct, engaging titles from the global summary using `llama3:instruct` (via Ollama).

---

## Features

* **Overlapping Character Splitting**: Ensures each chunk is ≤ 1 024 characters (approximate 1 024-token limit) with 200 characters of overlap for context continuity.
* **FAISS-Based Relevance**: Embeds chunks using OllamaEmbeddings and retrieves the 20 most relevant chunks for summarization.
* **GPU-Accelerated Summarization**: Uses DistilBART on GPU (if PyTorch+CUDA is available) for fast, high-quality chunk summaries.
* **LLM-Powered Combining & Titles**:

  * **Global Summary**: Merges mini-summaries into a coherent overview via `llama3:instruct` (temperature 0.5).
  * **Title Suggestions**: Generates three distinct titles from that overview via `llama3:instruct` (temperature 0.8).
* **GPU & Ollama Integration**:

  * Summarization auto-detects CUDA and runs on GPU if available.
  * Ollama’s `llama3:instruct` models run in fp16 on GPU if the server is started with `--gpu` and environment variables are set.

---

## Tech Stack

* **Python 3.8+**
* **Django 4.2+ & Django REST Framework**
* **LangChain** (`langchain`, `langchain-ollama`) for embeddings, chat integration, and utility functions
* **Ollama** (local llama3\:instruct) for combining summaries and generating titles
* **Hugging Face Transformers** (`transformers`, `torch`) for DistilBART summarization
* **FAISS** (`faiss-cpu`) for nearest-neighbor retrieval of chunk embeddings
* **BeautifulSoup4** for HTML cleaning
* **python-dotenv** for environment variable management

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/blog-title-suggester.git
   cd blog-title-suggester
   ```

2. **Create & Activate a Virtual Environment**

   ```bash
   python -m venv venv
   # macOS/Linux:
   source venv/bin/activate
   # Windows (PowerShell):
   venv\Scripts\activate
   ```

3. **Install Python Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Install PyTorch with CUDA**
   If you have a CUDA-compatible GPU and want summarization on GPU, reinstall PyTorch accordingly. For CUDA 11.8:

   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Verify in Python:

   ```python
   import torch
   print(torch.cuda.is_available())  # should be True
   ```

5. **Install & Start the Ollama Server**

   * Pull the llama3\:instruct model:

     ```bash
     ollama pull llama3:instruct
     ```
   * Start Ollama with GPU support (if available) or CPU:

     ```bash
     ollama serve --gpu
     ```

     Or, for CPU only:

     ```bash
     ollama serve
     ```

6. **Set Environment Variables**
   Create a file named `.env` in the project root (or export them in your shell):

   ```
   USE_OLLAMA_GPU=1
   OLLAMA_DEFAULT_DEVICE=cuda
   OLLAMA_DEFAULT_PRECISION=fp16
   ```

   * `USE_OLLAMA_GPU=1` tells ChatOllama to request GPU from Ollama.
   * `OLLAMA_DEFAULT_DEVICE=cuda` and `OLLAMA_DEFAULT_PRECISION=fp16` ensure Ollama runs in fp16 on GPU.

7. **Apply Migrations & Run the Server**

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

   The API will be available at `http://127.0.0.1:8000/`.

---

## Project Structure

```
blog_titles/
├── blog_titles/               # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── suggestions/               # DRF app for title suggestions
│   ├── serializers.py         # BlogContentSerializer (validates "content")
│   ├── urls.py                # "suggest-titles/" endpoint
│   ├── utils.py               # Pipeline: clean → chunk → embed → summarize → titles
│   └── views.py               # TitleSuggestionAPIView implementation
├── manage.py
├── requirements.txt
└── README.md
```

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

  * `content` (string, required): Full blog post (HTML allowed).

* **Success Response (200 OK)**

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
  * `500 Internal Server Error` if any processing step fails. Response format:

    ```json
    {
      "error": "Detailed error message"
    }
    ```

### Example Curl Request

```bash
curl -X POST http://127.0.0.1:8000/api/suggest-titles/ \
     -H "Content-Type: application/json" \
     -d '{ "content": "<h1>My Blog</h1><p>This is my blog post content…</p>" }'
```
---

## License

This project is MIT-licensed. See [LICENSE](LICENSE) for details.
