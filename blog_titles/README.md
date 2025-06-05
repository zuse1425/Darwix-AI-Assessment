
# Blog Post Title Suggestions

This folder contains a simple Python script to generate AI-powered blog post title suggestions given a topic or seed phrase. It uses OpenAI’s API (or another large-language-model endpoint) to propose multiple catchy headlines.

## Repository Structure

````

blog_titles/
├── blog_titles/
├── generate_titles/ 
├── manage.py
├── requirements.txt
└── README.md


````

## Clone the Repository

To clone the **entire** Darwix-AI-Assessment repo and navigate into this folder, run:

```bash
git clone https://github.com/zuse1425/Darwix-AI-Assessment.git
cd Darwix-AI-Assessment/blog_titles
````

## Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** `requirements.txt` typically includes `openai>=0.27.0` (or any other LLM client library you’re using), plus any helpers such as `python-dotenv` if you load API keys from a `.env` file.

3. Make sure you have a valid `OPENAI_API_KEY` (or equivalent) in your environment:

   ```bash
   export OPENAI_API_KEY="your_key_here"    # macOS/Linux
   set OPENAI_API_KEY="your_key_here"       # Windows (cmd)
   ```

## Usage

1. Run the title generator script with a topic string:

   ```bash
   python generate_titles.py --topic "How to Implement RAG in Django"
   ```

2. By default, it prints out 3 suggestions to stdout. Example output:

   ```
   1. “Supercharge Your Django App with a RAG-Powered Chatbot”
   2. “Step-by-Step: Building a Django RAG Pipeline for Smart Q&A”
   3. “RAG in Django: A Beginner’s Guide to Retrieval-Augmented Generation”
   ```

3. If you’d like the output in a file, you can redirect stdout:

   ```bash
   python generate_titles.py --topic "Data Cleaning Tools" > suggested_titles.txt
   ```

## Customization

* Inside `generate_titles.py`, look for constants such as `DEFAULT_NUM_TITLES` or `DEFAULT_MODEL`. You can tweak those or pass flags at runtime.
* If you’d like to use a different LLM backend (e.g., Cohere, Anthropic, etc.), modify the API call section accordingly.

