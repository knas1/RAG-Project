# Standards RAG chatbot

Ask questions about standards PDFs and get answers with clause and page citations.

## Run locally

```bash
pip install -r requirements.txt
# put your PDFs in Pdfs/
python ingest.py              # build the index into chroma_db/
python search.py "minimum password length"   # check retrieval, no LLM needed
export OPENAI_API_KEY=...     # or OPENAI_BASE_URL for Ollama etc.
python chatbot.py
```

Set the standard name and version per file in `MANIFEST` in `config.py`.

## Build the index on Kaggle, use it locally

Nothing is trained: `ingest.py` only embeds the PDFs. The output is the `chroma_db/` folder.

1. Upload your PDFs as a Kaggle dataset. In the notebook settings, turn on Internet (the models download from Hugging Face). A GPU is optional.
2. In the notebook:
   ```bash
   !git clone https://github.com/knas1/RAG-Project.git && cd RAG-Project && pip install -r requirements.txt
   !cd RAG-Project && PDF_DIR=/kaggle/input/<your-dataset> DB_DIR=/kaggle/working/chroma_db python ingest.py --reset
   !cd /kaggle/working && zip -r chroma_db.zip chroma_db
   ```
3. Download `chroma_db.zip` from the notebook Output tab and unzip it into the project folder so `./chroma_db` exists.
4. Run `python chatbot.py` locally. Use the same `EMBED_MODEL` and the same chromadb version as on Kaggle, or the index will not load or will give wrong results.
