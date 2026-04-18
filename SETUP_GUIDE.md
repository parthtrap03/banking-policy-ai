# Banking Policy Intelligence Platform — Complete Guide

> Run this project on any Windows PC without any IDE or AI assistant.  
> All you need: **Python 3.10+**, **Internet connection**, and a **terminal**.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Setup](#2-first-time-setup)
3. [Running the App](#3-running-the-app)
4. [Using the App](#4-using-the-app)
5. [Adding New Documents](#5-adding-new-documents)
6. [Configuration](#6-configuration)
7. [Troubleshooting](#7-troubleshooting)
8. [Project File Reference](#8-project-file-reference)
9. [API Limits & Alternatives](#9-api-limits--alternatives)
10. [Monthly Data Refresh](#10-monthly-data-refresh)

---

## 1. Prerequisites

### Install Python (if not already installed)

1. Open https://www.python.org/downloads/
2. Download **Python 3.12** (or any 3.10+).
3. During installation, **check** ✅ "Add Python to PATH".
4. After installation, open **PowerShell** and verify:
   ```powershell
   python --version
   ```
   You should see something like: `Python 3.12.x`

### Get a Groq API Key (free)

1. Go to https://console.groq.com/keys
2. Sign up (free — no credit card needed).
3. Click **"Create API Key"**.
4. Copy the key (starts with `gsk_...`).
5. You'll paste this in the `.env` file in the next step.

---

## 2. First-Time Setup

Open **PowerShell** and run these commands one by one:

```powershell
# Step 1: Navigate to the project folder
cd "c:\Users\03par\OneDrive\Desktop\policy ai\legal_rag_project"

# Step 2: Install all dependencies (takes 2-5 minutes)
pip install -r requirements.txt

# Step 3: Set your API key
# Open .env in Notepad and paste your Groq API key:
notepad .env
```

In Notepad, find this line:
```
GROQ_API_KEY=gsk_your_key_here
```
Replace `gsk_your_key_here` with the key you copied from Groq. Save and close.

```powershell
# Step 4: Build the vector database (first time only, takes 1-2 minutes)
$env:PYTHONIOENCODING='utf-8'
python legal_rag_system.py
```

Wait until you see:
```
✓ System Ready!
```

**Setup is complete!** You only need to do this once.

---

## 3. Running the App

### Option A: Web UI (Recommended)

```powershell
cd "c:\Users\03par\OneDrive\Desktop\policy ai\legal_rag_project"
$env:PYTHONIOENCODING='utf-8'
python -m streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**.

> **Note:** The first load takes ~30-50 seconds (loading AI models). After that, page refreshes are instant.

To stop: press `Ctrl+C` in the terminal.

### Option B: Command-Line Interface

```powershell
cd "c:\Users\03par\OneDrive\Desktop\policy ai\legal_rag_project"
$env:PYTHONIOENCODING='utf-8'
python query_interactive.py
```

Type your questions and press Enter. Type `quit` to exit.

---

## 4. Using the App

### Asking Questions

Type any banking or factoring policy question in the chat box at the bottom:
- "What is the UPI transaction limit?"
- "What is the difference between recourse and non-recourse factoring?"
- "What are the TReDS platforms in India?"
- "What is the penalty for data breach under DPDP Act?"

### Answer Style Toggle (sidebar)

In the left sidebar, under **💬 Answer Style**, choose:

| Style | What it does |
|-------|-------------|
| **None** | Gives the legal/technical answer as-is |
| **Short example** | Adds a one-line real-world example at the end |
| **Detailed example** | Adds a 2-3 sentence scenario with names and rupee amounts |

### Suggested Questions

Click **💡 Suggestions** in the sidebar to show pre-built questions organized by category (UPI, KYC, Factoring, etc.).

### Feedback

After each answer, click 👍 or 👎 to rate the quality. This helps track which topics need better coverage.

### Clear Chat

Click **🗑️ Clear Chat** in the sidebar to start a fresh conversation.

---

## 5. Adding New Documents

To add new bank policies or regulations:

1. **Create a `.txt` file** with the document content.
   - Use `## Headings` to mark sections (the system uses these for smart chunking).
   - Example filename: `hdfc_home_loan_terms.txt`

2. **Drop the file** into the `legal_documents/` folder.

3. **Rebuild the database:**
   ```powershell
   cd "c:\Users\03par\OneDrive\Desktop\policy ai\legal_rag_project"
   $env:PYTHONIOENCODING='utf-8'
   python legal_rag_system.py
   ```

4. **Restart Streamlit** (stop with Ctrl+C, then re-run):
   ```powershell
   python -m streamlit run app.py
   ```

### Document Format Tips

```text
BANK NAME — PRODUCT TERMS AND CONDITIONS

## 1. ELIGIBILITY
(a) The applicant must be a resident of India.
(b) Minimum age: 21 years.
...

## 2. FEES AND CHARGES
### 2.1 Processing Fee
Processing fee: 1% of the loan amount (minimum Rs. 5,000).
...

## 3. REPAYMENT
(a) EMI is deducted on the 5th of every month.
...
```

The system automatically:
- Detects section headings and creates smart chunks.
- Classifies the document type (RBI Circular, Act, T&C, etc.).
- Identifies the issuing authority.
- Tags sections (KYC, penalties, fees, factoring, etc.).

---

## 6. Configuration

All settings are in the `.env` file. Open it with any text editor:

```powershell
notepad .env
```

### Key Settings

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `GROQ_API_KEY` | *(your key)* | LLM API access |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (see below) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | LLM model name |
| `LLM_TEMPERATURE` | `0.2` | Creativity (0 = strict, 1 = creative) |
| `LLM_MAX_TOKENS` | `600` | Max answer length |
| `RETRIEVAL_K` | `5` | Number of source documents per answer |
| `RERANK_INITIAL_K` | `12` | Candidates before reranking (lower = faster) |

### Embedding Model Options

| Model | Cold Start | Accuracy | Best For |
|-------|-----------|----------|----------|
| `BAAI/bge-small-en-v1.5` | ~5s | Good | **Daily use (recommended)** |
| `BAAI/bge-large-en-v1.5` | ~15s | Best | Maximum accuracy |

**If you change the embedding model**, you MUST rebuild the database:
```powershell
# Delete the old database first
Remove-Item -Recurse -Force chroma_db_v2
# Rebuild
$env:PYTHONIOENCODING='utf-8'
python legal_rag_system.py
```

---

## 7. Troubleshooting

### "UnicodeEncodeError" or garbled text

**Fix:** Always set encoding before running:
```powershell
$env:PYTHONIOENCODING='utf-8'
```

To make this permanent, add it to your PowerShell profile:
```powershell
notepad $PROFILE
```
Add this line and save:
```
$env:PYTHONIOENCODING='utf-8'
```

### "ModuleNotFoundError: No module named 'xxx'"

**Fix:** Install dependencies:
```powershell
pip install -r requirements.txt
```

### "Connection error" or "Rate limit exceeded" from Groq

**Fix:** Groq's free tier allows 30 requests/minute. Wait 60 seconds and try again. For unlimited usage, see Section 9.

### App shows "Running load_rag_system()..." for a long time

This is normal on first load (~30-50 seconds). The system is loading AI models. After the first load, Streamlit caches everything — subsequent page refreshes are instant.

### "Collection expecting embedding with dimension of 1024, got 384"

**Fix:** You switched the embedding model without rebuilding the database:
```powershell
Remove-Item -Recurse -Force chroma_db_v2
$env:PYTHONIOENCODING='utf-8'
python legal_rag_system.py
```

### Answers seem wrong or unrelated

1. Check if your document was loaded:
   - Look at the sidebar → "Policy Documents" section.
   - Your document should appear in the list.
2. If not listed, rebuild the database (see Section 5, Step 3).
3. If listed but answers are wrong, your document may need better headings — use `## Section Title` format.

### How to stop the app

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## 8. Project File Reference

| File | What it does | Do you edit it? |
|------|-------------|----------------|
| `.env` | API keys and settings | **Yes** — your config |
| `app.py` | Web UI (Streamlit) | No (unless customizing UI) |
| `legal_rag_system.py` | Core AI engine | No (unless adding features) |
| `hybrid_retriever.py` | Search engine (Vector + BM25) | No |
| `banking_prompts.py` | Question templates | No (unless adding categories) |
| `analytics.py` | Usage tracking | No |
| `query_interactive.py` | Command-line interface | No |
| `download_corpus.py` | Data refresh script | No |
| `requirements.txt` | Python dependencies | No |
| `SETUP_GUIDE.md` | This guide | No |
| `legal_documents/*.txt` | Policy documents | **Yes** — add your docs here |
| `chroma_db_v2/` | Vector database | No (auto-generated) |

---

## 9. API Limits & Alternatives

### Groq Free Tier Limits
- 30 requests per minute
- 14,400 tokens per minute
- 30,000 requests per day

### If You Hit the Limit

**Option 1: Wait 60 seconds** — the limit resets every minute.

**Option 2: Get a paid Groq plan** — visit https://console.groq.com/settings/billing

**Option 3: Use Ollama (100% free, runs locally, no limits)**

1. Download and install Ollama: https://ollama.com/download
2. Open a new terminal and run:
   ```powershell
   ollama pull llama3.1:8b
   ```
   (This downloads the model — ~5 GB, one-time download)
3. Edit `legal_rag_system.py`:
   - Find this block (around line 358):
     ```python
     from langchain_groq import ChatGroq
     self.llm = ChatGroq(
         model=llm_model,
         api_key=api_key,
         temperature=float(os.getenv('LLM_TEMPERATURE', '0.2')),
         max_tokens=int(os.getenv('LLM_MAX_TOKENS', '600')),
     )
     ```
   - Replace it with:
     ```python
     from langchain_community.llms import Ollama
     self.llm = Ollama(
         model="llama3.1:8b",
         temperature=float(os.getenv('LLM_TEMPERATURE', '0.2')),
     )
     ```
4. Save and restart the app. No API key needed!

**Option 4: Use Google AI Studio (free, 60 RPM)**

1. Get a free API key: https://aistudio.google.com/apikey
2. Install: `pip install langchain-google-genai`
3. Replace the LLM block with:
   ```python
   from langchain_google_genai import ChatGoogleGenerativeAI
   self.llm = ChatGoogleGenerativeAI(
       model="gemini-1.5-flash",
       google_api_key="your_google_key_here",
       temperature=0.2,
   )
   ```

---

## 10. Monthly Data Refresh

To pull the latest factoring and banking data from online sources:

```powershell
cd "c:\Users\03par\OneDrive\Desktop\policy ai\legal_rag_project"
$env:PYTHONIOENCODING='utf-8'

# One-time download
python download_corpus.py

# Or run continuously (refreshes on the 1st of every month)
python download_corpus.py --schedule
```

After downloading new data, rebuild the database:
```powershell
python legal_rag_system.py
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  BANKING POLICY AI — QUICK COMMANDS             │
├─────────────────────────────────────────────────┤
│                                                 │
│  FIRST TIME:                                    │
│    pip install -r requirements.txt              │
│    python legal_rag_system.py                   │
│                                                 │
│  DAILY USE:                                     │
│    $env:PYTHONIOENCODING='utf-8'                │
│    python -m streamlit run app.py               │
│                                                 │
│  ADD DOCUMENTS:                                 │
│    1. Drop .txt into legal_documents/           │
│    2. python legal_rag_system.py                │
│    3. Restart Streamlit                         │
│                                                 │
│  CLI MODE:                                      │
│    python query_interactive.py                  │
│                                                 │
│  STOP:                                          │
│    Ctrl+C                                       │
│                                                 │
└─────────────────────────────────────────────────┘
```
