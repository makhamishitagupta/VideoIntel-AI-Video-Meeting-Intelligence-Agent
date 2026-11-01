<div align="center">

# ⚡ VideoIntel AI
### Enterprise-Grade Meeting & Video Intelligence Assistant

<!-- [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Inference-F05A28?style=for-the-badge)](https://groq.com/)
[![OpenAI Whisper](https://img.shields.io/badge/Whisper-Local_STT-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper) -->

<p align="center">
  <strong>Transform lengthy meetings, lecture recordings, and YouTube videos into structured executive summaries, actionable task checklists, and an interactive RAG conversational intelligence engine.</strong>
</p>

</div>

---

## 🌟 Core Capabilities

| Domain | Capability | Description |
| :--- | :--- | :--- |
| **🎙️ Audio & Speech** | **Local Whisper AI** | 100% offline transcription on CPU/GPU — zero audio leaks or API transcription costs. |
| | **Multi-Source Ingestion** | Ingest public YouTube URLs via `yt-dlp` or upload local media (`.mp4`, `.mp3`, `.wav`, `.m4a`, `.mov`). |
| **🧠 Intelligence & RAG** | **Groq LPU™ Acceleration** | Sub-second inference and reasoning powered by `openai/gpt-oss-120b`. |
| | **ChromaDB Vector Store** | Dense semantic indexing using `all-MiniLM-L6-v2` for chunk-level context retrieval. |
| | **Conversational QA** | Natural-language meeting chat grounded strictly in the transcript with zero hallucinations. |
| **📋 Insights & Export** | **Executive Synthesis** | Professional meeting summaries, titles, and key takeaways generated in seconds. |
| | **Action Items & Decisions** | Automatically isolates tasks with assignees and deadlines, key decisions, and open issues. |
| | **1-Click Export** | Download structured reports in Markdown (`.md`) or raw transcripts in Text (`.txt`). |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10** or higher installed.
- **FFmpeg** installed and accessible in your system `PATH`:
  - **Windows (via Chocolatey / Scoop):**
    ```powershell
    choco install ffmpeg
    # or
    scoop install ffmpeg
    ```
  - **macOS (via Homebrew):**
    ```bash
    brew install ffmpeg
    ```
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

---

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/video-intel-assistant.git
cd video-intel-assistant
```

---

### 3. Create a Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required packages
pip install -r Requirements.txt
```

---

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your credentials:
```ini
# Get your free key at https://console.groq.com/
GROQ_API_KEY=gsk_your_actual_groq_key_here

# LLM model identifier on Groq
GROQ_MODEL=openai/gpt-oss-120b

# Whisper STT model (tiny, base, small, medium, large)
WHISPER_MODEL=base
```

---

## 💻 Running the Application

### Option A: Modern Streamlit Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the interactive executive dashboard.

### Option B: Terminal / CLI Mode
```bash
python main.py
```
Enter your YouTube link or local file path in the terminal prompt to generate a terminal summary and enter chat mode.

---

## 📂 Project Structure

```text
├── core/
│   ├── transcriber.py        # Local Whisper speech-to-text logic
│   ├── summarizer.py         # Meeting title & executive summary pipeline
│   ├── extractor.py          # Action items, decisions & questions extraction
│   ├── vector_store.py       # ChromaDB vector store & embeddings
│   └── rag_engine.py         # LCEL RAG chain for conversational chat
├── utils/
│   └── audio_processor.py    # yt-dlp downloading & 10-min audio chunking
├── app.py                    # Streamlit web UI & dashboard
├── main.py                   # Terminal CLI interface
├── Requirements.txt          # Python dependencies
├── .env.example              # Sample environment configuration
├── .gitignore                # Protects secrets, temporary files & vector db
└── README.md                 # Project documentation
```

---

## ⚙️ Supported Models & Performance

| Component | Default Engine | Alternatives |
| :--- | :--- | :--- |
| **LLM Inference** | `openai/gpt-oss-120b` (Groq Cloud) | `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` |
| **Speech-to-Text** | `OpenAI Whisper (base)` | `tiny`, `small`, `medium`, `large-v3` |
| **Vector Embeddings** | `all-MiniLM-L6-v2` (SentenceTransformers) | HuggingFace Hub embedding models |
| **Vector Store** | `ChromaDB` (Local Disk) | In-memory Chroma |

---

## 🔒 Security & Privacy

- **Protected API Keys:** Credentials in `.env` are excluded via `.gitignore` and never committed.
- **Local Audio Processing:** Audio files and Whisper transcription run 100% locally on your machine.
- **Temporary Files:** Downloaded and converted WAV audio files are stored in the git-ignored `downloads/` directory.

---
