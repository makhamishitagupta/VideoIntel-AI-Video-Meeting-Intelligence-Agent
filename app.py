import streamlit as st
import time
import os
import html
import shutil
from datetime import datetime
from dotenv import load_dotenv

from utils.audio_processor import split_audio_into_chunks
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# Sync Streamlit Cloud secrets to os.environ if running on cloud
try:
    if hasattr(st, "secrets"):
        for sec_key, sec_val in st.secrets.items():
            if isinstance(sec_val, str):
                os.environ.setdefault(sec_key, sec_val)
except Exception:
    pass

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoIntel AI — Meeting & Video Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0b0b0d;
    --surface: #121215;
    --surface-2: #18181c;
    --line: #26262c;
    --line-strong: #34343c;
    --text: #ededef;
    --text-dim: #a1a1aa;
    --text-faint: #6b6b74;
    --accent: #c8f53c;
    --accent-ink: #11140a;
    --accent-soft: rgba(200, 245, 60, 0.10);
    --accent-line: rgba(200, 245, 60, 0.35);
    --ok: #4ade80;
    --run: #fbbf24;
    --radius: 14px;
}

/* ── Base ── */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
/* Keep Streamlit's icon font intact (upload, send, arrows, etc.) */
[data-testid="stIconMaterial"],
span[class*="material-symbols"],
span[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1240px; padding-top: 1.5rem !important; padding-bottom: 4rem !important; }

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}
p, li { color: var(--text-dim); line-height: 1.7; }
strong { color: var(--text); }
a { color: var(--accent) !important; }

/* ── Top bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap;
    padding: 0.4rem 0 1.4rem 0;
    margin-bottom: 2.25rem;
    border-bottom: 1px solid var(--line);
}
.wordmark {
    display: flex; align-items: center; gap: 0.65rem;
    font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; color: var(--text);
}
.wordmark .mark {
    width: 28px; height: 28px; border-radius: 8px;
    background: var(--accent); color: var(--accent-ink);
    display: grid; place-items: center; font-size: 0.9rem; font-weight: 800;
}
.wordmark .sub { color: var(--text-faint); font-weight: 500; font-size: 0.85rem; }
.chips { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.02em;
    padding: 0.32rem 0.7rem; border-radius: 999px;
    border: 1px solid var(--line-strong); color: var(--text-dim);
    background: var(--surface);
    display: inline-flex; align-items: center; gap: 0.45rem;
}
.chip .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 8px var(--ok); }

/* ── Hero ── */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-weight: 400; font-size: clamp(2.4rem, 5vw, 3.6rem);
    line-height: 1.02; letter-spacing: -0.02em; color: var(--text);
    margin: 0 0 1rem 0;
}
.hero-title em { color: var(--accent); font-style: italic; }
.hero-lede { color: var(--text-dim); font-size: 1rem; max-width: 560px; margin-bottom: 1.75rem; }

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-faint); margin: 0 0 0.75rem 0;
}

/* ── Bordered containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0.35rem 0.5rem; }

/* ── Feature list ── */
.feature { display: flex; gap: 0.9rem; padding: 0.95rem 0; border-bottom: 1px solid var(--line); }
.feature:last-child { border-bottom: none; padding-bottom: 0.2rem; }
.feature .num {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent);
    min-width: 1.6rem; padding-top: 0.15rem;
}
.feature .ft { color: var(--text); font-weight: 600; font-size: 0.92rem; margin-bottom: 0.15rem; }
.feature .fd { color: var(--text-dim); font-size: 0.85rem; line-height: 1.55; }

/* ── Radio as segmented control ── */
div[role="radiogroup"] { gap: 0.4rem !important; flex-wrap: wrap; }
div[role="radiogroup"] > label {
    background: var(--surface-2); border: 1px solid var(--line);
    border-radius: 10px; padding: 0.5rem 0.95rem !important; margin: 0 !important;
    transition: border-color .15s ease, background .15s ease;
}
div[role="radiogroup"] > label:hover { border-color: var(--line-strong); }
div[role="radiogroup"] > label:has(input:checked) {
    background: var(--accent-soft); border-color: var(--accent-line);
}
div[role="radiogroup"] > label:has(input:checked) p { color: var(--text) !important; }
div[role="radiogroup"] > label > div:first-child { display: none; }
div[role="radiogroup"] p { font-size: 0.86rem !important; font-weight: 500; color: var(--text-dim) !important; }

/* ── Inputs ── */
.stTextInput label p, .stSelectbox label p, .stFileUploader label p {
    font-size: 0.78rem !important; font-weight: 600 !important; color: var(--text-dim) !important;
}
.stTextInput > div > div, .stSelectbox > div > div, .stTextArea > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 10px !important;
}
.stTextInput input, .stTextArea textarea {
    background: transparent !important; color: var(--text) !important; font-size: 0.92rem !important;
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg) !important;
    border: 1px dashed var(--line-strong) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent-line) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-faint) !important; }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 0.6rem 1.1rem !important;
    transition: transform .12s ease, background .15s ease, border-color .15s ease !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important; color: var(--accent-ink) !important;
    border: 1px solid var(--accent) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #d6ff5a !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"] p { color: var(--accent-ink) !important; }
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button[data-testid="stBaseButton-secondary"],
.stDownloadButton > button {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--line-strong) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent-line) !important; background: #1d1d22 !important;
}

/* ── Pipeline stepper ── */
.step {
    display: grid; grid-template-columns: 2rem 1fr auto; align-items: center; gap: 0.8rem;
    padding: 0.8rem 1rem; margin-bottom: 0.45rem;
    border-radius: 10px; background: var(--surface); border: 1px solid var(--line);
    font-size: 0.88rem; color: var(--text-faint);
}
.step .idx { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-faint); }
.step .state {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 6px;
    border: 1px solid var(--line-strong); color: var(--text-faint);
}
.step.active { border-color: rgba(251, 191, 36, 0.4); color: var(--text); background: rgba(251,191,36,0.05); }
.step.active .idx { color: var(--run); }
.step.active .state { color: var(--run); border-color: rgba(251,191,36,0.4); }
.step.done { color: var(--text-dim); }
.step.done .idx { color: var(--ok); }
.step.done .state { color: var(--ok); border-color: rgba(74,222,128,0.35); }

.stProgress > div > div > div { background: var(--line) !important; }
.stProgress > div > div > div > div { background: var(--accent) !important; }

/* ── Report header & stats ── */
.report-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-faint); margin-bottom: 0.6rem;
}
.report-meta b { color: var(--accent); font-weight: 600; }
.report-title {
    font-family: 'Instrument Serif', Georgia, serif; font-weight: 400;
    font-size: clamp(2rem, 4vw, 3rem); line-height: 1.05; letter-spacing: -0.015em;
    color: var(--text); margin: 0 0 1.5rem 0;
}
.stats {
    display: grid; grid-template-columns: repeat(4, 1fr);
    border: 1px solid var(--line); border-radius: var(--radius);
    background: var(--surface); overflow: hidden; margin-bottom: 2rem;
}
.stat { padding: 1.1rem 1.3rem; border-right: 1px solid var(--line); }
.stat:last-child { border-right: none; }
.stat .k {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-faint);
}
.stat .v { font-size: 1.65rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text); margin-top: 0.3rem; }
.stat .v small { font-size: 0.85rem; color: var(--text-faint); font-weight: 500; margin-left: 0.15rem; }
@media (max-width: 760px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .stat:nth-child(2) { border-right: none; }
    .stat:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem; background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 0.3rem; width: fit-content; max-width: 100%; overflow-x: auto;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
    display: none !important; background: transparent !important; height: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    height: auto; border-radius: 8px; padding: 0.5rem 1rem;
    background: transparent; border: none;
}
.stTabs [data-baseweb="tab"] p { font-size: 0.85rem; font-weight: 500; color: var(--text-dim); }
.stTabs [aria-selected="true"] { background: var(--surface-2) !important; box-shadow: inset 0 0 0 1px var(--line-strong); }
.stTabs [aria-selected="true"] p { color: var(--text) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

.panel-title { font-size: 1.15rem; font-weight: 600; color: var(--text); margin: 0.2rem 0 0.2rem 0; }
.panel-sub { font-size: 0.85rem; color: var(--text-faint); margin-bottom: 1rem; }

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important; border: 1px solid var(--line);
    border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.6rem;
}
[data-testid="stChatInput"] > div {
    background: var(--surface) !important; border: 1px solid var(--line-strong) !important; border-radius: 12px !important;
}
[data-testid="stChatInput"] > div:focus-within { border-color: var(--accent) !important; }
[data-testid="stAlert"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
}

/* ── Transcript ── */
.transcript-scroll {
    background: var(--bg); border: 1px solid var(--line); border-radius: 12px;
    padding: 1.5rem 1.75rem; max-height: 520px; overflow-y: auto;
    font-size: 0.92rem; line-height: 1.85; color: #c9c9cf; white-space: pre-wrap;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ───────────────────────────────────────────────
session_defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_active": False,
    "current_step": "",
    "step_status": {},
    "analysis_time": 0.0,
}
for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Top Bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="wordmark">
        <div class="mark">V</div>
        <span>VideoIntel</span><span class="sub">/ meeting intelligence</span>
    </div>
    <div class="chips">
        <span class="chip"><span class="dot"></span>engine online</span>
        <span class="chip">groq · gpt-oss-120b</span>
        <span class="chip">whisper · stt</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display labels only — the underlying option values are unchanged
SOURCE_LABELS = {
    "🔗 YouTube Video URL": "YouTube link",
    "📁 Upload Audio/Video File": "Upload file",
    "📂 Local File Path": "Local path",
}
LANGUAGE_LABELS = {
    "English": "English",
    "Hinglish (Hindi + English)": "Hinglish (Hindi + English)",
}

# ─── Input & Analysis Control ───────────────────────────────────────────────────
if not st.session_state.result:
    col_input, col_info = st.columns([7, 5], gap="large")

    with col_input:
        st.markdown("""
        <div class="eyebrow">Meeting &amp; video analysis</div>
        <h1 class="hero-title">Hours of talk,<br>reduced to <em>what matters.</em></h1>
        <p class="hero-lede">
            Drop in a YouTube link or a recording. Get a summary, action items, decisions,
            open questions — and a transcript you can ask questions of.
        </p>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-label">01 — Source</div>', unsafe_allow_html=True)
            input_method = st.radio(
                "Source Type",
                ["🔗 YouTube Video URL", "📁 Upload Audio/Video File", "📂 Local File Path"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda o: SOURCE_LABELS.get(o, o),
            )

            source_path = ""
            if input_method == "🔗 YouTube Video URL":
                yt_url = st.text_input(
                    "YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    help="Paste any standard YouTube video URL."
                )
                source_path = yt_url.strip()

            elif input_method == "📁 Upload Audio/Video File":
                uploaded_file = st.file_uploader(
                    "Upload Recording",
                    type=["mp4", "mp3", "wav", "m4a", "mov", "mkv", "webm"],
                    help="Upload an audio or video recording from your device."
                )
                if uploaded_file is not None:
                    upload_dir = os.path.join("downloads", "uploads")
                    os.makedirs(upload_dir, exist_ok=True)
                    temp_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    source_path = temp_path
                    st.caption(f"✓ {uploaded_file.name} · {round(uploaded_file.size / 1024 / 1024, 2)} MB")

            else:
                local_path = st.text_input(
                    "Local File Path",
                    placeholder="e.g. C:/Recordings/team_sync.mp4",
                    help="Provide direct absolute or relative file path on disk."
                )
                source_path = local_path.strip()

            st.markdown('<div style="height:0.5rem"></div><div class="section-label">02 — Settings</div>', unsafe_allow_html=True)
            col_lang, col_btn = st.columns([1, 1], gap="medium")
            with col_lang:
                language_option = st.selectbox(
                    "Spoken Audio Language",
                    ["English", "Hinglish (Hindi + English)"],
                    index=0,
                    help="Choose Hinglish if your audio mixes Hindi & English to enable Whisper translation.",
                    format_func=lambda o: LANGUAGE_LABELS.get(o, o),
                )
            with col_btn:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                analyze_trigger = st.button("Analyze recording  →", use_container_width=True, type="primary")

    with col_info:
        st.markdown("<div style='height: 2.2rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <div class="section-label">What you get</div>
            <div class="feature"><div class="num">01</div><div>
                <div class="ft">Local speech-to-text</div>
                <div class="fd">Transcribes audio on your machine with OpenAI's Whisper.</div>
            </div></div>
            <div class="feature"><div class="num">02</div><div>
                <div class="ft">Smart synthesis</div>
                <div class="fd">An executive summary, a concise title and structured action items.</div>
            </div></div>
            <div class="feature"><div class="num">03</div><div>
                <div class="ft">ChromaDB vector RAG</div>
                <div class="fd">The transcript is chunked and indexed so you can query it in plain language.</div>
            </div></div>
            <div class="feature"><div class="num">04</div><div>
                <div class="ft">Groq acceleration</div>
                <div class="fd">Fast answers from Groq's high-throughput cloud inference.</div>
            </div></div>
            """, unsafe_allow_html=True)

    # ── Pipeline Execution ──────────────────────────────────────────────────────
    if analyze_trigger:
        if not source_path:
            st.error("Please provide a valid YouTube URL, upload an audio/video file, or specify a file path.")
        else:
            status_container = st.container()
            progress_bar = st.progress(0)
            start_time = time.time()

            steps = [
                ("audio", "🔊 Extracting & Chunking Audio"),
                ("transcribe", "🎙️ Transcribing with Whisper AI"),
                ("title", "🏷️ Generating Session Title"),
                ("summary", "📋 Synthesizing Executive Summary"),
                ("extract", "🔍 Extracting Action Items & Decisions"),
                ("rag", "🧠 Building Chroma Vector Index"),
            ]
            step_index = {key: i + 1 for i, (key, _) in enumerate(steps)}

            def render_step(step_key, status, label):
                state_text = {"pending": "queued", "active": "running", "done": "done"}[status]
                css = "" if status == "pending" else status
                step_placeholders[step_key].markdown(
                    f'<div class="step {css}">'
                    f'<span class="idx">{step_index[step_key]:02d}</span>'
                    f'<span>{html.escape(label)}</span>'
                    f'<span class="state">{state_text}</span></div>',
                    unsafe_allow_html=True
                )

            step_placeholders = {}
            with status_container:
                st.markdown('<div style="height:1.5rem"></div><div class="section-label">Processing pipeline</div>', unsafe_allow_html=True)
                for key, label in steps:
                    step_placeholders[key] = st.empty()
                    render_step(key, "pending", label.split(" ", 1)[1])

            def mark_step(step_key, status, label):
                if status in ("active", "done"):
                    render_step(step_key, status, label)

            try:
                # 1. Audio
                mark_step("audio", "active", "Extracting & Chunking Audio")
                progress_bar.progress(10)
                chunks = split_audio_into_chunks(source_path)
                mark_step("audio", "done", "Audio Processed & Chunked")

                # 2. Transcribe
                mark_step("transcribe", "active", "Transcribing with Whisper AI")
                progress_bar.progress(35)
                translate_mode = (language_option != "English")
                transcript = transcribe_all(chunks, translate=translate_mode)
                mark_step("transcribe", "done", "Transcription Complete")

                # 3. Title
                mark_step("title", "active", "Generating Session Title")
                progress_bar.progress(55)
                title = generate_title(transcript)
                mark_step("title", "done", f"Title: {title}")

                # 4. Summary
                mark_step("summary", "active", "Synthesizing Executive Summary")
                progress_bar.progress(70)
                summary_text = summarize(transcript)
                mark_step("summary", "done", "Summary Created")

                # 5. Extract
                mark_step("extract", "active", "Extracting Action Items & Decisions")
                progress_bar.progress(85)
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)
                mark_step("extract", "done", "Action Items & Decisions Extracted")

                # 6. RAG
                mark_step("rag", "active", "Building Chroma Vector Index")
                progress_bar.progress(95)
                rag_chain = build_rag_chain(transcript)
                mark_step("rag", "done", "Vector Index Ready")

                progress_bar.progress(100)
                elapsed = round(time.time() - start_time, 1)

                st.session_state.result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary_text,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": questions,
                    "rag_chain": rag_chain,
                    "chunks_count": len(chunks),
                    "word_count": len(transcript.split()),
                    "elapsed_time": elapsed,
                    "created_at": datetime.now().strftime("%b %d, %Y • %I:%M %p"),
                }
                st.session_state.chat_history = []
                st.success("Analysis complete.")
                time.sleep(0.4)
                st.rerun()

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

# ─── Dashboard (Post-Analysis View) ─────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Header with title & quick actions
    top_col, btn_col = st.columns([8, 4], gap="medium")
    with top_col:
        st.markdown(f"""
        <div class="report-meta"><b>Session report</b> &nbsp;·&nbsp; {html.escape(r.get('created_at', ''))}</div>
        <h1 class="report-title">{html.escape(str(r['title']))}</h1>
        """, unsafe_allow_html=True)

    with btn_col:
        st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
        reset_col, dl_col = st.columns([1, 1], gap="small")
        with reset_col:
            if st.button("↺  New analysis", use_container_width=True, type="secondary"):
                st.session_state.result = None
                st.session_state.chat_history = []
                st.rerun()
        with dl_col:
            # Prepare export text
            full_export = f"""# {r['title']}
Generated by VideoIntel AI on {r.get('created_at', '')}

---

## 📋 Executive Summary
{r['summary']}

---

## ✅ Action Items
{r['action_items']}

---

## 🔑 Key Decisions
{r['key_decisions']}

---

## ❓ Open Questions
{r['open_questions']}

---

## 📝 Full Transcript
{r['transcript']}
"""
            st.download_button(
                label="↓  Export report",
                data=full_export,
                file_name=f"Meeting_Report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Metrics strip
    w_count = r.get("word_count", len(r["transcript"].split()))
    est_reading = max(1, round(w_count / 150))
    chunks_val = r.get("chunks_count", 1)
    time_val = r.get("elapsed_time", 0.0)

    st.markdown(f"""
    <div class="stats">
        <div class="stat"><div class="k">Words</div><div class="v">{w_count:,}</div></div>
        <div class="stat"><div class="k">Audio chunks</div><div class="v">{chunks_val}</div></div>
        <div class="stat"><div class="k">Reading time</div><div class="v">~{est_reading}<small>min</small></div></div>
        <div class="stat"><div class="k">Pipeline</div><div class="v">{time_val}<small>s</small></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Tabs ───────────────────────────────────────────────────────────────
    tab_summary, tab_tasks, tab_decisions, tab_chat, tab_transcript = st.tabs([
        "Summary",
        "Action items",
        "Decisions & questions",
        "Ask the meeting",
        "Transcript"
    ])

    with tab_summary:
        with st.container(border=True):
            st.markdown('<div class="section-label">Executive summary</div>', unsafe_allow_html=True)
            st.markdown(r["summary"])

    with tab_tasks:
        with st.container(border=True):
            st.markdown('<div class="section-label">Action items &amp; owners</div>', unsafe_allow_html=True)
            st.markdown(r["action_items"])

    with tab_decisions:
        c_left, c_right = st.columns(2, gap="medium")
        with c_left:
            with st.container(border=True):
                st.markdown('<div class="section-label">Key decisions</div>', unsafe_allow_html=True)
                st.markdown(r["key_decisions"])
        with c_right:
            with st.container(border=True):
                st.markdown('<div class="section-label">Open questions &amp; follow-ups</div>', unsafe_allow_html=True)
                st.markdown(r["open_questions"])

    with tab_chat:
        # Chat-tab styling (scoped with marker classes so it doesn't leak elsewhere)
        st.markdown("""
        <style>
        /* hide the zero-height marker elements */
        [data-testid="stElementContainer"]:has(.msg-marker),
        .element-container:has(.msg-marker),
        [data-testid="stElementContainer"]:has(.side-marker),
        .element-container:has(.side-marker) { display: none !important; }

        /* conversation header */
        .chat-head {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: 0.4rem 0.2rem 1rem 0.2rem; margin-bottom: 0.9rem;
            border-bottom: 1px solid var(--line);
        }
        .chat-head .t { font-size: 1.05rem; font-weight: 600; color: var(--text); }
        .chat-head .s { font-size: 0.8rem; color: var(--text-faint); margin-top: 0.15rem; }
        .chat-head .count {
            font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.06em;
            color: var(--text-dim); border: 1px solid var(--line-strong); border-radius: 999px;
            padding: 0.25rem 0.65rem; white-space: nowrap;
        }

        /* empty state */
        .chat-empty { text-align: center; padding: 3.5rem 1rem 3rem 1rem; }
        .chat-empty .glyph {
            width: 52px; height: 52px; margin: 0 auto 1rem auto; border-radius: 14px;
            display: grid; place-items: center; font-size: 1.4rem;
            background: var(--accent-soft); border: 1px solid var(--accent-line); color: var(--accent);
        }
        .chat-empty .h { font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 0.35rem; }
        .chat-empty .p { font-size: 0.86rem; color: var(--text-faint); max-width: 340px; margin: 0 auto; }

        /* message bubbles */
        [data-testid="stChatMessage"] { gap: 0.75rem; }
        [data-testid="stChatMessage"] { box-sizing: border-box; }
        [data-testid="stChatMessage"]:has(.msg-user) {
            flex-direction: row-reverse;
            background: var(--accent-soft) !important;
            border-color: var(--accent-line) !important;
            width: 82% !important;
            max-width: 82% !important;
            margin-left: auto !important;
            margin-right: 0 !important;
        }
        [data-testid="stChatMessage"]:has(.msg-assistant) {
            background: var(--surface-2) !important;
            width: 94% !important;
            max-width: 94% !important;
            margin-left: 0 !important;
            margin-right: auto !important;
        }
        [data-testid="stChatMessage"] > div { min-width: 0; }
        [data-testid="stChatMessage"]:has(.msg-user) p { color: var(--text) !important; }
        [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
            background: var(--bg) !important; border: 1px solid var(--line-strong);
            border-radius: 9px;
        }
        .msg-role {
            font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--text-faint); margin-bottom: 0.15rem;
        }
        [data-testid="stChatMessage"]:has(.msg-user) .msg-role { color: var(--accent); }
        @media (max-width: 760px) {
            [data-testid="stChatMessage"]:has(.msg-user),
            [data-testid="stChatMessage"]:has(.msg-assistant) { width: 100% !important; max-width: 100% !important; }
        }

        /* suggestion cards in the side column */
        [data-testid="stColumn"]:has(.side-marker) .stButton > button,
        [data-testid="column"]:has(.side-marker) .stButton > button {
            justify-content: flex-start !important; text-align: left !important;
            background: var(--surface) !important; border: 1px solid var(--line) !important;
            padding: 0.85rem 1rem !important; min-height: 3.2rem;
        }
        [data-testid="stColumn"]:has(.side-marker) .stButton > button:hover,
        [data-testid="column"]:has(.side-marker) .stButton > button:hover {
            border-color: var(--accent-line) !important; background: var(--surface-2) !important;
        }
        [data-testid="stColumn"]:has(.side-marker) .stButton > button > div,
        [data-testid="column"]:has(.side-marker) .stButton > button > div,
        [data-testid="stColumn"]:has(.side-marker) .stButton > button [data-testid="stMarkdownContainer"],
        [data-testid="column"]:has(.side-marker) .stButton > button [data-testid="stMarkdownContainer"] {
            width: 100%; justify-content: flex-start !important; text-align: left !important;
        }
        [data-testid="stColumn"]:has(.side-marker) .stButton > button p,
        [data-testid="column"]:has(.side-marker) .stButton > button p {
            font-weight: 500 !important; color: var(--text-dim) !important; text-align: left !important;
        }
        .side-note {
            font-size: 0.8rem; color: var(--text-faint); line-height: 1.6;
            border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 0.75rem;
        }
        .side-note b { color: var(--text-dim); font-weight: 600; }
        </style>
        """, unsafe_allow_html=True)

        chat_main, chat_side = st.columns([8, 4], gap="large")

        # ── Side column: suggested prompts (same four prompts as before) ──
        prompt_to_send = None
        with chat_side:
            st.markdown('<span class="side-marker"></span>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Suggested questions</div>', unsafe_allow_html=True)
            if st.button("→  What were the main conclusions?", use_container_width=True, type="secondary"):
                prompt_to_send = "What were the main conclusions and key takeaways of this meeting?"
            if st.button("→  Who was assigned which tasks?", use_container_width=True, type="secondary"):
                prompt_to_send = "Who was assigned tasks and what are their specific responsibilities?"
            if st.button("→  What deadlines were discussed?", use_container_width=True, type="secondary"):
                prompt_to_send = "What deadlines or timelines were discussed?"
            if st.button("→  Any unresolved issues?", use_container_width=True, type="secondary"):
                prompt_to_send = "Were there any disagreements or unresolved issues?"

            st.markdown("""
            <div class="side-note">
                <b>How it works.</b> Your question is matched against indexed chunks of the
                transcript, and the most relevant passages are sent to the model to answer.
                Specific questions get sharper answers.
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.chat_history:
                st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
                if st.button("Clear conversation", type="secondary", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

        # ── Main column: conversation ──
        with chat_main:
            n_questions = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
            with st.container(border=True):
                st.markdown(f"""
                <div class="chat-head">
                    <div>
                        <div class="t">Ask the meeting</div>
                        <div class="s">Answers are drawn from the transcript via semantic retrieval.</div>
                    </div>
                    <span class="count">{n_questions} {'question' if n_questions == 1 else 'questions'}</span>
                </div>
                """, unsafe_allow_html=True)

                if not st.session_state.chat_history:
                    st.markdown("""
                    <div class="chat-empty">
                        <div class="glyph">?</div>
                        <div class="h">Nothing asked yet</div>
                        <div class="p">Type a question below, or pick one of the suggestions to get started.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for msg in st.session_state.chat_history:
                        avatar = "👤" if msg["role"] == "user" else "✨"
                        role_label = "You" if msg["role"] == "user" else "VideoIntel"
                        with st.chat_message(msg["role"], avatar=avatar):
                            st.markdown(f'<span class="msg-marker msg-{msg["role"]}"></span>', unsafe_allow_html=True)
                            st.markdown(f'<div class="msg-role">{role_label}</div>', unsafe_allow_html=True)
                            st.markdown(msg["content"])

            # Chat input bar
            user_query = st.chat_input("Ask a question about this meeting…")

            # Check if a suggestion or the chat input triggered
            active_query = prompt_to_send or user_query
            if active_query:
                st.session_state.chat_history.append({"role": "user", "content": active_query})
                with st.spinner("Searching the transcript…"):
                    answer = ask_question(r["rag_chain"], active_query)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    with tab_transcript:
        t_header_col, t_dl_col = st.columns([8, 4])
        with t_header_col:
            st.markdown("""
            <div class="panel-title">Full transcript</div>
            <div class="panel-sub">Verbatim output from Whisper.</div>
            """, unsafe_allow_html=True)
        with t_dl_col:
            st.download_button(
                label="↓  Download .txt",
                data=r["transcript"],
                file_name=f"Transcript_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown(f'<div class="transcript-scroll">{html.escape(r["transcript"])}</div>', unsafe_allow_html=True)