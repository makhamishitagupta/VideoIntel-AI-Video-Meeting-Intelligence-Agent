import streamlit as st
import time
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv

from utils.audio_processor import split_audio_into_chunks
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoIntel AI — Meeting & Video Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom Modern Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-main: #07090e;
    --bg-surface: #0e131f;
    --bg-card: #141b2d;
    --bg-card-hover: #1b243b;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-focus: rgba(99, 102, 241, 0.5);
    
    --primary: #6366f1;
    --primary-glow: rgba(99, 102, 241, 0.25);
    --secondary: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

/* Global Reset */
html, body, [class*="st-emotion-cache"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 90%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
                var(--bg-main) !important;
}

/* Header & Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

/* Custom Header Bar */
.brand-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
}

.brand-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 20%, #a5b4fc 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
}

.badge-pill.green {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

.badge-pill.cyan {
    background: rgba(6, 182, 212, 0.12);
    border-color: rgba(6, 182, 212, 0.3);
    color: #67e8f9;
}

/* Glassmorphic Container */
.glass-box {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-box:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

.box-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 0.6rem;
}

/* Metric Stats Row */
.stat-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: left;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.05em;
}
.stat-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 0.2rem;
}

/* Action Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 4px 15px var(--primary-glow) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Tabs Redesign */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-secondary);
    background: transparent;
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #c7d2fe !important;
}

/* Stepper Bar */
.stepper-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    font-size: 0.85rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stepper-item.active {
    background: rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.4);
    color: #e0e7ff;
}
.stepper-item.done {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.3);
    color: #a7f3d0;
}

/* Input Fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-glow) !important;
}

/* Chat Styling */
.stChatMessage {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

/* Transcript Container */
.transcript-scroll {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem;
    max-height: 480px;
    overflow-y: auto;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #cbd5e1;
    white-space: pre-wrap;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }
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

# ─── Navigation Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-nav">
    <div class="brand-logo">
        <span>⚡ VideoIntel AI</span>
    </div>
    <div style="display:flex; gap:0.6rem; align-items:center;">
        <span class="badge-pill green">🟢 Engine Online</span>
        <span class="badge-pill">⚡ Groq GPT-OSS 120B</span>
        <span class="badge-pill cyan">🎙️ Whisper AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Input & Analysis Control ───────────────────────────────────────────────────
if not st.session_state.result:
    col_input, col_info = st.columns([7, 5], gap="large")

    with col_input:
        st.markdown("""
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size:1.8rem; margin-bottom:0.3rem;">🎬 Meeting & Video Analysis</h2>
            <p style="color:var(--text-secondary); font-size:0.92rem;">
                Transform lengthy YouTube videos or meeting recordings into actionable summaries, structured decisions, and an interactive conversational assistant.
            </p>
        </div>
        """, unsafe_allow_html=True)

        input_method = st.radio(
            "Source Type",
            ["🔗 YouTube Video URL", "📁 Upload Audio/Video File", "📂 Local File Path"],
            horizontal=True,
            label_visibility="collapsed"
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
                st.caption(f"✅ Uploaded: `{uploaded_file.name}` ({round(uploaded_file.size / 1024 / 1024, 2)} MB)")

        else:
            local_path = st.text_input(
                "Local File Path",
                placeholder="e.g. C:/Recordings/team_sync.mp4",
                help="Provide direct absolute or relative file path on disk."
            )
            source_path = local_path.strip()

        col_lang, col_btn = st.columns([1, 1], gap="medium")
        with col_lang:
            language_option = st.selectbox(
                "Spoken Audio Language",
                ["English", "Hinglish (Hindi + English)"],
                index=0,
                help="Choose Hinglish if your audio mixes Hindi & English to enable Whisper translation."
            )
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            analyze_trigger = st.button("🚀 Analyze Recording", use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="glass-box" style="margin-top:0.5rem;">
            <div class="box-header">⚡ Features Included</div>
            <div style="display:flex; flex-direction:column; gap:0.9rem; font-size:0.88rem; color:var(--text-secondary);">
                <div>
                    <strong style="color:var(--text-primary);">🎙️ Local Speech-to-Text:</strong>
                    <br>Accurately transcribes audio locally using OpenAI's Whisper model.
                </div>
                <div>
                    <strong style="color:var(--text-primary);">📋 Smart Synthesis:</strong>
                    <br>Extracts an executive summary, concise title, and structured action items.
                </div>
                <div>
                    <strong style="color:var(--text-primary);">🧠 ChromaDB Vector RAG:</strong>
                    <br>Chunks and indexes meeting transcripts for conversational semantic querying.
                </div>
                <div>
                    <strong style="color:var(--text-primary);">⚡ Groq Acceleration:</strong>
                    <br>Instant responses powered by Groq's high-throughput cloud inference.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline Execution ──────────────────────────────────────────────────────
    if analyze_trigger:
        if not source_path:
            st.error("⚠️ Please provide a valid YouTube URL, upload an audio/video file, or specify a file path.")
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

            step_placeholders = {}
            with status_container:
                st.markdown("### ⚙️ Processing Pipeline")
                for key, label in steps:
                    step_placeholders[key] = st.empty()
                    step_placeholders[key].markdown(
                        f'<div class="stepper-item"><span>⏳</span><span>{label}</span></div>',
                        unsafe_allow_html=True
                    )

            def mark_step(step_key, status, label):
                if status == "active":
                    step_placeholders[step_key].markdown(
                        f'<div class="stepper-item active"><span>⚙️</span><strong>{label} (running…)</strong></div>',
                        unsafe_allow_html=True
                    )
                elif status == "done":
                    step_placeholders[step_key].markdown(
                        f'<div class="stepper-item done"><span>✅</span><span>{label}</span></div>',
                        unsafe_allow_html=True
                    )

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
                st.success("✅ Analysis completed successfully!")
                time.sleep(0.4)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")

# ─── Dashboard (Post-Analysis View) ─────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Header with title & quick action
    top_col, btn_col = st.columns([8, 4], gap="medium")
    with top_col:
        st.markdown(f"""
        <div style="margin-bottom:0.5rem;">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--primary); font-weight:700;">
                📌 Session Report • {r.get('created_at', '')}
            </div>
            <h1 style="font-size: clamp(1.8rem, 3.5vw, 2.5rem); margin: 0.2rem 0 0.5rem 0;">
                {r['title']}
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with btn_col:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        reset_col, dl_col = st.columns([1, 1], gap="small")
        with reset_col:
            if st.button("🔄 New Analysis", use_container_width=True, type="secondary"):
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
                label="📥 Export Report",
                data=full_export,
                file_name=f"Meeting_Report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Metrics Strip
    w_count = r.get("word_count", len(r["transcript"].split()))
    est_reading = max(1, round(w_count / 150))
    chunks_val = r.get("chunks_count", 1)
    time_val = r.get("elapsed_time", 0.0)

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">📝 Word Count</div>
            <div class="stat-value">{w_count:,}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">🧩 Audio Chunks</div>
            <div class="stat-value">{chunks_val}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">⏱️ Est. Reading Time</div>
            <div class="stat-value">~{est_reading} min</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">⚡ Pipeline Speed</div>
            <div class="stat-value">{time_val}s</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Main Tabs ───────────────────────────────────────────────────────────────
    tab_summary, tab_tasks, tab_decisions, tab_chat, tab_transcript = st.tabs([
        "📋 Executive Summary",
        "✅ Action Items",
        "🔑 Decisions & Questions",
        "💬 Chat with Meeting",
        "📝 Transcript"
    ])

    with tab_summary:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Executive Summary")
        st.markdown(r["summary"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_tasks:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Action Items & Assigned Tasks")
        st.markdown(r["action_items"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_decisions:
        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.markdown('<div class="glass-box">', unsafe_allow_html=True)
            st.markdown("### 🔑 Key Decisions Reached")
            st.markdown(r["key_decisions"])
            st.markdown('</div>', unsafe_allow_html=True)
        with c_right:
            st.markdown('<div class="glass-box">', unsafe_allow_html=True)
            st.markdown("### ❓ Open Questions & Follow-ups")
            st.markdown(r["open_questions"])
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_chat:
        st.markdown("""
        <div style="margin-bottom:1rem;">
            <h3 style="margin-bottom:0.2rem;">💬 Conversational Meeting Assistant</h3>
            <p style="color:var(--text-secondary); font-size:0.88rem;">
                Ask questions to query the meeting transcript using semantic retrieval (RAG).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick Suggestion Chips
        st.markdown("<span style='font-size:0.75rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;'>Suggested Prompts:</span>", unsafe_allow_html=True)
        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4, gap="small")
        
        prompt_to_send = None
        with chip_col1:
            if st.button("📌 Main Conclusions?", use_container_width=True, type="secondary"):
                prompt_to_send = "What were the main conclusions and key takeaways of this meeting?"
        with chip_col2:
            if st.button("👥 Who was assigned tasks?", use_container_width=True, type="secondary"):
                prompt_to_send = "Who was assigned tasks and what are their specific responsibilities?"
        with chip_col3:
            if st.button("⏳ Deadlines mentioned?", use_container_width=True, type="secondary"):
                prompt_to_send = "What deadlines or timelines were discussed?"
        with chip_col4:
            if st.button("❓ Unresolved issues?", use_container_width=True, type="secondary"):
                prompt_to_send = "Were there any disagreements or unresolved issues?"

        # Chat history container
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.info("💡 Start chatting below or click any of the suggestion buttons above!")
            else:
                for msg in st.session_state.chat_history:
                    avatar = "👤" if msg["role"] == "user" else "🤖"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])

        # Chat input bar
        user_query = st.chat_input("Ask a question about this meeting transcript...")
        
        # Check if button chip or chat input triggered
        active_query = prompt_to_send or user_query
        if active_query:
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.spinner("Searching transcript & reasoning with Groq…"):
                answer = ask_question(r["rag_chain"], active_query)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    with tab_transcript:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        t_header_col, t_dl_col = st.columns([8, 4])
        with t_header_col:
            st.markdown("### 📝 Full Meeting Transcript")
        with t_dl_col:
            st.download_button(
                label="📥 Download Transcript (.txt)",
                data=r["transcript"],
                file_name=f"Transcript_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown(f'<div class="transcript-scroll">{r["transcript"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
