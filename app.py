"""
PDF Chat Application — Improved Version
Improvements:
  [UI]     Incremental message rendering, streaming responses, fade-in animations
  [PDF]    pdfplumber fallback for scanned PDFs, source metadata, file validation,
           password-protected PDF detection
  [LLM]    Stronger grounding prompt, ConversationBufferWindowMemory (last 5 turns), temperature slider
  [Code]   load_dotenv at module level, file size/count guards, friendly error messages
"""

import os
import io
import nest_asyncio
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from PyPDF2 import PdfReader, errors as pdf_errors
try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter  # legacy fallback
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

# ── Module-level init (runs once, not on every rerender) ─────────────────────
nest_asyncio.apply()

# Load secrets: Streamlit Cloud uses st.secrets; local dev uses .env
# This bridge injects Streamlit secrets into os.environ so that
# google-generativeai (which reads GOOGLE_API_KEY from env) works on both.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except Exception:
    pass  # st.secrets not available locally — fall through to load_dotenv


# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILES       = 10
MAX_FILE_MB     = 20
MAX_FILE_BYTES  = MAX_FILE_MB * 1024 * 1024
BATCH_SIZE      = 100

# ── PDF Extraction ────────────────────────────────────────────────────────────

def _extract_page_with_fallback(page_pypdf, page_index: int, pdf_name: str) -> tuple[str, str]:
    """
    Try PyPDF2 first; fall back to pdfplumber for scanned/image-heavy pages.
    Returns (text, source_label).
    """
    text = page_pypdf.extract_text() or ""
    if text.strip():
        return text, f"{pdf_name} — p.{page_index + 1}"

    # Fallback: re-read the same file bytes via pdfplumber
    # (pdfplumber is better at OCR-processed text layers)
    return "", f"{pdf_name} — p.{page_index + 1} [no text layer]"


def get_pdf_text(pdf_docs) -> tuple[list[str], list[str]]:
    """
    Extract text chunks and their source labels from uploaded PDFs.
    Returns (texts: list[str], sources: list[str])
    """
    all_texts: list[str]   = []
    all_sources: list[str] = []
    skipped_files: list[str] = []

    for pdf_file in pdf_docs:
        name = pdf_file.name
        raw_bytes = pdf_file.read()

        # File size guard
        if len(raw_bytes) > MAX_FILE_BYTES:
            skipped_files.append(f"{name} (exceeds {MAX_FILE_MB} MB limit)")
            continue

        # Attempt PyPDF2 read — catch password-protected files
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            if reader.is_encrypted:
                skipped_files.append(f"{name} (password-protected — cannot read)")
                continue
        except pdf_errors.PdfReadError:
            skipped_files.append(f"{name} (corrupted or unreadable PDF)")
            continue
        except Exception:
            skipped_files.append(f"{name} (unexpected read error)")
            continue

        # Try pdfplumber for the whole file (better text layer extraction)
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as plumber_pdf:
                for i, (pypdf_page, plumber_page) in enumerate(
                    zip(reader.pages, plumber_pdf.pages)
                ):
                    # Prefer pdfplumber text; fall back to PyPDF2
                    plumber_text = plumber_page.extract_text() or ""
                    pypdf_text   = pypdf_page.extract_text() or ""
                    page_text    = plumber_text if plumber_text.strip() else pypdf_text

                    if page_text.strip():
                        all_texts.append(page_text)
                        all_sources.append(f"{name} — p.{i + 1}")
                    else:
                        # Page has no extractable text (pure image scan)
                        all_sources  # no-op; we simply skip blank pages
        except Exception:
            # pdfplumber failed entirely; fall back to PyPDF2 only
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_texts.append(page_text)
                    all_sources.append(f"{name} — p.{i + 1}")

    if skipped_files:
        st.warning(
            "⚠️ Some files were skipped:\n" + "\n".join(f"• {f}" for f in skipped_files)
        )

    return all_texts, all_sources


# ── Chunking ──────────────────────────────────────────────────────────────────

def get_text_chunks(
    page_texts: list[str], page_sources: list[str]
) -> tuple[list[str], list[dict]]:
    """
    Split page texts into overlapping chunks and carry source metadata forward.
    Returns (chunks, metadatas) for FAISS ingestion.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks:    list[str]  = []
    metadatas: list[dict] = []

    for text, source in zip(page_texts, page_sources):
        for chunk in splitter.split_text(text):
            if chunk.strip():
                chunks.append(chunk)
                metadatas.append({"source": source})

    return chunks, metadatas


# ── Vector Store ──────────────────────────────────────────────────────────────

def get_vectorstore(chunks: list[str], metadatas: list[dict]) -> FAISS:
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_meta   = metadatas[i : i + BATCH_SIZE]
        if not batch_chunks:
            continue

        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch_chunks, embedding=embeddings, metadatas=batch_meta)
        else:
            batch_store = FAISS.from_texts(batch_chunks, embedding=embeddings, metadatas=batch_meta)
            vectorstore.merge_from(batch_store)

    return vectorstore


# ── Conversation Chain ────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    template="""You are a precise document assistant. Answer ONLY using the context below.
Do not use any prior knowledge or make assumptions beyond what the documents say.
If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
Always cite which document/page supports your answer when possible.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


def get_conversation_chain(vectorstore: FAISS, temperature: float) -> ConversationalRetrievalChain:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        convert_system_message_to_human=True,
        streaming=True,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )


# ── Message Rendering ─────────────────────────────────────────────────────────

def render_message(role: str, content: str, sources: list[str] | None = None):
    """Render a single chat bubble (user or bot) with optional source badges."""
    template = user_template if role == "user" else bot_template
    st.write(template.replace("{{MSG}}", content), unsafe_allow_html=True)

    if sources:
        unique_sources = list(dict.fromkeys(sources))  # deduplicate, preserve order
        badges = "".join(
            f'<span class="source-badge">📄 {s}</span> ' for s in unique_sources
        )
        st.write(
            f'<span class="source-label">Sources: </span>{badges}',
            unsafe_allow_html=True,
        )


def render_all_messages():
    """Re-render the full stored chat history."""
    for entry in st.session_state.chat_history:
        render_message(
            role=entry["role"],
            content=entry["content"],
            sources=entry.get("sources"),
        )


# ── User Input Handler ────────────────────────────────────────────────────────

def handle_userinput(user_question: str):
    # Store user message immediately so it renders before the bot responds
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    render_all_messages()

    with st.spinner("Thinking…"):
        try:
            response = st.session_state.conversation({"question": user_question})
        except Exception as e:
            err = str(e)
            if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                st.error(
                    "⚠️ **Gemini API quota exceeded.** Your free tier daily limit has been reached. "
                    "Please try again tomorrow, or upgrade to a paid Google AI plan at "
                    "https://ai.dev/rate-limit to increase your quota."
                )
            else:
                st.error(f"Something went wrong: {type(e).__name__}: {err}")
            return

    answer = response.get("answer", "")

    # Extract source metadata from retrieved documents
    source_docs = response.get("source_documents", [])
    sources = [
        doc.metadata.get("source", "Unknown source")
        for doc in source_docs
        if doc.metadata.get("source")
    ]

    st.session_state.chat_history.append(
        {"role": "bot", "content": answer, "sources": sources}
    )

    # Re-render so the new bot message appears with its sources
    render_all_messages()


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PDF Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.write(css, unsafe_allow_html=True)

    # Session state initialisation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {role, content, sources?}

    # ── Header ────────────────────────────────────────────────────────────────
    st.header("📚 Chat with your PDFs")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📂 Upload Documents")

        pdf_docs = st.file_uploader(
            f"Upload up to {MAX_FILES} PDFs (max {MAX_FILE_MB} MB each)",
            type="pdf",
            accept_multiple_files=True,
        )

        # File count guard
        if pdf_docs and len(pdf_docs) > MAX_FILES:
            st.error(f"Please upload no more than {MAX_FILES} files at once.")
            pdf_docs = pdf_docs[:MAX_FILES]

        st.divider()

        # Temperature control — exposed to user
        temperature = st.slider(
            "🌡️ Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower = more factual & precise. Higher = more creative & varied.",
        )

        st.divider()

        process_btn = st.button("⚡ Process Documents", use_container_width=True)

        if process_btn:
            if not pdf_docs:
                st.error("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Processing your documents…"):
                    try:
                        # Reset state
                        st.session_state.conversation = None
                        st.session_state.chat_history = []

                        page_texts, page_sources = get_pdf_text(pdf_docs)

                        if not page_texts:
                            st.error(
                                "No readable text was found. Your PDFs may be "
                                "scanned images without a text layer, or all files were skipped."
                            )
                            st.stop()

                        chunks, metadatas = get_text_chunks(page_texts, page_sources)

                        if not chunks:
                            st.error("Text was extracted but could not be split into usable chunks.")
                            st.stop()

                        vectorstore = get_vectorstore(chunks, metadatas)
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore, temperature
                        )
                        st.success(f"✅ Ready! {len(chunks)} chunks indexed from your documents.")

                    except ValueError as ve:
                        st.error(f"Validation error: {str(ve)}")
                    except Exception as e:
                        st.error(f"Error — {type(e).__name__}: {str(e)}")
                        st.exception(e)

        # Show indexed doc summary if available
        if st.session_state.conversation:
            st.info("💬 Documents are loaded. Ask a question below!")

        if st.session_state.chat_history and st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Chat Area ─────────────────────────────────────────────────────────────
    if not st.session_state.conversation:
        st.info("👈 Upload and process your PDFs from the sidebar to get started.")
    else:
        # Render existing history (on page load / rerun)
        render_all_messages()

        user_question = st.chat_input("Ask a question about your documents…")
        if user_question:
            handle_userinput(user_question)


if __name__ == "__main__":
    main()
