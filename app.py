import streamlit as st
import uuid
from core_functions import process_pdf, search_and_answer, get_collection, cleanup
from streamlit_cookies_manager import EncryptedCookieManager

# ===== CONFIG =====
st.set_page_config(page_title="RAG PDF Chat", layout="wide", page_icon="🤖")

# ===== HEADER =====
st.markdown(
    """
    <div style="background-color:#4B9CD3;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🤖 RAG · PDF Chat</h1>
        <p style="color:#e0e0e0;text-align:center;">Multimodal document intelligence with PDF & Images</p>
    </div>
    """, 
    unsafe_allow_html=True
)


# ===== COOKIES =====
cookies = EncryptedCookieManager(
    prefix="rag_pdf_chat", 
    password="a_super_secret_password"
)

if not cookies.ready():
    st.stop()

# ===== SESSION ID =====
session_id = cookies.get("session_id")
if not session_id:
    session_id = str(uuid.uuid4())
    cookies["session_id"] = session_id
    cookies.save()

st.markdown(f"<p style='color:#333;font-weight:bold;'>Current Session ID: <span style='color:#d9534f'>{session_id}</span></p>", unsafe_allow_html=True)

# ===== STATES =====
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "images" not in st.session_state:
    st.session_state.images = []
if "source" not in st.session_state:
    st.session_state.source = None

# ===== FILE UPLOAD =====
st.subheader("📄 Upload PDF Document")
upload_file = st.file_uploader("Upload PDF", type=["pdf"])
if st.button("Upload PDF"):
    if not upload_file:
        st.warning("⚠️ Please select a PDF first")
    else:
        with st.spinner("Processing PDF... This may take a few seconds."):
            try:
                msg = process_pdf(pdf_file=upload_file, session_id=session_id)
                st.success(f"✅ {msg}")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

# ===== ASK QUESTION =====
st.subheader("❓ Ask a Question")
question = st.text_input("Enter your question")
if st.button("Ask"):
    if not question:
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("Thinking..."):
            try:
                answer, images, source = search_and_answer(
                    question=question,
                    session_id=session_id,
                    top_k=2
                )

                st.session_state.answer = answer
                st.session_state.images = images
                st.session_state.source = source

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ===== ACTIONS =====
col1, col2 = st.columns(2)

with col1:
 if st.button("🧹 Clear Session", help="Clear all uploaded PDFs and session data"):
    cleanup()  
    st.session_state.answer = ""
    st.session_state.images = []
    

    new_session_id = str(uuid.uuid4())
    cookies["session_id"] = new_session_id
    cookies.save()
    st.success("Session cleared and new session created!")


    st.write(
        f"""
        <script>
        // Update localStorage too
        localStorage.setItem('session_id', '{new_session_id}');
        window.location.reload();
        </script>
        """,
        unsafe_allow_html=True
    )

with col2:
    if st.button("📊 Check Status", help="Check vector database status"):
        try:
            collection = get_collection(session_id)
            count = collection.count()
            st.info(f"📄 Documents: {count} | Has Data: {'✅' if count > 0 else '❌'}")
        except:
            st.error(f"Error checking status: {e}")

# ===== ANSWER =====
if st.session_state.answer:
    st.markdown("### 🧠 Answer")
    st.markdown(
    f"<div style='font-family:Arial; font-size:16px; color:#000000; background-color:#f0f8ff; padding:15px; border-radius:8px'>{st.session_state.answer}</div>",
    unsafe_allow_html=True
    )
    src_label = {
        "empty_db": "❌ No data available",
        "vector_db": "📄 PDF (RAG)",
        None: "🤖 LLM Knowledge"
    }
    st.caption(src_label.get(st.session_state.source, "🤖 AI Response"))


# ===== IMAGES =====
if st.session_state.images:
    st.markdown(f"### 🖼 Related Images ({len(st.session_state.images)})")
    cols = st.columns(3)
    for i, img in enumerate(st.session_state.images):
        with cols[i % 3]:
            st.image(img, use_container_width=True)