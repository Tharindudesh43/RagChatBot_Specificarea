import streamlit as st
import uuid
from core_functions import process_pdf, search_and_answer, get_collection, cleanup
from streamlit_cookies_manager import EncryptedCookieManager

# ===== CONFIG =====
st.set_page_config(page_title="RAG PDF Chat", layout="centered")
st.title("🤖 RAG · PDF Chat")
st.caption("Multimodal document intelligence")

# ===== COOKIES =====
cookies = EncryptedCookieManager(
    prefix="rag_pdf_chat", 
    password="a_super_secret_password"  # change this to something secure
)

# Wait until cookies are ready
if not cookies.ready():
    st.stop()

# ===== SESSION ID =====
session_id = cookies.get("session_id")
if not session_id:
    session_id = str(uuid.uuid4())
    cookies["session_id"] = session_id
    cookies.save()

st.write(f"Current Session ID: {session_id}")

# ===== STATES =====
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "images" not in st.session_state:
    st.session_state.images = []
if "source" not in st.session_state:
    st.session_state.source = None

# ===== FILE UPLOAD =====
st.subheader("📄 Upload Document")
upload_file = st.file_uploader("Upload PDF", type=["pdf"])
if st.button("Upload PDF"):
    if not upload_file:
        st.warning("Select a PDF first")
    else:
        with st.spinner("Processing PDF..."):
            try:
                msg = process_pdf(pdf_file=upload_file, session_id=session_id)
                st.success(msg)
            except Exception as e:
                st.error(f"Error: {e}")

# ===== ASK QUESTION =====
st.subheader("❓ Ask a Question")
question = st.text_input("Enter your question")
if st.button("Ask"):
    if not question:
        st.warning("Enter a question")
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
                st.error(f"Error: {e}")

# ===== ACTIONS =====
col1, col2 = st.columns(2)

with col1:
 if st.button("🧹 Clear Session"):
    cleanup()  # clear the collection or session data
    st.session_state.answer = ""
    st.session_state.images = []
    
    # Generate a new session_id and store in cookies/localStorage
    new_session_id = str(uuid.uuid4())
    cookies["session_id"] = new_session_id
    cookies.save()

    # Force Streamlit to reload the page
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
    if st.button("📊 Check Status"):
        try:
            collection = get_collection(session_id)
            count = collection.count()
            st.info(f"Docs: {count} | Has Data: {count > 0}")
        except:
            st.error("Error getting status")

# ===== ANSWER =====
if st.session_state.answer:
    st.subheader("🧠 Answer")
    st.markdown(st.session_state.answer)

    source = st.session_state.source
    if source == "empty_db":
        st.caption("❌ No data available")
    elif source == "vector_db":
        st.caption("📄 PDF (RAG)")
    else:
        st.caption("🤖 LLM Knowledge")

# ===== IMAGES =====
if st.session_state.images:
    st.subheader(f"🖼 Images ({len(st.session_state.images)})")
    cols = st.columns(3)
    for i, img in enumerate(st.session_state.images):
        with cols[i % 3]:
            st.image(img, use_container_width=True)