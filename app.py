# app.py
import os
import uuid
import base64
import io
import fitz
import cohere
import chromadb
from PIL import Image
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# ========= API Keys =========
COHERE_API_KEY = os.getenv("ZH6spcBGDJILqDcZQ0yrZd8N4e6F0njJf2B4aFRN")
GEMINI_API_KEY = os.getenv("AIzaSyCC61HUPMORqWaDJPS6wEkO8ms-QaI4OAE")

co = cohere.ClientV2(api_key=COHERE_API_KEY)
from google import generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# ========= ChromaDB =========
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="Multimodal_embeddings")

# ========= Image Utils =========
max_pixels = 1568 * 1568

def resize_image(pil_image):
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path):
    pil_image = Image.open(img_path).convert("RGB")
    resize_image(pil_image)
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        return "data:image/png;base64," + base64.b64encode(img_buffer.read()).decode("utf-8")

# ========= LLM Setup =========
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers using ONLY the provided context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

response_chain = (
    RunnablePassthrough()
    | (lambda inputs: {"question": inputs["question"], "context": inputs["context"]})
    | prompt_template
    | llm
    | StrOutputParser()
)

# ========= Core Functions =========
def process_pdf(pdf_file):
    """Extracts text + images, stores embeddings in ChromaDB."""
    doc = fitz.open(pdf_file.name)
    image_folder = "./pdf_images"
    os.makedirs(image_folder, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            res = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                texts=[text]
            )
            emb = res.embeddings.float[0]
            collection.add(
                ids=[f"text_{uuid.uuid4()}"],
                documents=[text],
                embeddings=[emb],
                metadatas=[{"type": "text"}]
            )

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"page{page_num+1}_img{img_index+1}.{img_ext}"
            img_path = os.path.join(image_folder, img_filename)

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            base64_img = base64_from_image(img_path)
            api_input = {"content": [{"type": "image", "image": base64_img}]}
            res = co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input]
            )
            emb = res.embeddings.float[0]
            collection.add(
                ids=[f"image_{uuid.uuid4()}"],
                documents=[img_path],
                embeddings=[emb],
                metadatas=[{"type": "image"}]
            )
    return "PDF processed successfully ✅"

def search_and_answer(question, top_k=2):
    """Search in ChromaDB and generate response using Gemini."""
    q_res = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[question]
    )
    query_emb = q_res.embeddings.float[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k * 2)

    top_texts, top_images = [], []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if meta["type"] == "text":
            top_texts.append(doc)
        elif meta["type"] == "image":
            top_images.append(doc)

    top_texts = top_texts[:top_k]
    top_images = top_images[:top_k]

    context = "\n\n".join(top_texts + [f"[Image: {img}]" for img in top_images])
    response_text = response_chain.invoke({"question": question, "context": context})

    return response_text, top_images

# ========= Gradio UI =========
def chatbot_interface(pdf, question):
    if pdf is not None:
        process_pdf(pdf)

    if question.strip():
        answer, images = search_and_answer(question)
        return answer, images
    return "Please upload a PDF and ask a question.", []

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a question")],
    outputs=[gr.Textbox(label="Answer"), gr.Gallery(label="Relevant Images")],
    title="ASKPDF Multimodal RAG Chatbot",
    description="Upload a PDF and ask questions. The bot retrieves text + images and answers using Gemini."
)

if __name__ == "__main__":
    iface.launch()
