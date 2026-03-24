import io
import os
import uuid
import base64
import fitz
import cohere
import chromadb
from dotenv import load_dotenv
from bytez import Bytez
from flask import Flask,session,request
from datetime import datetime, time, timedelta
import threading
from PIL import Image
import time
from groq import Groq


load_dotenv()


# Clients
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
# Bytez Setup
#sdk = Bytez(os.getenv("BYTEZ_API_KEY"))
#llm = sdk.model("meta-llama/Llama-3.2-11B-Vision-Instruct")
# llm.load()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))



# ChromaDB Cloud
chroma_client = chromadb.CloudClient(
  api_key=os.getenv("CHROMA_API_KEY"),
  tenant=os.getenv("CHROMA_TENANT"),
  database=os.getenv("CHROMA_DATABASE")
)

collection_registry = {}

def get_collection():
    session_id = None

    # Handle JSON request
    if request.is_json:
        data = request.get_json()
        session_id = data.get("session_id")

    # Handle form-data (upload)
    if not session_id:
        session_id = request.form.get("session_id")

    if not session_id:
        return None
        # raise ValueError("No session_id provided")

    name = f"user_{session_id}"

    try:
        return chroma_client.get_collection(name)
    except:
        collection = chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        collection_registry[name] = datetime.now()
        return collection

# Image -> Base64 
def image_to_base64(image_bytes, img_ext="jpeg"):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{img_ext};base64,{encoded}"


# Embedding Function
def process_pdf(pdf_file):
    collection  = get_collection()
    pdf_bytes   = pdf_file.read()
    doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_count  = 0
    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract Text
        text = page.get_text().strip()
        if text:
            try:
                res = co.embed(
                    model="embed-v4.0",
                    input_type="search_document",
                    embedding_types=["float"],
                    output_dimension=1024,
                    inputs=[{
                        "content": [{"type": "text", "text": text}]
                    }]
                )
                collection.add(
                    ids=[f"text_{uuid.uuid4()}"],
                    documents=[text],
                    embeddings=[res.embeddings.float[0]],
                    metadatas=[{
                        "type": "text",
                        "page": page_num + 1
                    }]
                )
                text_count += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Text skip page {page_num+1}: {e}")
                continue

        # Extract Images
        for img in page.get_images(full=True):
            try:
                xref        = img[0]
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_ext     = base_image["ext"]

                # Size filter - skip tiny images
                pil_img = Image.open(io.BytesIO(image_bytes))
                w, h    = pil_img.size
                if w < 100 or h < 100:
                    continue

                # Compress image to fit ChromaDB limit
                compressed  = compress_image(image_bytes)
                compressed_b64 = (
                  f"data:image/jpeg;base64,"
                  f"{base64.b64encode(compressed).decode('utf-8')}"
               )

               
                verify = Image.open(io.BytesIO(compressed))
                pixels = list(verify.getdata())
                all_black = all(p == (0, 0, 0) or p == 0 for p in pixels[:100])
                if all_black:
                    print(f"⚠️ Black image detected page {page_num+1} - skipping")
                    continue

                # Embed original quality
                original_b64 = (
                    f"data:image/{img_ext};base64,"
                    f"{base64.b64encode(image_bytes).decode('utf-8')}"
                )

                res = co.embed(
                    model="embed-v4.0",
                    input_type="search_document",
                    embedding_types=["float"],
                    output_dimension=1024,
                    inputs=[{
                        "content": [{"type": "image", "image": original_b64}]
                    }]
                )

                # Store compressed image in metadata
                collection.add(
                    ids=[f"image_{uuid.uuid4()}"],
                    documents=[f"image_page_{page_num+1}"],  
                    embeddings=[res.embeddings.float[0]],
                    metadatas=[{
                        "type":  "image",
                        "page":  page_num + 1,
                        "image": compressed_b64              
                    }]
                )
                image_count += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"⚠️ Image skip page {page_num+1}: {e}")
                continue

    doc.close()
    return f"✅ Processed {text_count} text chunks, {image_count} images"


def compress_image(image_bytes, max_kb=12):
    """Compress image while keeping it visible."""
    
    image = Image.open(io.BytesIO(image_bytes))
    
    print(f"📐 Original: {image.size} | {image.mode} | {len(image_bytes)/1024:.1f}KB")

    # Step 1 - Convert mode
    if image.mode == "RGBA":
        # White background for transparency
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode in ("P", "CMYK", "L", "LA"):
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Step 2 - Resize keeping aspect ratio
    max_dim = 300   # bigger than before
    if image.width > max_dim or image.height > max_dim:
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)

    print(f"📐 Resized to: {image.size}")

    # Step 3 - Compress with decent quality
    for quality in [85, 75, 65, 55, 45]:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        size_kb = buffer.tell() / 1024
        print(f"🗜️  Quality {quality}: {size_kb:.1f}KB")
        
        if size_kb <= max_kb:
            buffer.seek(0)
            result = buffer.read()
            print(f"✅ Final size: {len(result)/1024:.1f}KB")
            return result

    # Last resort - still keep quality at 40 minimum
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=40)
    buffer.seek(0)
    return buffer.read()


def search_and_answer(question, top_k=2):
    collection = get_collection()
    answer = "Please provide a PDF document first."  
    
    if collection.count() == 0:
        return answer, [], "empty_db"
    
    # Embed Question
    q_res = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=1024,
        inputs=[{
            "content": [{"type": "text", "text": question}]
        }]
    )
    query_emb = q_res.embeddings.float[0]
    
    results = collection.query(
        query_embeddings=[query_emb], 
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    texts  = []
    images = []
    scores = results.get("distances", [[]])[0]
    
    for doc, meta, score  in zip(
        results["documents"][0], 
        results["metadatas"][0],
        scores
    ):
        if not doc:
            continue
        
        print(f"📌 Type: {meta['type']} | Score: {score:.4f}")
        
        if meta["type"] == "text":
            if score is not None and score > 1.5:
                continue
            texts.append(doc)
        elif meta["type"] == "image":
            img_data = meta.get("image", "")
            if img_data and img_data.startswith("data:image"):
                images.append((img_data, score)) 
                print(f"✅ Image found page {meta.get('page')} score {score:.4f}")
            else:
                print(f"❌ Image missing data: keys={list(meta.keys())}")
    
    # Sort images by relevance score (lower = better)
    images.sort(key=lambda x: x[1])
    images = [img for img, _ in images]
    
    clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    is_from_db  = len(clean_texts) > 0 or len(images) > 0
    
    # rerank texts
    reranked_texts = clean_texts
    if is_from_db and len(clean_texts) > 1:
        try:
            reranked = co.rerank(
                model="rerank-v3.5",
                query=question,
                documents=clean_texts,
                top_n=min(top_k, len(clean_texts))
            )
            reranked_texts = [clean_texts[r.index] for r in reranked.results]
        except Exception as e:
            print(f"⚠️ Rerank failed: {e}")
            reranked_texts = clean_texts[:top_k]

        
    # Build Context
    context = "\n\n".join(reranked_texts)
        
    # Debug
    print(f"📄 Texts found : {len(clean_texts)}")
    print(f"🖼️  Images found: {len(images)}")
    print(f"📝 Context     : {context[:200]}...")

    if is_from_db:
        answer = generate_multimodal_answer(
            question=question,
            context=context,
            images=images[:5]
        )
        source = "vector_db"
    else:
        answer = generate_multimodal_answer(
            question=question,
            context="",
            images=[]
        )
        source = "llm"
    if not answer or "429" in str(answer) or "quota" in str(answer).lower():
        answer = "⚠️ AI model is temporarily unavailable. Please try again shortly."
        
    return answer,images[:5],source
    

# Multimodal query (text + image)
def generate_multimodal_answer(question, context, images=[]):

    # Case 1 — Note images found in prompt
    image_note = ""
    if images:
        image_note = f"\n\n[{len(images)} relevant image(s) found on related pages]"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer using the provided context only."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}{image_note}\n\nQuestion: {question}"
        }
    ]

    result = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    answer = extract_llm_response(result)

    if not answer:
        return "⚠️ Could not generate answer. Please try again."

    return answer
    
    
def extract_llm_response(result):
    """
    Extracts text from different LLM response formats:
    - Groq (object)
    - Bytez
    - OpenAI-like dict
    - List responses
    - Plain string
    """

    # Case 1 — Groq object response
    try:
        if hasattr(result, "choices"):
            return result.choices[0].message.content.strip()
    except Exception:
        pass

    # Case 2 — Bytez Response object
    if hasattr(result, 'output'):
        output = result.output
        if isinstance(output, dict):
            return output.get("content", "").strip()

    # Case 3 — list of responses
    if isinstance(result, list) and len(result) > 0:
        r = result[0]

        # Groq inside list (rare but safe)
        try:
            if hasattr(r, "choices"):
                return r.choices[0].message.content.strip()
        except Exception:
            pass

    # Case 4 — Bytez in list
        if hasattr(r, 'output'):
            output = r.output
            if isinstance(output, dict):
                return output.get("content", "").strip()

        # Plain dict
        if isinstance(r, dict):
            text = (
                r.get("text") or
                r.get("content") or
                str(r)
            )
            return text.strip()

        return str(r).strip()

    # Case 4 — dict (OpenAI style)
    elif isinstance(result, dict):
        try:
            return result['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, TypeError):
            pass

        text = (
            result.get("content") or
            result.get("text") or
            str(result)
        )
        return text.strip()

    # Case 5 — plain string
    elif isinstance(result, str):
        return result.strip()

    # ⚠️ Fallback
    print(f"⚠️ Unexpected format: {type(result)} → {result}")
    return str(result).strip()



def cleanup():
    """Clears all collections for a fresh start."""
    try:
        collections = chroma_client.list_collections()
        for col in collections:
            chroma_client.delete_collection(col.name)
        print("🧹 All collections deleted successfully ✅")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")



def auto_cleanup():
    """Runs in background - deletes old collections."""
    while True:
        now = datetime.now()
        for name, created_at in list(collection_registry.items()):
            if now - created_at > timedelta(hours=1):
                try:
                    chroma_client.delete_collection(name)
                    del collection_registry[name]
                    print(f"🗑️ Auto deleted: {name}")
                except:
                    pass
        
        threading.Event().wait(600)

cleanup_thread = threading.Thread(
    target=auto_cleanup,
    daemon=True     
)
cleanup_thread.start()