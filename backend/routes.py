from flask import Flask,session, request, jsonify,send_from_directory
from app import process_pdf,search_and_answer,cleanup,get_collection
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.getenv("SECRET_KEY")
app.permanent_session_lifetime = timedelta(hours=1)


@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

# Handle SPA routing
@app.route("/<path:path>")
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")
    
@app.before_request
def make_session_permanent():
    session.permanent = True
    
    
@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    pdf = request.files['pdf']
    msg = process_pdf(pdf)
    return jsonify({"message": msg})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer, images,source = search_and_answer(question)
    return jsonify({
        "answer": answer,
        "images": images,
        "source": source
    })

@app.route('/status', methods=['POST'])
def status():
    data = request.json
    session_id = data.get('session_id')
    
    collection = get_collection()
    if collection is None:
        return jsonify({"error": "No active session"}), 400
    else:
        return jsonify({
        "session_id": session_id,
        "doc_count":  collection.count(),
        "has_data":   collection.count() > 0
        })


@app.route('/cleanup', methods=['POST'])
def end_session():
    cleanup()
    return jsonify({"status": "cleared ✅"})

@app.route('/debug', methods=['GET'])
def debug():
    collection = get_collection()
    peek = collection.peek(limit=10)
    
    image_count = sum(1 for m in peek['metadatas'] if m.get('type') == 'image')
    text_count  = sum(1 for m in peek['metadatas'] if m.get('type') == 'text')
    
    # Check if images have data
    image_has_data = [
        bool(m.get('image', ''))
        for m in peek['metadatas']
        if m.get('type') == 'image'
    ]
    
    return jsonify({
        "total":          collection.count(),
        "text_count":     text_count,
        "image_count":    image_count,
        "image_has_data": image_has_data,
        "metadatas":      peek['metadatas']
    })
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)