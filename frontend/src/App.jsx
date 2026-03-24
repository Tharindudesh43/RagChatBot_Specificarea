import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState(null);

  const API = "";

  //get if there is a session id in local storage, if not create one and return it
  const getSessionId = () => {
    let sessionId = localStorage.getItem("rag_session_id");

    if (!sessionId) {
      sessionId = crypto.randomUUID(); // generate unique ID
      localStorage.setItem("rag_session_id", sessionId);
    }

    return sessionId;
  };

  const currentSessionId = () => {
    return localStorage.getItem("rag_session_id") || "none";
  };

  // ── EXACT SAME LOGIC — untouched ──
  const handleUpload = async () => {
    if (!file) return alert("Select a PDF first");

    const formData = new FormData();
    formData.append("pdf", file);
    formData.append("session_id", getSessionId());

    setLoading(true);
    try {
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      alert(data.message || "Uploaded");
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    }
    setLoading(false);
  };

  const handleAsk = async () => {
    if (!question) return alert("Enter a question");

    setLoading(true);
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          session_id: getSessionId(),
        }),
      });

      const data = await res.json();
      setAnswer(data.answer);
      setImages(data.images || []);
      setSource(data.source);
      console.log("Answer source:", data);
    } catch (err) {
      console.error(err);
      alert("Error getting answer");
    }
    setLoading(false);
  };

  const handleStatus = async () => {
    try {
      const res = await fetch(`${API}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId(),
        }),
      });
      const data = await res.json();
      alert(
        `Session ID: ${data.session_id || "none"}` +
          `\nDocuments in Session: ${data.doc_count || 0}` +
          `\nHas Data: ${data.has_data ? "Yes" : "No"}`,
      );
    } catch (err) {
      console.error(err);
      alert("Error fetching status");
    }
  };

  const handleCleanup = async () => {
    await fetch(`${API}/cleanup`, {
      method: "POST",
    });
    localStorage.removeItem("rag_session_id");
    setAnswer("");
    setImages([]);
    setQuestion("");
    setFile(null);
    alert("Session cleared");
  };

  return (
    <>
      <style>{`
        

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: #e4eae8;
          font-family: 'DM Sans', sans-serif;
          min-height: 100vh;
        }

        .rag-root {
          min-height: 100vh;
          background: #e4eae8;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 36px 16px 48px;
          position: relative;
          overflow-x: hidden;
        }

        /* Ambient background blobs */
        .rag-root::before {
          content: '';
          position: fixed;
          top: -120px; left: -120px;
          width: 500px; height: 500px;
          background: radial-gradient(circle, rgba(185, 81, 16, 0.07) 0%, transparent 70%);
          pointer-events: none;
        }
        .rag-root::after {
          content: '';
          position: fixed;
          bottom: -100px; right: -100px;
          width: 400px; height: 400px;
          background: radial-gradient(circle, rgba(58, 190, 146, 0.05) 0%, transparent 70%);
          pointer-events: none;
        }

        /* Header */
        .rag-header {
          text-align: center;
          margin-bottom: 36px;
        }
        .rag-header h1 {
          font-family: 'Syne', sans-serif;
          font-size: clamp(26px, 5vw, 36px);
          font-weight: 800;
          color: #48a086;
          letter-spacing: -0.03em;
          line-height: 1;
        }
        .rag-header h1 span { color: #10b981; }
        .rag-header p {
          font-size: 12px;
          color: rgba(255,255,255,0.25);
          margin-top: 6px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        /* Card */
        .rag-card {
          width: 100%;
          max-width: 680px;
          background: rgba(255,255,255,0.025);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 20px;
          overflow: hidden;
          box-shadow: 0 5px 10px rgba(0,0,0,0.5), 0 0 0 1px rgba(16,185,129,0.04);
        }

        /* Section blocks inside card */
        .rag-section {
          padding: 22px 24px;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .rag-section:last-child { border-bottom: none; }

        .rag-label {
          font-size: 10px;
          font-weight: 600;
          color: rgba(0, 0, 0, 0.7);
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-bottom: 12px;
        }

        /* Upload row */
        .upload-row {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
        }

        .file-label {
          flex: 1;
          min-width: 0;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          background: rgba(232, 18, 18, 0.04);
          border: 1px dashed rgba(255, 0, 0, 0.1);
          border-radius: 11px;
          cursor: pointer;
          transition: all 0.2s;
          color: rgba(250, 1, 1, 0.4);
          font-size: 13px;
          overflow: hidden;
          white-space: nowrap;
          text-overflow: ellipsis;
        }
        .file-label:hover {
          border-color: rgba(255, 0, 0, 0.35);
          background: rgba(255, 0, 0, 0.05);
          color: rgba(250, 1, 1, 0.7);
        }
        .file-label svg { flex-shrink: 0; }
        .file-label input { display: none; }

        /* Buttons */
        .btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 10px 18px;
          border-radius: 11px;
          border: none;
          font-size: 13px;
          font-weight: 600;
          font-family: 'DM Sans', sans-serif;
          cursor: pointer;
          transition: all 0.18s ease;
          white-space: nowrap;
          flex-shrink: 0;
        }
        .btn:active { transform: scale(0.97); }

        .btn-primary {
          background: #10b981;
          color: #fff;
        }
        .btn-primary:hover { background: #0ea572; box-shadow: 0 4px 20px rgba(16,185,129,0.3); }
        .btn-primary:disabled { background: rgba(16,185,129,0.2); color: rgba(16,185,129,0.4); cursor: not-allowed; }

        .btn-danger {
          background: rgba(239,68,68,0.08);
          border: 1px solid rgba(239,68,68,0.2);
          color: rgba(239,68,68,0.65);
        }
        .btn-danger:hover { background: rgba(239,68,68,0.15); border-color: rgba(239,68,68,0.4); color: rgba(239,68,68,0.9); }

        /* Ask row */
        .ask-row {
          display: flex;
          gap: 10px;
          align-items: center;
        }

        .rag-input {
          flex: 1;
          background: rgba(13, 255, 0, 0.04);
          border: 1px solid rgba(86, 195, 27, 0.6);
          border-radius: 11px;
          padding: 11px 14px;
          color: #000000;
          font-size: 13px;
          font-family: 'DM Sans', sans-serif;
          outline: none;
          transition: border-color 0.2s;
        }
        .rag-input::placeholder { color: rgba(71, 145, 22, 0.9); }
        .rag-input:focus { border-color: rgba(88, 175, 17, 0.7); }

        /* Actions row */
        .actions-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 10px;
        }

        /* Loading bar */
        .loading-bar {
          height: 2px;
          background: linear-gradient(90deg, transparent, #10b981, transparent);
          background-size: 200% 100%;
          animation: shimmer 1.2s infinite;
        }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }

        .loading-text {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: rgba(16,185,129,0.7);
          padding: 10px 24px;
        }
        .dot { width: 5px; height: 5px; border-radius: 50%; background: #10b981; animation: bounce 1.2s ease-in-out infinite; }
        .dot:nth-child(2) { animation-delay: 0.15s; }
        .dot:nth-child(3) { animation-delay: 0.3s; }
        @keyframes bounce {
          0%,80%,100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-4px); opacity: 1; }
        }

        /* Answer block */
        .answer-block {
          animation: fadeUp 0.35s ease;
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .answer-tag {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: #10b981;
          margin-bottom: 10px;
        }
        .answer-tag::before {
          content: '';
          width: 5px; height: 5px;
          border-radius: 50%;
          background: #10b981;
          box-shadow: 0 0 6px #10b981;
        }

        .answer-text {
          font-size: 14px;
          color: rgba(255,255,255,0.8);
          line-height: 1.75;
        }

        /* Images grid */
        .images-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 10px;
        }
        .images-grid img {
          width: 100%;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.07);
          object-fit: cover;
          transition: transform 0.2s;
        }
        .images-grid img:hover { transform: scale(1.02); }

        /* File name chip */
        .file-chip {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          padding: 3px 10px;
          border-radius: 20px;
          background: rgba(16,185,129,0.1);
          border: 1px solid rgba(16,185,129,0.2);
          font-size: 11px;
          color: #10b981;
          margin-top: 8px;
          margin-left: 8px;
        }
      `}</style>

      <div className="rag-root">
        {/* Header */}
        <div className="rag-header">
          <h1>
            RAG<span>·</span>PDF{" "}
            <span style={{ fontSize: "0.6em", opacity: 0.8 }}>Chat</span>
          </h1>
          <p>Multimodal document intelligence</p>
        </div>

        {/* Card */}
        <div className="rag-card p-21">
          {/* ── Upload Section ── */}
          <div className="rag-section">
            <div className="rag-label">Upload Document</div>
            <div className="upload-row">
              <label className="file-label">
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
                  />
                </svg>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                  {file ? file.name : "Click to browse PDF..."}
                  {file && (
                    <>
                      <span className="file-chip">
                        {Math.round(file.size / 1024)} KB
                      </span>
                      <button
                        style={{
                          marginLeft: 8,
                          background: "transparent",
                          border: "none",
                          cursor: "pointer",
                          color: "rgba(255, 0, 0, 0.7)",
                        }}
                        className="bg-red-500 hover:bg-red-600 text-white font-bold py-1 px-2 rounded"
                        onClick={(e) => {
                          e.stopPropagation();
                          setFile(null);
                        }}
                      >
                        Clear
                      </button>
                    </>
                  )}
                </span>
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => setFile(e.target.files[0])}
                />
              </label>

              <button
                className="btn btn-primary"
                onClick={handleUpload}
                disabled={loading}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
                Upload PDF
              </button>
            </div>
          </div>

          {/* ── Ask Section ── */}
          <div className="rag-section">
            <div className="rag-label">Ask a Question</div>
            <div className="ask-row">
              <input
                type="text"
                placeholder="What does this document say about..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                className="rag-input"
              />
              <button
                className="btn btn-primary"
                onClick={handleAsk}
                disabled={loading}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
                  />
                </svg>
                Ask
              </button>
            </div>
          </div>

          {/* ── Actions Row ── */}
          <div className="rag-section">
            <div className="actions-col">
              <button className="btn btn-danger" onClick={handleCleanup}>
                <svg
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
                  />
                </svg>
                Clear Session
              </button>
              <span> </span>
              <button className="btn btn-primary" onClick={handleStatus}>
                <svg
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Check Session
              </button>
              {/* <span style={{fontSize:11, color:"rgba(255,255,255,0.15)"}}>Enter to send</span> */}
            </div>
          </div>

          {/* ── Loading ── */}
          {loading && (
            <>
              <div className="loading-bar" />
              <div className="loading-text">
                <div className="dot" />
                <div className="dot" />
                <div className="dot" />
                Processing your request...
              </div>
            </>
          )}

          {/* ── Answer ── */}
          {answer && (
            <div className="mt-4 p-4 bg-gray-700 rounded-md rag-section answer-block">
              <h2 className="font-semibold" style={{ color: "#000000" }}>
                Answer:
              </h2>

              <ReactMarkdown>{answer}</ReactMarkdown>

              <p className="text-xm mt-20 m-10 text-gray-500">
                Source:{" "}
                {source === "empty_db"
                  ? "❌ No data available"
                  : source === "vector_db"
                    ? "📄 PDF (RAG)"
                    : "🤖 LLM Knowledge"}
              </p>
            </div>
          )}

          {/* ── Images ── */}
          {images && images.length > 0 && (
            <div className="rag-section">
              <div className="rag-label">Related Images ({images.length})</div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
                  gap: 10,
                }}
              >
                {images.map((img, i) => (
                  <img
                    key={i}
                    src={img} // ✅ direct Data URL
                    alt={`result-${i}`}
                    style={{
                      width: "100%",
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.1)",
                      objectFit: "cover",
                    }}
                    onError={(e) => {
                      console.log("❌ Image failed to load:", i);
                      e.target.style.display = "none"; // hide broken images
                    }}
                    onLoad={() => console.log(`✅ Image ${i} loaded`)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
