import os
import sys
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import AssistantOrchestrator

app = FastAPI(
    title="Mini RAG + Dynamic Memory",
    description="نظام ذكاء اصطناعي مصغر يجمع بين RAG والذاكرة الحية مع قراءة المستندات",
    version="1.0.0"
)

# Initialize Orchestrator
data_dir = os.path.join(os.path.dirname(__file__), "data")
orchestrator = AssistantOrchestrator(data_dir=data_dir)
orchestrator.index_documents_directory()

class ChatRequest(BaseModel):
    message: str
    top_k_rag: Optional[int] = 3

class ChatResponse(BaseModel):
    response: str
    rag_sources: List[Dict[str, Any]]
    memory_updates: List[Dict[str, Any]]
    active_memory: Dict[str, Any]

@app.get("/api/status")
def status():
    return {
        "status": "online",
        "service": "Mini RAG + Dynamic Memory",
        "model": orchestrator.llm.model,
        "indexed_chunks": len(orchestrator.vector_store.chunks),
        "supported_extensions": list(orchestrator.vector_store.SUPPORTED_EXTENSIONS)
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    result = orchestrator.process_message(req.message, top_k_rag=req.top_k_rag)
    return result

@app.get("/api/memory")
def get_memory():
    return orchestrator.memory.to_dict()

@app.delete("/api/memory")
def clear_memory():
    orchestrator.memory.facts.clear()
    orchestrator.memory.notes.clear()
    orchestrator.memory.history.clear()
    orchestrator.memory.save_to_file(orchestrator.memory_path)
    return {"message": "تم تصفير الذاكرة بنجاح"}

@app.get("/api/documents")
def list_documents():
    docs = []
    if os.path.exists(orchestrator.docs_dir):
        docs = os.listdir(orchestrator.docs_dir)
    return {
        "files": docs,
        "total_chunks": len(orchestrator.vector_store.chunks),
        "supported_formats": list(orchestrator.vector_store.SUPPORTED_EXTENSIONS)
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in orchestrator.vector_store.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"صيغة الملف غير مدعومة. الصيغ المدعومة هي: {', '.join(orchestrator.vector_store.SUPPORTED_EXTENSIONS)}"
        )

    file_path = os.path.join(orchestrator.docs_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Index the newly uploaded file immediately into RAG
    chunks_added = orchestrator.vector_store.add_file(file_path)
    orchestrator.vector_store.save(orchestrator.vector_path)

    return {
        "message": "تم رفع وفهرسة الملف بنجاح",
        "filename": file.filename,
        "chunks_added": chunks_added,
        "total_chunks": len(orchestrator.vector_store.chunks)
    }

@app.post("/api/documents/reindex")
def reindex_documents():
    chunks = orchestrator.index_documents_directory()
    return {"message": "تمت إعادة الفهرسة بنجاح", "total_chunks": chunks}

# =========================================================================
# Web UI Dashboard (Single-file elegant interface)
# =========================================================================
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mini RAG + Dynamic Memory Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d0f17;
      --card-bg: #151926;
      --border: #242c42;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #a855f7;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --user-msg: #1e293b;
      --bot-msg: #132338;
      --badge-bg: rgba(56, 189, 248, 0.12);
      --badge-border: rgba(56, 189, 248, 0.3);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Cairo', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .logo-area { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
      width: 38px; height: 38px; border-radius: 10px;
      background: linear-gradient(135deg, #38bdf8, #a855f7);
      display: flex; align-items: center; justify-content: center;
      font-size: 20px; font-weight: bold; color: #fff;
    }
    .logo-text h1 { font-size: 18px; font-weight: 700; }
    .logo-text span { font-size: 12px; color: var(--text-muted); }
    .header-actions { display: flex; gap: 10px; align-items: center; }
    .btn {
      padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600;
      cursor: pointer; border: 1px solid var(--border); background: #1e2538;
      color: var(--text); font-family: inherit; transition: all 0.2s;
    }
    .btn:hover { background: #28334d; border-color: var(--primary); }
    .btn-primary { background: #0369a1; border-color: #0ea5e9; color: #fff; }
    .btn-primary:hover { background: #0284c7; }
    .btn-danger:hover { background: #451a1a; border-color: #ef4444; color: #fca5a5; }

    .main-layout { display: flex; flex: 1; overflow: hidden; }

    /* Chat Section */
    .chat-section {
      flex: 1.5; display: flex; flex-direction: column;
      border-left: 1px solid var(--border);
      background: #090b11;
    }
    .messages-container {
      flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px;
    }
    .msg-wrapper { display: flex; flex-direction: column; max-width: 85%; }
    .msg-user { align-self: flex-start; }
    .msg-bot { align-self: flex-end; }
    .msg-bubble {
      padding: 12px 18px; border-radius: 14px; font-size: 14.5px; line-height: 1.6;
    }
    .msg-user .msg-bubble {
      background: var(--user-msg); border: 1px solid #334155; border-top-right-radius: 2px;
    }
    .msg-bot .msg-bubble {
      background: var(--bot-msg); border: 1px solid #1e3a5f; border-top-left-radius: 2px;
    }
    .msg-sender { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
    .sources-tag {
      margin-top: 8px; font-size: 11.5px; color: #38bdf8; background: rgba(56, 189, 248, 0.08);
      padding: 4px 8px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2);
    }

    .input-bar {
      padding: 16px 20px; background: var(--card-bg); border-top: 1px solid var(--border);
      display: flex; gap: 10px;
    }
    .input-bar input {
      flex: 1; background: #0b0e17; border: 1px solid var(--border); border-radius: 10px;
      padding: 12px 16px; color: #fff; font-size: 14px; font-family: inherit; outline: none;
    }
    .input-bar input:focus { border-color: var(--primary); }
    .send-btn {
      background: var(--primary); color: #0f172a; border: none; font-weight: 700;
      padding: 0 20px; border-radius: 10px; cursor: pointer; transition: 0.2s;
    }
    .send-btn:hover { background: var(--primary-hover); }

    /* Side Panel */
    .side-panel {
      flex: 1; display: flex; flex-direction: column; background: var(--card-bg);
      overflow-y: auto; padding: 20px; gap: 20px;
    }
    .panel-card {
      background: #0f131e; border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    }
    .panel-header {
      font-size: 14px; font-weight: 700; color: var(--primary); margin-bottom: 12px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .fact-item {
      background: #171d2d; border: 1px solid #232c44; border-radius: 8px; padding: 8px 12px;
      margin-bottom: 8px; font-size: 13px; display: flex; justify-content: space-between;
    }
    .fact-key { color: #94a3b8; font-weight: 600; }
    .fact-val { color: #38bdf8; font-weight: 700; }
    .note-item {
      font-size: 12.5px; color: #cbd5e1; background: #171d2d; border-right: 3px solid var(--accent);
      padding: 6px 10px; margin-bottom: 6px; border-radius: 4px;
    }
    .doc-pill {
      font-size: 11.5px; background: #1e2638; border: 1px solid #2e3a54; padding: 4px 8px;
      border-radius: 6px; margin: 3px; display: inline-block; color: #a5b4fc;
    }
    .upload-zone {
      border: 2px dashed #2e3a54; border-radius: 10px; padding: 14px; text-align: center;
      margin-top: 10px; background: #0c0f18; cursor: pointer; transition: 0.2s;
    }
    .upload-zone:hover { border-color: var(--primary); background: #111624; }
    .upload-zone span { font-size: 12px; color: var(--text-muted); }
  </style>
</head>
<body>
  <header>
    <div class="logo-area">
      <div class="logo-icon">🧠</div>
      <div class="logo-text">
        <h1>Mini RAG + Dynamic Memory</h1>
        <span id="model-tag">جاري التحميل...</span>
      </div>
    </div>
    <div class="header-actions">
      <input type="file" id="file-uploader" style="display:none" onchange="handleFileUpload(this)">
      <button class="btn btn-primary" onclick="document.getElementById('file-uploader').click()">📁 رفع مستند (PDF/Word/TXT)</button>
      <button class="btn" onclick="reindexDocs()">🔄 إعادة فهرسة RAG</button>
      <button class="btn btn-danger" onclick="clearMemory()">🗑️ تصفير الذاكرة</button>
      <a href="/docs" target="_blank" class="btn">📖 API Docs</a>
    </div>
  </header>

  <div class="main-layout">
    <!-- Chat Section -->
    <div class="chat-section">
      <div class="messages-container" id="chat-box">
        <div class="msg-wrapper msg-bot">
          <div class="msg-sender">المساعد الذكي</div>
          <div class="msg-bubble">
            أهلاً بك! أنا مساعد ذكي أمتلك ذاكرة ديناميكية تتذكر معلوماتك وتحدثها باستمرار، بالإضافة إلى محرك RAG يسترجع المعلومات من وثائق الشركة بدقة. يمكنك أيضاً رفع ملفات PDF أو Word وسأقرؤها فوراً!
          </div>
        </div>
      </div>
      <div class="input-bar">
        <input type="text" id="user-input" placeholder="اكتب سؤالك أو حدّث بياناتك هنا..." onkeydown="if(event.key==='Enter') sendMessage()">
        <button class="send-btn" id="send-button" onclick="sendMessage()">إرسال</button>
      </div>
    </div>

    <!-- Live Dynamic Memory & RAG Panel -->
    <div class="side-panel">
      <!-- Active Memory -->
      <div class="panel-card">
        <div class="panel-header">
          <span>📌 الذاكرة الحية للمستخدم (Dynamic Memory)</span>
          <span style="font-size:11px;color:#38bdf8;" id="fact-count">0 حقائق</span>
        </div>
        <div id="facts-list">
          <p style="font-size:12px;color:var(--text-muted);">لا توجد حقائق مسجلة بعد. عرّف نفسك في الشات!</p>
        </div>
      </div>

      <!-- Episodic Notes -->
      <div class="panel-card">
        <div class="panel-header">
          <span>📝 ملاحظات وسياق ديناميكي</span>
        </div>
        <div id="notes-list">
          <p style="font-size:12px;color:var(--text-muted);">لا توجد ملاحظات حالياً.</p>
        </div>
      </div>

      <!-- Indexed Documents -->
      <div class="panel-card">
        <div class="panel-header">
          <span>📚 قاعدة المعرفة المفهرسة (RAG)</span>
          <span style="font-size:11px;color:#a855f7;" id="chunks-count">0 Chunks</span>
        </div>
        <div id="docs-list"></div>

        <div class="upload-zone" onclick="document.getElementById('file-uploader').click()">
          <p style="font-size:13px;font-weight:600;color:#38bdf8;margin-bottom:2px;">+ اضغط لرفع ملف جديد</p>
          <span>يدعم: PDF, DOCX, TXT, MD, CSV, JSON</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('model-tag').innerText = `النموذج: ${data.model} | Chunks: ${data.indexed_chunks}`;
      } catch (e) {}
    }

    async function refreshMemory() {
      try {
        const res = await fetch('/api/memory');
        const data = await res.json();
        
        // Facts
        const factsList = document.getElementById('facts-list');
        const factKeys = Object.keys(data.facts || {});
        document.getElementById('fact-count').innerText = `${factKeys.length} حقائق`;
        
        if (factKeys.length === 0) {
          factsList.innerHTML = '<p style="font-size:12px;color:var(--text-muted);">لا توجد حقائق مسجلة بعد.</p>';
        } else {
          factsList.innerHTML = factKeys.map(k => {
            const f = data.facts[k];
            return `<div class="fact-item">
              <span class="fact-key">${f.key}:</span>
              <span class="fact-val">${f.value}</span>
            </div>`;
          }).join('');
        }

        // Notes
        const notesList = document.getElementById('notes-list');
        const notes = data.notes || [];
        if (notes.length === 0) {
          notesList.innerHTML = '<p style="font-size:12px;color:var(--text-muted);">لا توجد ملاحظات مسجلة.</p>';
        } else {
          notesList.innerHTML = notes.map(n => `<div class="note-item">• ${n}</div>`).join('');
        }
      } catch (e) {}
    }

    async function refreshDocs() {
      try {
        const res = await fetch('/api/documents');
        const data = await res.json();
        document.getElementById('chunks-count').innerText = `${data.total_chunks} أجزاء`;
        const docsList = document.getElementById('docs-list');
        docsList.innerHTML = (data.files || []).map(f => {
          let icon = '📄';
          if (f.endsWith('.pdf')) icon = '📕';
          else if (f.endsWith('.docx')) icon = '📘';
          return `<span class="doc-pill">${icon} ${f}</span>`;
        }).join(' ');
      } catch (e) {}
    }

    async function handleFileUpload(input) {
      if (!input.files || input.files.length === 0) return;
      const file = input.files[0];
      const formData = new FormData();
      formData.append('file', file);

      const chatBox = document.getElementById('chat-box');
      chatBox.innerHTML += `
        <div class="msg-wrapper msg-user">
          <div class="msg-sender">أنت</div>
          <div class="msg-bubble">📁 جاري رفع وفهرسة الملف: <strong>${file.name}</strong>...</div>
        </div>
      `;
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });
        const result = await res.json();

        if (res.ok) {
          chatBox.innerHTML += `
            <div class="msg-wrapper msg-bot">
              <div class="msg-sender">المساعد الذكي</div>
              <div class="msg-bubble">✔ تم استلام وقراءة الملف <strong>${file.name}</strong> بنجاح! تم تقسيمه إلى ${result.chunks_added} أجزاء دلالية وإضافته فوراً إلى قاعدة معرفة RAG. يمكنك الآن سؤالي عن أي تفاصيل داخله!</div>
            </div>
          `;
          refreshDocs();
          refreshStatus();
        } else {
          chatBox.innerHTML += `
            <div class="msg-wrapper msg-bot">
              <div class="msg-sender">المساعد الذكي</div>
              <div class="msg-bubble" style="color:#f87171;">فشل رفع الملف: ${result.detail || 'خطأ غير معروف'}</div>
            </div>
          `;
        }
      } catch (e) {
        alert('حدث خطأ أثناء رفع الملف: ' + e);
      }
      input.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      const chatBox = document.getElementById('chat-box');
      chatBox.innerHTML += `
        <div class="msg-wrapper msg-user">
          <div class="msg-sender">أنت</div>
          <div class="msg-bubble">${text}</div>
        </div>
      `;
      input.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;

      const loadingId = 'loading-' + Date.now();
      chatBox.innerHTML += `
        <div class="msg-wrapper msg-bot" id="${loadingId}">
          <div class="msg-sender">المساعد الذكي</div>
          <div class="msg-bubble">⏳ جاري الاسترجاع والتفكير...</div>
        </div>
      `;
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();

        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();

        let sourcesHtml = '';
        if (data.rag_sources && data.rag_sources.length > 0) {
          sourcesHtml = `<div class="sources-tag">📚 مصادر RAG: ${data.rag_sources.map(s => `${s.source} (${(s.score*100).toFixed(0)}%)`).join(' | ')}</div>`;
        }

        chatBox.innerHTML += `
          <div class="msg-wrapper msg-bot">
            <div class="msg-sender">المساعد الذكي</div>
            <div class="msg-bubble">${data.response}${sourcesHtml}</div>
          </div>
        `;
        chatBox.scrollTop = chatBox.scrollHeight;

        refreshMemory();
      } catch (err) {
        const loader = document.getElementById(loadingId);
        if (loader) loader.innerHTML = `<div class="msg-bubble" style="color:#f87171;">حدث خطأ أثناء الاتصال بالخادم.</div>`;
      }
    }

    async function clearMemory() {
      if (confirm('هل تريد بالتأكيد تصفير الذاكرة الديناميكية؟')) {
        await fetch('/api/memory', { method: 'DELETE' });
        refreshMemory();
        alert('تم تصفير الذاكرة بنجاح.');
      }
    }

    async function reindexDocs() {
      await fetch('/api/documents/reindex', { method: 'POST' });
      refreshDocs();
      refreshStatus();
      alert('تمت إعادة الفهرسة بنجاح.');
    }

    refreshStatus();
    refreshMemory();
    refreshDocs();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    server_port = int(os.getenv("PORT", "8040"))
    uvicorn.run("api_server:app", host="127.0.0.1", port=server_port, reload=False)
