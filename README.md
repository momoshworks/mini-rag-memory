# 🧠 Mini RAG + Dynamic Memory

نظام ذكاء اصطناعي مدمج وخفيف الوزن يجمع بين **RAG (الاسترجاع المعزز بالتوليد)** و **Dynamic Memory (الذاكرة الحية والديناميكية)**.

---

## 🌟 مميزات النظام

1. **الذاكرة الديناميكية (Dynamic Memory Engine):**
   - استخراج تلقائي للحقائق الثابتة وتفضيلات المستخدم بعد كل جولة محادثة.
   - حل التناقضات وتحديث القديم (Update / Invalidate) عند تغيير المستخدم لرأيه أو ظروفه بدلاً من تكديس تاريخ المحادثة بدون تنظيم.
   - دعم الملاحظات السياقية (Episodic Notes) ونافذة المحادثة القريبة (Sliding Window Buffer).
   - حفظ مستمر ودائم في ملف `data/memory_store.json`.

2. **محرك الـ RAG الخفيف (Zero-Heavy-Dependency Vector Store):**
   - يدعم تقسيم المستندات النصية (`.md`, `.txt`, `.json`) إلى أجزاء دلالية مع تداخل (Chunking with Overlap).
   - بحث دلالي فائق السرعة عبر TF-IDF + Cosine Similarity مع دعم التوكين العربي والإنجليزي.
   - حفظ الفهرس في `data/vector_index.json`.

3. **تشغيل محلي بالكامل وبدون تكاليف (Local & Offline):**
   - يعمل مباشرة مع **Ollama** ونماذج مثل `qwen2.5:1.5b` أو `qwen2.5:3b`.
   - إمكانية التبديل إلى **Google Gemini API** في حال توفر المفتاح.

4. **واجهات استخدام متعددة:**
   - **`demo.py`**: سيناريو تجريبي آلي ومحكم يوضح تطور الذاكرة واسترجاع المستندات خطوة بخطوة.
   - **`app.py`**: محادثة تفاعلية في الطرفية (CLI) مع ألوان وأوامر تحكم (`/memory`, `/docs`, `/clear`).
   - **`api_server.py`**: واجهة REST API سريعة باستخدام **FastAPI** لربطه بأي تطبيق ويب أو موبايل.

---

## 📁 هيكلية المشروع

```
mini-rag-memory/
├── core/
│   ├── llm_client.py       # محرك الاتصال بـ Ollama / Gemini
│   ├── vector_store.py     # محرك RAG والبحث الدلالي
│   ├── dynamic_memory.py   # الذاكرة الحية ومستخرج الحقائق (Extractor)
│   └── orchestrator.py     # منسق النظام وسلسلة المعالجة
├── data/
│   ├── documents/          # مستندات المعرفة الخارجية (سياسات، معمارية...)
│   ├── memory_store.json   # ملف حفظ الذاكرة الديناميكية
│   └── vector_index.json   # فهرس الـ RAG
├── demo.py                 # عرض تجريبي آلي
├── app.py                  # طرفية تفاعلية (CLI)
├── api_server.py           # واجهة برمجية FastAPI
└── requirements.txt
```

---

## 🚀 كيفية التشغيل

### 1. تشغيل العرض التجريبي الآلي (Automated Demo)
```powershell
python d:\AntiyGravity\projects\mini-rag-memory\demo.py
```

### 2. تشغيل المحادثة التفاعلية في الطرفية (Interactive CLI)
```powershell
python d:\AntiyGravity\projects\mini-rag-memory\app.py
```

### 3. تشغيل الـ REST API (FastAPI)
```powershell
python d:\AntiyGravity\projects\mini-rag-memory\api_server.py
```
ثم فتح لوحة التحكم والتوثيق التفاعلي Swagger على المتصفح:
`http://127.0.0.1:8040` أو `http://127.0.0.1:8040/docs`
