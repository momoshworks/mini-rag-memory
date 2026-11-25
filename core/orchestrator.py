import os
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient
from .vector_store import MiniVectorStore, SearchResult
from .dynamic_memory import DynamicMemory, MemoryExtractor

class AssistantOrchestrator:
    """
    Main orchestrator that coordinates:
    - Dynamic Memory (user profiling, fact extraction, contradiction resolution)
    - RAG Knowledge Base (TF-IDF semantic retrieval from documents)
    - LLM Inference (local Ollama or remote Gemini)
    """
    def __init__(
        self,
        data_dir: str = "data",
        llm_client: Optional[LLMClient] = None,
        model_name: str = "qwen2.5:3b"
    ):
        self.data_dir = data_dir
        self.docs_dir = os.path.join(data_dir, "documents")
        self.memory_path = os.path.join(data_dir, "memory_store.json")
        self.vector_path = os.path.join(data_dir, "vector_index.json")

        # 1. Initialize LLM (prefer 3b for higher reasoning capacity)
        self.llm = llm_client or LLMClient(model=model_name)

        # 2. Initialize RAG Vector Store
        self.vector_store = MiniVectorStore()
        if os.path.exists(self.vector_path):
            self.vector_store.load(self.vector_path)

        # 3. Initialize Dynamic Memory & Extractor
        self.memory = DynamicMemory()
        if os.path.exists(self.memory_path):
            self.memory.load_from_file(self.memory_path)
        self.extractor = MemoryExtractor(self.llm)

    def index_documents_directory(self) -> int:
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            return 0

        total_chunks = 0
        for filename in os.listdir(self.docs_dir):
            if filename.lower().endswith(self.vector_store.SUPPORTED_EXTENSIONS):
                full_path = os.path.join(self.docs_dir, filename)
                chunks_added = self.vector_store.add_file(full_path)
                total_chunks += chunks_added

        self.vector_store.save(self.vector_path)
        return total_chunks

    def add_knowledge(self, text: str, source: str = "manual"):
        self.vector_store.add_text(text, source=source)
        self.vector_store.save(self.vector_path)

    def process_message(self, user_message: str, top_k_rag: int = 3) -> Dict[str, Any]:
        # --- 1. RAG Retrieval ---
        search_results: List[SearchResult] = self.vector_store.search(user_message, top_k=top_k_rag)
        if search_results:
            rag_context = "\n---\n".join([
                f"[نص مسترجع من: {r.chunk.source}]\n{r.chunk.text}"
                for r in search_results
            ])
        else:
            rag_context = "لا توجد مستندات خارجية مطابقة."

        # --- 2. Dynamic Memory Context ---
        memory_context = self.memory.get_memory_context()

        # --- 3. Build Strict, Clear Prompt ---
        system_prompt = (
            "أنت مساعد ذكي ومتخصص يمتلك مصدرين للمعرفة:\n"
            "المصدر الأول: [الذاكرة الديناميكية للمستخدم] - تحتوي على أحدث الحقائق والتعديلات عن المستخدم (مكان إقامته، لغته المفضلة، اسمه).\n"
            "المصدر الثاني: [قاعدة المعرفة المسترجعة RAG] - تحتوي على نصوص السياسات والمعمارية التقنية للشركة.\n\n"
            "=== الذاكرة الديناميكية للمستخدم ===\n"
            f"{memory_context}\n\n"
            "=== قاعدة المعرفة المسترجعة (RAG) ===\n"
            f"{rag_context}\n\n"
            "تعليمات الإجابة الحاسمة:\n"
            "1. أجب بشكل مباشر ودقيق بناءً على النصوص والحقائق الموضحة أعلاه فقط.\n"
            "2. إذا سأل المستخدم عن معلوماته الشخصية (أين يعيش، لغته المفضلة)، استخرج الإجابة من [الذاكرة الديناميكية].\n"
            "3. إذا سأل المستخدم عن سياسات الشركة، الدوام، الميزانية، أو معمارية المشاريع، استخرج الأرقام والتفاصيل فوراً من [قاعدة المعرفة المسترجعة RAG].\n"
            "4. لا تُحِل المستخدم لقراءة الملفات ولا تعتذر إذا كانت المعلومات متوفرة في النصوص أعلاه."
        )

        prompt_parts = []
        for turn in self.memory.history:
            role_label = "المستخدم" if turn["role"] == "user" else "المساعد"
            prompt_parts.append(f"{role_label}: {turn['content']}")

        prompt_parts.append(f"المستخدم: {user_message}\nالمساعد:")
        full_user_prompt = "\n".join(prompt_parts)

        # --- 4. Generate Response ---
        response_text = self.llm.generate(
            prompt=full_user_prompt,
            system=system_prompt,
            temperature=0.2
        )

        # --- 5. Update Short-term History ---
        self.memory.add_message("user", user_message)
        self.memory.add_message("assistant", response_text)

        # --- 6. Dynamic Memory Extraction & Updating ---
        memory_updates = self.extractor.extract_and_update(
            user_message=user_message,
            assistant_response=response_text,
            memory=self.memory
        )

        # --- 7. Persist Memory & Vector Store ---
        self.memory.save_to_file(self.memory_path)

        return {
            "response": response_text,
            "rag_sources": [r.to_dict() for r in search_results],
            "memory_updates": memory_updates,
            "active_memory": self.memory.to_dict()
        }
