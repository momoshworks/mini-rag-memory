import os
import sys
import json
import time

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import AssistantOrchestrator
from core.llm_client import LLMClient

def print_separator(title: str):
    print("\n" + "=" * 65)
    print(f"🔹 {title}")
    print("=" * 65)

def run_demo():
    print_separator("تهيئة نظام RAG + Dynamic Memory الصغير")
    
    # Clean previous memory store for demo purposes
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    mem_file = os.path.join(data_dir, "memory_store.json")
    if os.path.exists(mem_file):
        os.remove(mem_file)

    # Initialize Orchestrator
    orchestrator = AssistantOrchestrator(data_dir=data_dir)
    print(f"✔ تم الاتصال بالنموذج اللغوي: {orchestrator.llm.model} (عبر {orchestrator.llm.provider})")

    # Step 1: Index Documents (RAG)
    print_separator("الخطوة 1: فهرسة مستندات قاعدة المعرفة (RAG)")
    total_chunks = orchestrator.index_documents_directory()
    print(f"✔ تمت فهرسة المستندات بنجاح! إجمالي الأجزاء الدلالية (Chunks): {total_chunks}")

    # Simulated Conversational Script demonstrating Dynamic Memory Evolution
    dialogue_script = [
        {
            "title": "الجولة 1: تعريف المستخدم بنفسه (بناء الذاكرة الأولية)",
            "user": "أهلاً بك، أنا المهندس عمر، أعيش حالياً في الرياض وأفضل دائماً البرمجة بلغة بايثون."
        },
        {
            "title": "الجولة 2: استرجاع معرفة وثائقية (اختبار RAG)",
            "user": "ما هي ميزانية التعلم السنوية المخصصة لي في الشركة، وكم يوماً يُسمح لي بالعمل عن بعد؟"
        },
        {
            "title": "الجولة 3: تحديث ديناميكي وتعديل حقائق سابقة (Dynamic Memory Update & Contradiction)",
            "user": "تحديث سريع: لقد انتقلت للعيش في دبي، وأصبحت أفضل لغة Rust بدلاً من بايثون."
        },
        {
            "title": "الجولة 4: دمج الذاكرة الديناميكية المحدثة مع قاعدة المعرفة المسترجعة (RAG + Memory)",
            "user": "ذكّرني، أين أعيش الآن وما هي لغتي المفضلة؟ وهل هي نفس اللغة المعتمدة في الواجهة الخلفية لمشروع ألفا؟"
        }
    ]

    for turn in dialogue_script:
        print_separator(turn["title"])
        print(f"👤 المستخدم:\n   {turn['user']}")
        print("\n⏳ جاري المعالجة واسترجاع المعرفة وتحديث الذاكرة...")

        start_time = time.time()
        result = orchestrator.process_message(turn["user"])
        elapsed = time.time() - start_time

        print(f"\n🤖 المساعد (في {elapsed:.2f} ثانية):\n{result['response']}")

        # Show RAG sources if retrieved
        if result["rag_sources"]:
            print("\n📚 المستندات المسترجعة (RAG Evidence):")
            for src in result["rag_sources"]:
                print(f"   - [{src['source']}] (درجة التطابق: {src['score']}) -> {src['text'][:90]}...")

        # Show Memory updates applied
        if result["memory_updates"]:
            print("\n🧠 تحديثات الذاكرة الديناميكية التي طرأت في هذه الجولة:")
            for op in result["memory_updates"]:
                action = op.get("action")
                if action == "upsert":
                    print(f"   [تحديث/إضافة حقيقة] {op.get('key')} = '{op.get('value')}' (تصنيف: {op.get('category')})")
                elif action == "delete":
                    print(f"   [حذف حقيقة ملغاة] {op.get('key')}")
                elif action == "add_note":
                    print(f"   [تسجيل ملاحظة] {op.get('text')}")

        # Show current active dynamic profile
        print("\n📋 الحالة الحالية للذاكرة النشطة:")
        print(orchestrator.memory.get_memory_context())

    print_separator("اكتمال التجربة بنجاح")
    print(f"تم حفظ الذاكرة الديناميكية في الملف: {orchestrator.memory_path}")
    print(f"تم حفظ فهرس الـ RAG في الملف: {orchestrator.vector_path}")

if __name__ == "__main__":
    run_demo()
