import os
import sys
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import AssistantOrchestrator

def print_banner(model_name: str, total_chunks: int):
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════════╗
║              Mini RAG + Dynamic Memory Assistant             ║
║            مساعد الذكاء الاصطناعي مع ذاكرة حية و RAG           ║
╚══════════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)
    print(Fore.GREEN + f"✔ النموذج اللغوي: {model_name}")
    print(Fore.GREEN + f"✔ الوثائق المفهرسة: {total_chunks} جزءاً دلالياً")
    print(Fore.YELLOW + "الأوامر المتاحة: /memory (عرض الذاكرة), /docs (عرض المستندات), /clear (مسح الذاكرة), /help, /exit")
    print("-" * 64)

def interactive_cli():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    orchestrator = AssistantOrchestrator(data_dir=data_dir)
    total_chunks = orchestrator.index_documents_directory()

    print_banner(orchestrator.llm.model, total_chunks)

    while True:
        try:
            user_input = input(Fore.BLUE + Style.BRIGHT + "\nأنت: " + Style.RESET_ALL).strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "خروج"]:
                print(Fore.YELLOW + "وداعاً! تم حفظ جميع التحديثات بنجاح.")
                break

            elif user_input.lower() == "/memory":
                print(Fore.MAGENTA + Style.BRIGHT + "\n=== الحالة الحالية للذاكرة الديناميكية ===")
                print(orchestrator.memory.get_memory_context())
                continue

            elif user_input.lower() == "/docs":
                print(Fore.CYAN + Style.BRIGHT + "\n=== الوثائق المفهرسة في قاعدة المعرفة ===")
                files = os.listdir(orchestrator.docs_dir)
                for f in files:
                    print(f"  • {f}")
                print(f"إجمالي الأجزاء في الفهرس: {len(orchestrator.vector_store.chunks)}")
                continue

            elif user_input.lower() == "/reindex":
                print(Fore.YELLOW + "جاري إعادة فهرسة المستندات...")
                cnt = orchestrator.index_documents_directory()
                print(Fore.GREEN + f"✔ تمت إعادة الفهرسة بنجاح! ({cnt} أجزاء)")
                continue

            elif user_input.lower() == "/clear":
                orchestrator.memory.facts.clear()
                orchestrator.memory.notes.clear()
                orchestrator.memory.history.clear()
                orchestrator.memory.save_to_file(orchestrator.memory_path)
                print(Fore.RED + "✔ تم تصفير الذاكرة الديناميكية للمستخدم.")
                continue

            elif user_input.lower() == "/help":
                print(Fore.YELLOW + """
الأوامر المتاحة:
  /memory   - عرض الحقائق والملاحظات المسجلة عنك في الذاكرة الحية
  /docs     - استعراض ملفات قاعدة المعرفة المفهرسة في RAG
  /reindex  - إعادة فحص وفهرسة ملفات مجلد data/documents
  /clear    - تصفير الذاكرة والمحادثات
  /exit     - الخروج من البرنامج
                """)
                continue

            print(Fore.YELLOW + "⏳ جاري التفكير والاسترجاع..." + Style.RESET_ALL)
            result = orchestrator.process_message(user_input)

            # Display Response
            print(Fore.GREEN + Style.BRIGHT + "\nالمساعد:\n" + Style.RESET_ALL + result["response"])

            # Display RAG sources if any
            if result["rag_sources"]:
                print(Fore.CYAN + "\n[المصادر المسترجعة (RAG)]:")
                for s in result["rag_sources"]:
                    print(f"  ↳ {s['source']} (دقة: {s['score']:.2f})")

            # Display Memory Updates if any occurred
            if result["memory_updates"]:
                print(Fore.MAGENTA + "[تحديث الذاكرة الديناميكية]:")
                for op in result["memory_updates"]:
                    if op.get("action") == "upsert":
                        print(f"  ★ تم تعديل/تثبيت: {op.get('key')} = {op.get('value')}")
                    elif op.get("action") == "add_note":
                        print(f"  ★ ملاحظة جديدة: {op.get('text')}")

        except KeyboardInterrupt:
            print("\nتم الإنهاء.")
            break
        except Exception as e:
            print(Fore.RED + f"\nحدث خطأ: {e}")

if __name__ == "__main__":
    interactive_cli()
