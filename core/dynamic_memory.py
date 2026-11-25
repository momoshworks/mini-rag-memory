import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

class FactEntry:
    def __init__(self, key: str, value: str, category: str = "general", updated_at: Optional[str] = None):
        self.key = key
        self.value = value
        self.category = category
        self.updated_at = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            category=data.get("category", "general"),
            updated_at=data.get("updated_at")
        )

class DynamicMemory:
    """
    Manages dynamic, evolving user memory:
    1. Long-term key-value facts (identity, location, preferences) that update and resolve contradictions.
    2. Contextual episodic notes.
    3. Short-term dialogue buffer (sliding window).
    """
    def __init__(self, max_history_turns: int = 6):
        self.max_history_turns = max_history_turns
        self.facts: Dict[str, FactEntry] = {}
        self.notes: List[str] = []
        self.history: List[Dict[str, str]] = []

    def upsert_fact(self, key: str, value: str, category: str = "general"):
        """Inserts or dynamically updates a fact, overwriting obsolete info."""
        clean_key = key.strip().lower()
        self.facts[clean_key] = FactEntry(key=clean_key, value=value.strip(), category=category)

    def delete_fact(self, key: str):
        clean_key = key.strip().lower()
        if clean_key in self.facts:
            del self.facts[clean_key]

    def add_note(self, note: str):
        note_clean = note.strip()
        if note_clean and note_clean not in self.notes:
            self.notes.append(note_clean)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

    def get_memory_context(self) -> str:
        """Formats the active dynamic memory into a structured prompt block."""
        lines = []
        if self.facts:
            lines.append("📌 حقائق المستخدم الحالية والمحدثة (Current Active Facts):")
            for fact in self.facts.values():
                lines.append(f"  • {fact.key}: {fact.value} (تصنيف: {fact.category})")

        if self.notes:
            lines.append("\n📝 ملاحظات سياقية إضافية:")
            for note in self.notes:
                lines.append(f"  • {note}")

        if not lines:
            return "لا توجد حقائق مسجلة عن المستخدم حتى الآن."

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": {k: f.to_dict() for k, f in self.facts.items()},
            "notes": self.notes,
            "history": self.history
        }

    def save_to_file(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.facts = {k: FactEntry.from_dict(v) for k, v in data.get("facts", {}).items()}
        self.notes = data.get("notes", [])
        self.history = data.get("history", [])


class MemoryExtractor:
    """
    Intelligent extraction layer that inspects dialogue turns and outputs
    structured atomic memory operations (upsert, delete, add_note).
    """
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_and_update(self, user_message: str, assistant_response: str, memory: DynamicMemory) -> List[Dict[str, Any]]:
        current_memory_view = memory.get_memory_context()

        system_prompt = (
            "You are a strict Memory Extraction Engine for an AI Assistant.\n"
            "Analyze the USER'S message only. Extract permanent facts about the user:\n"
            "- Name, job title, residence/location, city\n"
            "- Tools, favorite languages, tech stack, preferences\n"
            "- Life/work updates (e.g., moved to another city, switched tools)\n\n"
            "RULES:\n"
            "1. ONLY extract information EXPLICITLY stated by the user in this turn.\n"
            "2. Never hallucinate or copy placeholder values.\n"
            "3. If user updates an existing fact (e.g. new location or new language), use 'upsert' with the same key to overwrite.\n"
            "4. Return strictly JSON.\n\n"
            f"Current Memory:\n{current_memory_view}\n\n"
            "Format:\n"
            "{\n"
            '  "operations": [\n'
            '    {"action": "upsert", "key": "<field_name>", "value": "<value>", "category": "identity|preference|location|project"},\n'
            '    {"action": "add_note", "text": "<factual summary of user status>"}\n'
            "  ]\n"
            "}\n"
            "If no personal facts exist in the user message, return: {\"operations\": []}"
        )

        user_prompt = f"User said: {user_message}"

        result = self.llm.generate_json(user_prompt, system=system_prompt)
        operations = result.get("operations", [])
        applied_ops = []

        if isinstance(operations, list):
            for op in operations:
                if not isinstance(op, dict):
                    continue
                action = op.get("action")
                key = str(op.get("key", "")).strip().lower()
                value = str(op.get("value", "")).strip()

                # Guard against placeholder values
                if "<" in key or "<" in value:
                    continue

                if action == "upsert" and key and value:
                    memory.upsert_fact(key, value, op.get("category", "general"))
                    applied_ops.append(op)
                elif action == "delete" and key:
                    memory.delete_fact(key)
                    applied_ops.append(op)
                elif action == "add_note" and op.get("text"):
                    note_text = str(op["text"]).strip()
                    if "<" not in note_text and note_text:
                        memory.add_note(note_text)
                        applied_ops.append(op)

        return applied_ops
