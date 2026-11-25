import os
import json
import re
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

class LLMClient:
    """
    Lightweight, resilient LLM client.
    Supports local Ollama (default: qwen2.5:1.5b or qwen2.5:3b)
    with optional fallback to Google GenAI API if GEMINI_API_KEY is configured.
    """
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:1.5b",
        ollama_url: str = "http://127.0.0.1:11434"
    ):
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.api_key = os.environ.get("GEMINI_API_KEY", "")

        # Auto-detect available models if Ollama is chosen
        if self.provider == "ollama":
            self._detect_ollama_model()

    def _detect_ollama_model(self):
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", headers={"User-Agent": "MiniRAG/1.0"})
            with urllib.request.urlopen(req, timeout=3) as res:
                data = json.loads(res.read().decode("utf-8"))
                available = [m["name"].split(":")[0] for m in data.get("models", [])]
                full_names = [m["name"] for m in data.get("models", [])]
                
                # Check if current model exists
                if not any(self.model in name for name in full_names):
                    # Pick best available local model
                    for preferred in ["qwen2.5:1.5b", "qwen2.5:3b", "momo1:latest", "llama3.2:3b"]:
                        if any(preferred in name for name in full_names):
                            self.model = preferred
                            break
        except Exception:
            pass

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generates standard text response from the configured LLM."""
        if self.provider == "ollama":
            return self._call_ollama(prompt, system, temperature)
        elif self.provider == "gemini":
            return self._call_gemini(prompt, system, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "MiniRAG/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"[Error connecting to Ollama at {self.ollama_url}: {e}]"
        except Exception as e:
            return f"[Error generating response: {e}]"

    def _call_gemini(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            return response.text or ""
        except Exception as e:
            return f"[Error calling Gemini: {e}]"

    def generate_json(self, prompt: str, system: str = "", temperature: float = 0.1) -> Dict[str, Any]:
        """Ensures extracted output is strictly valid JSON."""
        system_with_json_instruction = (
            (system + "\n" if system else "") +
            "CRITICAL: You must reply ONLY with valid JSON. Do not write explanations or markdown outside the JSON."
        )
        raw_text = self.generate(prompt, system=system_with_json_instruction, temperature=temperature)
        
        # 1. Direct parse attempt
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # 2. Extract block from ```json ... ``` or { ... }
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # 3. Find first { and last }
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start:end+1])
            except Exception:
                pass

        return {"raw_text": raw_text, "error": "Failed to parse structured JSON"}
