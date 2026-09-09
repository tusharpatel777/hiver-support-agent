import os
import json
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class LLMClient:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if self.gemini_key:
            self.provider = "gemini"
        elif self.openai_key:
            self.provider = "openai"
        elif self.anthropic_key:
            self.provider = "anthropic"
        else:
            self.provider = "local_semantic"

    def is_api_available(self) -> bool:
        return self.provider != "local_semantic"

    def _call_gemini_rest(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        """Calls Google Gemini API via lightweight HTTP REST."""
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000
            }
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
            else:
                # If model name 2.0-flash is not available, try gemini-1.5-flash
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
                resp2 = requests.post(fallback_url, headers=headers, json=payload, timeout=12)
                if resp2.status_code == 200:
                    data = resp2.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
        except Exception as e:
            pass
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Calls LLM to generate structured JSON output."""
        full_prompt = f"{system_prompt}\n\nUser Input: {user_prompt}\n\nRespond ONLY with valid JSON."
        
        if self.provider == "gemini":
            text = self._call_gemini_rest(full_prompt, json_mode=True)
            if text:
                try:
                    # Clean markdown code blocks if present
                    clean = text.strip()
                    if clean.startswith("```json"):
                        clean = clean[7:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    return json.loads(clean.strip())
                except Exception:
                    pass

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)
            except Exception:
                pass

        return None

    def generate_text(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Calls LLM to generate plain text."""
        full_prompt = f"{system_prompt}\n\nUser Input:\n{user_prompt}"
        
        if self.provider == "gemini":
            return self._call_gemini_rest(full_prompt, json_mode=False)

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass

        return None

llm_client = LLMClient()
