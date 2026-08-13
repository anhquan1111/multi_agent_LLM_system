"""Kiểm tra GOOGLE_API_KEY và in ra danh sách model có thật.

Chạy:  uv run python scripts/check_key.py

Dùng để đối chiếu tên model trong .env với những gì API thật sự cung cấp,
thay vì tin vào tên model chép từ blog hay tài liệu cũ.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

# Console Windows mặc định là cp1252, không in được emoji/tiếng Việt có dấu.
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "").strip()
if not api_key:
    sys.exit("❌ Chưa có GOOGLE_API_KEY trong .env — mở file .env và dán key vào.")
if api_key.startswith(("'", '"')):
    sys.exit("❌ Key đang bị bọc trong dấu nháy. Bỏ dấu nháy đi, chỉ để GOOGLE_API_KEY=AIza...")

client = genai.Client(api_key=api_key)

chat_models: list[str] = []
embed_models: list[str] = []
for m in client.models.list():
    actions = set(getattr(m, "supported_actions", None) or [])
    if "embedContent" in actions:
        embed_models.append(m.name)
    if "generateContent" in actions:
        chat_models.append(m.name)

print(f"✅ Key hợp lệ. Tổng {len(chat_models) + len(embed_models)} model dùng được.\n")
print("--- Model sinh text (dùng cho LLM_MODEL) ---")
for name in sorted(chat_models):
    print(" ", name)
print("\n--- Model embedding (dùng cho EMBEDDING_MODEL) ---")
for name in sorted(embed_models):
    print(" ", name)

print("\nĐối chiếu với .env hiện tại:")
for var in ("LLM_MODEL", "EMBEDDING_MODEL"):
    value = os.getenv(var, "")
    pool = chat_models if var == "LLM_MODEL" else embed_models
    ok = any(name.endswith(value) for name in pool) if value else False
    verdict = "✅ có thật" if ok else "⚠️ KHÔNG thấy trong danh sách trên"
    print(f"  {var}={value or '(trống)'}  ->  {verdict}")
