"""Phase 0 smoke test: checks OmniRoute combos return valid judge JSON.

Setup (Git Bash):
    pip install openai pydantic pillow
    export OMNIROUTE_KEY="<key from Dashboard -> Endpoints>"

Run:
    python smoke_test.py                       # text combo only
    python smoke_test.py "/c/path/frame.jpg"   # also tests the vision combo with a game screenshot

Bypass a combo and hit one model directly:
    VISION_MODEL=gemini/gemini-3.8-flash python smoke_test.py "/c/path/frame.jpg"
"""
import base64
import json
import os
import re
import sys
import time

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

BASE_URL = os.getenv("OMNIROUTE_URL", "http://localhost:20128/v1")
TEXT_MODEL = os.getenv("TEXT_MODEL", "clipper-text")
VISION_MODEL = os.getenv("VISION_MODEL", "clipper-vision")

client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("OMNIROUTE_KEY", "none"))


class Judgement(BaseModel):
    funny_score: int = Field(ge=0, le=10)
    category: str
    title: str
    hook_line: str
    start: float
    end: float
    reason: str


SYSTEM = (
    "You judge livestream moments for short-form highlight clips. "
    "Reply with ONLY a JSON object with keys: funny_score (int 0-10), category, "
    "title (max 60 chars), hook_line, start (seconds), end (seconds), reason."
)

SAMPLE = """Game: Rocket League. Window 0-60s.
Transcript:
[12.0] okay I got this, I got this, easy aerial
[15.5] ...
[16.2] HOW DID I MISS THAT. THE BALL WAS RIGHT THERE
[19.0] I'm uninstalling. I'm actually uninstalling.
[24.0] chat don't clip that
Chat (16-25s): KEKW x14, OMEGALUL x9, "CLIP IT" x6, "whiff of the year" x3"""


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # strip reasoning blocks
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


def load_image_b64(path: str, max_side: int = 1024) -> str:
    """Downscale to max_side px and re-encode as JPEG so the payload stays small (~100-200 KB)."""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        data = buf.getvalue()
        print(f"[image] resized to {img.size[0]}x{img.size[1]}, {len(data)//1024} KB")
    except ImportError:
        data = open(path, "rb").read()
        print(f"[image] Pillow not installed, sending original ({len(data)//1024} KB) - pip install pillow")
    return base64.b64encode(data).decode()


def run(model: str, content) -> bool:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            temperature=0.3,
        )
    except Exception as e:  # network / auth / no route
        print(f"[{model}] REQUEST FAILED: {e}")
        return False
    raw = resp.choices[0].message.content or ""
    served_by = getattr(resp, "model", "?")
    try:
        j = Judgement(**extract_json(raw))
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        print(f"[{model}] served by {served_by} in {time.time()-t0:.1f}s -> INVALID JSON: {e}\n{raw[:500]}")
        return False
    print(f"[{model}] served by {served_by} in {time.time()-t0:.1f}s -> OK")
    print(json.dumps(j.model_dump(), indent=2))
    return True


if __name__ == "__main__":
    ok = run(TEXT_MODEL, SAMPLE)
    if len(sys.argv) > 1:
        b64 = load_image_b64(sys.argv[1])
        ok &= run(VISION_MODEL, [
            {"type": "text", "text": "Game screenshot from a stream, window 0-60s. First describe what you see in the "
             "'reason' field (so we can confirm the image arrived), then judge if it's a funny/visual-gag moment."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])
    sys.exit(0 if ok else 1)
