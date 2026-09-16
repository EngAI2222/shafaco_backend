"""
ai_service.py — Google Generative AI (gemini-1.5-flash) integration for Shafaco backend.

Provides a single unified function:
  - process_audio_report(file_path): Uploads the audio file to the Gemini
    Files API, then asks gemini-1.5-flash to simultaneously transcribe the
    Arabic recording AND extract a structured maintenance report — all in
    one API call.
"""

import json
import mimetypes
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# SDK configuration
# ---------------------------------------------------------------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_ID = "gemini-1.5-flash"

# ---------------------------------------------------------------------------
# Task prompt (sent together with the audio)
# ---------------------------------------------------------------------------

AUDIO_PROMPT = """
أنت نظام متخصص في تحليل تسجيلات الصيانة الصوتية العربية.
ستُعطَى تسجيلاً صوتياً لمهندس أو فني صيانة يشرح ما قام به.

مهمتك:
1. استخرج النص الكامل من الصوت (raw_text).
2. من النص، استخرج الحقول المنظمة التالية.

أعد **حصراً** كائن JSON واحداً صحيحاً بالمفاتيح التالية — بدون أي نص قبله أو بعده:

{
  "raw_text":      "النص الكامل المُستخرَج من الصوت كما هو",
  "engineer_name": "اسم المهندس أو الفني إن ذُكر، وإلا 'غير محدد'",
  "equipment":     "اسم الجهاز أو المعدة أو الغرفة التي تمت عليها الصيانة",
  "action_taken":  "وصف مختصر للإجراء الذي تم تنفيذه",
  "status":        "اختر واحدة فقط بالضبط: 'طبيعي' أو 'يحتاج متابعة' أو 'عطل حرج'"
}

قواعد صارمة:
- لا تُضف مفاتيح إضافية.
- قيمة status يجب أن تكون إحدى القيم الثلاث المذكورة حرفياً.
- إذا لم تتوفر معلومة معينة (عدا engineer_name)، استخدم سلسلة فارغة "".
- لا تُرفق ماركداون أو أي تنسيق آخر — JSON فقط.
"""

# ---------------------------------------------------------------------------
# MIME type helpers
# ---------------------------------------------------------------------------

_AUDIO_MIME_MAP = {
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
    ".m4a":  "audio/mp4",
    ".aac":  "audio/aac",
    ".webm": "audio/webm",
    ".opus": "audio/ogg",
    ".amr":  "audio/amr",
}


def _get_mime_type(file_path: str) -> str:
    """Return a Gemini-compatible MIME type for the given audio file."""
    ext = Path(file_path).suffix.lower()
    if ext in _AUDIO_MIME_MAP:
        return _AUDIO_MIME_MAP[ext]
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "audio/mpeg"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def process_audio_report(file_path: str) -> dict:
    """
    Upload an audio file to Gemini Files API, then send it to
    gemini-1.5-flash to transcribe and extract a structured maintenance
    report in a single API call.

    Args:
        file_path: Path to the saved audio file on disk.

    Returns:
        Dict with keys: raw_text, engineer_name, equipment, action_taken, status.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the model response is not valid JSON.
        Exception: On any Gemini API error.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    mime_type = _get_mime_type(file_path)

    # ── Step 1: Upload the audio via the Files API ───────────────────────────
    # The Files API handles large audio files and avoids inline base64 limits.
    uploaded_file = genai.upload_file(path=file_path, mime_type=mime_type)

    # Wait until the file is fully processed (state == ACTIVE)
    max_wait_seconds = 60
    waited = 0
    while uploaded_file.state.name == "PROCESSING":
        if waited >= max_wait_seconds:
            raise TimeoutError(
                "Gemini Files API did not finish processing the audio in time."
            )
        time.sleep(2)
        waited += 2
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(
            f"Gemini Files API failed to process the audio: {uploaded_file.name}"
        )

    # ── Step 2: Generate the structured report ───────────────────────────────
    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",  # Force JSON output
            temperature=0,                           # Deterministic extraction
        ),
    )

    response = model.generate_content([uploaded_file, AUDIO_PROMPT])

    raw_content = response.text.strip()

    # Strip accidental markdown fences the model might still add
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        raw_content = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON content: {raw_content[:300]}"
        ) from exc

    # Normalise and guarantee all expected keys exist
    return {
        "raw_text":      data.get("raw_text", ""),
        "engineer_name": data.get("engineer_name", "غير محدد") or "غير محدد",
        "equipment":     data.get("equipment", ""),
        "action_taken":  data.get("action_taken", ""),
        "status":        data.get("status", "طبيعي"),
    }
