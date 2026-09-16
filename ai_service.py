"""
ai_service.py — OpenAI Whisper + GPT-4o-mini integration for Shafaco backend.

Provides:
  - transcribe_audio(): Converts an audio file to Arabic text via Whisper.
  - extract_report_data(): Parses the transcript into a structured JSON report
                           using GPT-4o-mini.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI client (reads OPENAI_API_KEY from environment automatically)
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt for structured data extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
أنت مساعد متخصص في استخراج البيانات من تقارير الصيانة الصوتية.
سيُعطَى لك نص عربي مُستخرَج من تسجيل صوتي لمهندس أو فني صيانة.
مهمتك هي استخراج المعلومات التالية وإرجاعها **حصراً** بصيغة JSON صحيحة دون أي نص إضافي.

المخطط المطلوب:
{
  "engineer_name": "اسم المهندس أو الفني إن ذُكر، وإلا أعد 'غير محدد'",
  "equipment":     "اسم الجهاز أو المعدة أو الغرفة التي تمت عليها الصيانة",
  "action_taken":  "وصف مختصر للإجراء الذي تم تنفيذه",
  "status":        "اختر واحدة فقط: ('طبيعي' / 'يحتاج متابعة' / 'عطل حرج')"
}

قواعد صارمة:
1. لا تضف أي مفاتيح إضافية خارج المخطط أعلاه.
2. قيمة "status" يجب أن تكون إحدى القيم الثلاث المحددة بالضبط.
3. لا تُرفق شرحاً أو نصاً قبل أو بعد JSON.
4. إذا كانت المعلومة غير موجودة في النص، استخدم سلسلة نصية فارغة "" باستثناء engineer_name الذي يعود بـ 'غير محدد'.
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def transcribe_audio(file_path: str) -> str:
    """
    Send an audio file to the OpenAI Whisper API and return the Arabic transcript.

    Args:
        file_path: Absolute or relative path to the saved audio file.

    Returns:
        The raw transcript text as a string.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        openai.OpenAIError: On API-level errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ar",          # Hint the model toward Arabic for better accuracy
            response_format="text",
        )

    # `response_format="text"` returns the transcript directly as a string
    return transcript.strip()


def extract_report_data(raw_text: str) -> dict:
    """
    Use GPT-4o-mini to extract structured maintenance-report data from an
    Arabic transcript.

    Args:
        raw_text: The Arabic transcript obtained from Whisper.

    Returns:
        A dictionary with keys: engineer_name, equipment, action_taken, status.

    Raises:
        ValueError: If the model response cannot be parsed as valid JSON.
        openai.OpenAIError: On API-level errors.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": raw_text},
        ],
        temperature=0,          # Deterministic output for structured extraction
        max_tokens=512,
        response_format={"type": "json_object"},   # Enforce JSON mode
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"GPT-4o-mini returned non-JSON content: {content}"
        ) from exc

    # Guarantee all expected keys exist with sensible fallbacks
    return {
        "engineer_name": data.get("engineer_name", "غير محدد") or "غير محدد",
        "equipment":     data.get("equipment", ""),
        "action_taken":  data.get("action_taken", ""),
        "status":        data.get("status", "طبيعي"),
    }
