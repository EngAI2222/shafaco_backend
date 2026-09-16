"""
main.py — FastAPI application entry-point for the Shafaco backend.

Routes:
  POST /api/upload-audio  — Upload audio, transcribe, extract report, persist to DB.
  GET  /api/reports       — Return all reports (newest first) as JSON.
  GET  /dashboard         — Render the HTML dashboard.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ai_service import process_audio_report
from database import Report, get_db, init_db

load_dotenv()

# ---------------------------------------------------------------------------
# Directory setup (created before the app mounts anything)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "static" / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Shafaco Backend",
    description="Audio-based maintenance report processor powered by OpenAI.",
    version="1.0.0",
)

# --- CORS (allow all origins so the Flutter app can call this API) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files -----------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# --- Jinja2 templates -------------------------------------------------------
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Startup event: initialise the database
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    """Create database tables if they do not exist yet."""
    init_db()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload-audio", summary="Upload audio and generate a maintenance report")
async def upload_audio(
    audio_file: UploadFile,
    engineer_name: str = Form(default="غير محدد"),
    db: Session = Depends(get_db),
):
    """
    1. Save the uploaded audio file to ``static/uploads/`` with a unique name.
    2. Transcribe it via Whisper.
    3. Extract structured report data via GPT-4o-mini.
    4. Persist the report to SQLite.
    5. Return a success response with the saved report.
    """
    # ---- Validate the uploaded file ----------------------------------------
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # ---- Persist the audio file with a UUID-based name ---------------------
    file_extension = Path(audio_file.filename).suffix or ".m4a"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    save_path = UPLOADS_DIR / unique_filename

    try:
        contents = await audio_file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to save audio file: {exc}"
        ) from exc

    # Relative URL path to serve the file via the /static mount
    audio_url = f"/static/uploads/{unique_filename}"

    # ---- AI processing (single Gemini call: transcription + extraction) ----
    try:
        report_data = process_audio_report(str(save_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502, detail=f"Gemini returned invalid JSON: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Gemini processing failed: {exc}"
        ) from exc

    # ---- Override engineer_name if provided explicitly in the form ----------
    final_engineer_name = report_data.get("engineer_name", "غير محدد")
    if engineer_name and engineer_name.strip() and engineer_name.strip() != "غير محدد":
        final_engineer_name = engineer_name.strip()

    # ---- Persist to database -----------------------------------------------
    try:
        new_report = Report(
            engineer_name=final_engineer_name,
            raw_text=report_data.get("raw_text", ""),
            equipment=report_data.get("equipment", ""),
            action_taken=report_data.get("action_taken", ""),
            status=report_data.get("status", "طبيعي"),
            audio_path=audio_url,
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Database error: {exc}"
        ) from exc

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "تم معالجة التقرير وحفظه بنجاح.",
            "report": new_report.to_dict(),
        },
    )


@app.get("/api/reports", summary="Retrieve all maintenance reports")
def get_reports(db: Session = Depends(get_db)):
    """
    Return all reports from the database ordered by creation date (newest first).
    """
    try:
        reports = (
            db.query(Report)
            .order_by(Report.created_at.desc())
            .all()
        )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "count": len(reports),
                "reports": [r.to_dict() for r in reports],
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch reports: {exc}"
        ) from exc


@app.get("/dashboard", summary="Maintenance reports dashboard")
def dashboard(request: Request):
    """Render the HTML dashboard template."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ---------------------------------------------------------------------------
# Local development entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
