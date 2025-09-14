#!/usr/bin/env python3
import os, re, uuid, tempfile, threading, time
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- IMPORT YOUR EXISTING PIPELINE CODE (with small tweaks below) ---
# Move your whole script into a module called pipeline.py:
# (see 'pipeline.py' further down)
from pipeline import run_pipeline_from_youtube, set_target_language_hint

app = FastAPI(title="Lyra-TF-Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    youtube_url: str
    target_lang: Optional[str] = "en"
    vocals_vol: float = 0.7
    instr_vol: float = 0.5

JOBS: Dict[str, Dict] = {}   # naive in-memory job store

def _worker(job_id: str, req: TranslateRequest):
    try:
        JOBS[job_id]["status"] = "running"
        set_target_language_hint(req.target_lang or "en")
        out_dir = os.path.join(tempfile.gettempdir(), f"lyra_job_{job_id}")
        os.makedirs(out_dir, exist_ok=True)
        final_out = run_pipeline_from_youtube(
            youtube_url=req.youtube_url,
            vocals_vol=req.vocals_vol,
            instr_vol=req.instr_vol,
            out_dir=out_dir,
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result_path"] = final_out
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

@app.post("/api/translate")
def start_translate(req: TranslateRequest):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "result_path": None, "error": None}
    t = threading.Thread(target=_worker, args=(job_id, req), daemon=True)
    t.start()
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.get("/api/file/{job_id}")
def get_file(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("result_path"):
        raise HTTPException(status_code=404, detail="result not available")
    return FileResponse(job["result_path"], filename="final_translated_song.wav", media_type="audio/wav")
