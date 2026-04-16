from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tracker import EyeTrackerSession
import cv2
import numpy as np
import base64
import logging
import os
import shutil
import tempfile
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from mimetypes import guess_type

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR = RESULTS_DIR / "videos"
ANNOTATED_DIR.mkdir(exist_ok=True)
RESULTS_INDEX_FILE = RESULTS_DIR / "index.json"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "CogniTracker API"}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as temp_file:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, temp_file)
        return Path(temp_file.name)


def _load_results_index():
    if not RESULTS_INDEX_FILE.exists():
        return []
    try:
        with open(RESULTS_INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_results_index(index_rows):
    with open(RESULTS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, indent=2)


def _persist_result_record(original_video_path: Path, annotated_video_path: Path, summary: dict):
    result_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()

    detail_record = {
        "id": result_id,
        "created_at": created_at,
        "video_file": original_video_path.name,
        "annotated_video_file": annotated_video_path.name,
        "summary": summary,
    }
    detail_path = RESULTS_DIR / f"{result_id}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail_record, f, indent=2)

    index_rows = _load_results_index()
    index_rows.append(
        {
            "id": result_id,
            "created_at": created_at,
            "video_file": original_video_path.name,
            "annotated_video_file": annotated_video_path.name,
            "attention_final": summary.get("attention_final"),
            "focus_pct": summary.get("focus_pct"),
            "duration_s": summary.get("duration_s"),
            "processed_frames": summary.get("processed_frames"),
        }
    )
    index_rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    _save_results_index(index_rows)
    return detail_record


def _draw_overlay(frame, frame_stats):
    h, w = frame.shape[:2]
    face_box = frame_stats.get("face_box")
    eye_dir = frame_stats.get("eye_dir", "Center")
    head_dir = frame_stats.get("head_dir", "Center")
    face_detected = frame_stats.get("face_detected", False)

    cv2.rectangle(frame, (int(0.04 * w), int(0.08 * h)), (int(0.96 * w), int(0.9 * h)), (230, 230, 230), 2)

    if face_detected and face_box:
        x = int(face_box.get("x", 0) * w)
        y = int(face_box.get("y", 0) * h)
        bw = int(face_box.get("w", 0) * w)
        bh = int(face_box.get("h", 0) * h)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 140, 40), 2)

        ex = int(x + 0.22 * bw)
        ey = int(y + 0.30 * bh)
        ew = int(0.56 * bw)
        eh = int(0.18 * bh)
        cv2.ellipse(frame, (ex + ew // 2, ey + eh // 2), (max(ew // 2, 1), max(eh // 2, 1)), 0, 0, 360, (0, 175, 255), 2)

    status_text = f"Eye: {eye_dir} | Head: {head_dir}"
    color = (0, 215, 255) if (eye_dir != "Center" or head_dir != "Center") else (120, 255, 120)
    cv2.rectangle(frame, (12, 12), (430, 46), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _summarize_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open uploaded video.")

    session = EyeTrackerSession()
    processed_frames = 0
    frame_trace = []
    last_trace_t = -1.0
    last_eye = "Center"
    last_head = "Center"
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    annotated_name = f"annotated_{uuid.uuid4().hex[:12]}.mp4"
    annotated_path = ANNOTATED_DIR / annotated_name
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise HTTPException(status_code=500, detail="Could not create annotated result video.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = (timestamp_ms / 1000.0) if timestamp_ms and timestamp_ms > 0 else (processed_frames / fps)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_stats = session.process_frame(rgb_frame, timestamp_s=timestamp_s)
            _draw_overlay(frame, frame_stats)
            writer.write(frame)

            eye_dir = frame_stats.get("eye_dir", "Center")
            head_dir = frame_stats.get("head_dir", "Center")
            should_trace = (
                (timestamp_s - last_trace_t) >= 0.2
                or eye_dir != last_eye
                or head_dir != last_head
            )
            if should_trace:
                frame_trace.append(
                    {
                        "t": round(timestamp_s, 2),
                        "eye_dir": eye_dir,
                        "head_dir": head_dir,
                        "face_detected": bool(frame_stats.get("face_detected", False)),
                        "face_box": frame_stats.get("face_box"),
                    }
                )
                last_trace_t = timestamp_s
                last_eye = eye_dir
                last_head = head_dir
            processed_frames += 1

        if processed_frames == 0:
            raise HTTPException(status_code=400, detail="The uploaded video does not contain readable frames.")

        duration_s = processed_frames / fps if fps > 0 else 0.0
        session_path = session.save_session(duration_s=duration_s)
        summary = session.build_summary(duration_s=duration_s)
        summary["processed_frames"] = processed_frames
        summary["source_video"] = video_path.name
        summary["session_file"] = os.path.basename(session_path)
        summary["frame_trace"] = frame_trace
        return summary, annotated_path
    finally:
        writer.release()
        capture.release()
        session.close()


@app.post("/upload/video-summary")
async def upload_video_summary(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")

    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported.")

    saved_path = _save_upload(file)
    logging.info("Processing uploaded video: %s", saved_path.name)

    try:
        summary, annotated_path = _summarize_video(saved_path)
        record = _persist_result_record(saved_path, annotated_path, summary)
        return {
            "status": "ok",
            "result_id": record["id"],
            "summary": summary,
            "video_url": f"/results/{record['id']}/video",
            "annotated_video_url": f"/results/{record['id']}/video/annotated",
        }
    finally:
        file.file.close()


@app.get("/results")
def list_results():
    return {"status": "ok", "results": _load_results_index()}


@app.get("/results/{result_id}")
def get_result(result_id: str):
    detail_path = RESULTS_DIR / f"{result_id}.json"
    if not detail_path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")

    with open(detail_path, "r", encoding="utf-8") as f:
        detail = json.load(f)
    detail["video_url"] = f"/results/{result_id}/video"
    detail["annotated_video_url"] = f"/results/{result_id}/video/annotated"
    return {"status": "ok", "result": detail}


@app.get("/results/{result_id}/video")
def get_result_video(result_id: str):
    detail_path = RESULTS_DIR / f"{result_id}.json"
    if not detail_path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")

    with open(detail_path, "r", encoding="utf-8") as f:
        detail = json.load(f)
    video_file = detail.get("video_file")
    if not video_file:
        raise HTTPException(status_code=404, detail="Video not found.")

    video_path = UPLOAD_DIR / video_file
    if not video_path.exists():
        annotated_video_file = detail.get("annotated_video_file")
        if annotated_video_file:
            annotated_path = ANNOTATED_DIR / annotated_video_file
            if annotated_path.exists():
                return FileResponse(path=str(annotated_path), media_type="video/mp4", filename=annotated_video_file)
        raise HTTPException(status_code=404, detail="Video file missing on server.")
    media_type = guess_type(str(video_path))[0] or "application/octet-stream"
    return FileResponse(path=str(video_path), media_type=media_type, filename=video_file)


@app.get("/results/{result_id}/video/annotated")
def get_result_annotated_video(result_id: str):
    detail_path = RESULTS_DIR / f"{result_id}.json"
    if not detail_path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")

    with open(detail_path, "r", encoding="utf-8") as f:
        detail = json.load(f)
    annotated_video_file = detail.get("annotated_video_file")
    if not annotated_video_file:
        raise HTTPException(status_code=404, detail="Annotated video not found.")

    video_path = ANNOTATED_DIR / annotated_video_file
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video file missing on server.")
    return FileResponse(path=str(video_path), media_type="video/mp4", filename=annotated_video_file)


@app.websocket("/ws/live")
async def live_session(websocket: WebSocket):
    await websocket.accept()
    session = EyeTrackerSession()
    logging.info("New live session started")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Strip data-URL prefix if present
                if "," in data:
                    data = data.split(",")[1]

                img_bytes = base64.b64decode(data)
                np_arr    = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stats = session.process_frame(rgb)
                    await websocket.send_json(stats)

            except Exception as e:
                logging.error(f"Frame error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logging.info("Client disconnected — saving session...")

    finally:
        # Auto-save when connection closes (client ends session)
        try:
            path = session.save_session()
            logging.info(f"Session saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save session: {e}")
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
