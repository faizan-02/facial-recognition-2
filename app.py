import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
INSIGHTFACE_PKG = ROOT / "insightface" / "python-package"
sys.path.insert(0, str(INSIGHTFACE_PKG))

from insightface.app import FaceAnalysis

UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="InsightFace Demo",
    description="CPU-only face detection and comparison demo",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

face_app: FaceAnalysis | None = None


@app.on_event("startup")
def startup():
    global face_app
    face_app = FaceAnalysis(
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=-1, det_size=(320, 320))


def _read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return img


def _annotate(image: np.ndarray, faces: list) -> np.ndarray:
    annotated = image.copy()
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        score = float(face.det_score) if hasattr(face, "det_score") else 0.0
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 190, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 190, 0),
            2,
        )
    return annotated


def _save_image(image: np.ndarray) -> str:
    name = f"result_{uuid.uuid4().hex[:10]}.jpg"
    out_path = UPLOADS_DIR / name
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name


def _similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / denom)


@app.get("/")
def index():
    return FileResponse(str(ROOT / "static" / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/api/detect")
async def detect_faces(file: UploadFile = File(...)):
    if face_app is None:
        raise HTTPException(500, "Model not initialized")

    contents = await file.read()
    image = _read_image(contents)
    faces = face_app.get(image)

    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        results.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(face.det_score),
        })

    annotated = _annotate(image, faces)
    filename = _save_image(annotated)

    return {
        "faces": results,
        "total_faces": len(results),
        "annotated_image": f"/api/files/{filename}",
    }


@app.post("/api/compare")
async def compare_faces(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    if face_app is None:
        raise HTTPException(500, "Model not initialized")

    img_a = _read_image(await image_a.read())
    img_b = _read_image(await image_b.read())

    faces_a = face_app.get(img_a)
    faces_b = face_app.get(img_b)

    if not faces_a:
        raise HTTPException(400, "No face found in image A")
    if not faces_b:
        raise HTTPException(400, "No face found in image B")

    sim = _similarity(faces_a[0].embedding, faces_b[0].embedding)
    if sim < 0.2:
        verdict = "Not the same person"
    elif sim < 0.28:
        verdict = "Likely the same person"
    else:
        verdict = "Same person"

    return {
        "similarity": round(sim, 4),
        "verdict": verdict,
    }


@app.get("/api/files/{filename}")
def get_file(filename: str):
    safe_name = Path(filename).name
    filepath = UPLOADS_DIR / safe_name
    if not filepath.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(filepath), media_type="image/jpeg")
