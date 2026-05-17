import cv2
import insightface
from insightface.app import FaceAnalysis
import time

# Use detection only (skip recognition & landmarks = much faster)
app = FaceAnalysis(allowed_modules=['detection', 'genderage'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320))  # 320 instead of 640 = 4x faster

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened! Press Q to quit.")

frame_count = 0
faces = []
fps_start = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection every 2nd frame to boost FPS
    if frame_count % 2 == 0:
        faces = app.get(frame)

    # Draw results
    out = app.draw_on(frame, faces)

    # Show FPS on screen
    frame_count += 1
    if frame_count % 15 == 0:
        fps = 15 / (time.time() - fps_start)
        fps_start = time.time()

    cv2.putText(out, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(out, f'Faces: {len(faces)}', (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('InsightFace - Live Camera', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
