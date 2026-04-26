import cv2
from deepface import DeepFace
import threading

print("Starting Multi Face Detector...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

print("Camera opened! Press Q to quit")

# Colors for emotions
colors = {
    "happy":    (0, 255, 0),
    "sad":      (255, 0, 0),
    "angry":    (0, 0, 255),
    "fear":     (128, 0, 128),
    "surprise": (0, 255, 255),
    "disgust":  (0, 128, 0),
    "neutral":  (255, 255, 255)
}

emojis = {
    "happy":    "HAPPY :)",
    "sad":      "SAD :(",
    "angry":    "ANGRY >:(",
    "fear":     "FEAR !!",
    "surprise": "SURPRISE :O",
    "disgust":  "DISGUST :/",
    "neutral":  "NEUTRAL :|"
}

# Shared data between threads
face_data = []
is_analyzing = False
frame_count = 0

def analyze_frame(frame):
    global face_data, is_analyzing
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        face_data = results
    except:
        face_data = []
    is_analyzing = False

# ================================
# MAIN LOOP
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Analyze in background thread every 15 frames
    if frame_count % 15 == 0 and not is_analyzing:
        is_analyzing = True
        t = threading.Thread(
            target=analyze_frame,
            args=(frame.copy(),)
        )
        t.daemon = True
        t.start()

    # ---- DRAW RESULTS FOR EACH FACE ----
    for i, face in enumerate(face_data):
        try:
            emotion = face['dominant_emotion']
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']

            color = colors.get(emotion, (255, 255, 255))
            label = emojis.get(emotion, emotion.upper())

            # Draw face box
            cv2.rectangle(frame,
                          (x, y), (x+w, y+h),
                          color, 3)

            # Face number
            cv2.putText(frame,
                        f"Face {i+1}",
                        (x, y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

            # Emotion label
            cv2.putText(frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            # Emotion score inside box
            all_emotions = face['emotion']
            top_score = all_emotions[emotion]
            cv2.putText(frame,
                        f"{top_score:.1f}%",
                        (x + 5, y + h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        except:
            pass

    # ---- TOP BAR ----
    cv2.rectangle(frame, (0, 0),
                  (frame.shape[1], 45), (0,0,0), -1)

    # Face count
    face_count = len(face_data)
    cv2.putText(frame,
                f"Faces Detected: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 200, 255), 2)

    # Quit instruction
    cv2.putText(frame,
                "Press Q to Quit",
                (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    # ---- BOTTOM BAR ----
    cv2.rectangle(frame,
                  (0, frame.shape[0] - 35),
                  (frame.shape[1], frame.shape[0]),
                  (0,0,0), -1)

    # Show all detected emotions summary
    summary = " | ".join([
        f"Face {i+1}: {f['dominant_emotion'].upper()}"
        for i, f in enumerate(face_data)
    ]) if face_data else "No faces detected"

    cv2.putText(frame, summary,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    cv2.imshow("Multi Face Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended!")
input("Press Enter to close...")