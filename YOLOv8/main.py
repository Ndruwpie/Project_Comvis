
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Buka webcam
cap = cv2.VideoCapture(0)

# Aturan: key mengalahkan value
wins_against = {
    "rock": "scissors",
    "scissors": "paper",
    "paper": "rock"
}

print("Kamera menyala! Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Balik gambar secara horizontal agar tidak mirror
    frame = cv2.flip(frame, 1)

    # Jalankan object detection
    results = model(frame, conf=0.5, verbose=False)
    result = results[0]

    hands = []

    # Ambil hasil deteksi tangan
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower().strip()
        confidence = float(box.conf[0])

        # Ambil koordinat bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if class_name in wins_against:
            hands.append({
                "pose": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "status": ""
            })

    # Urutkan tangan dari kiri ke kanan
    hands.sort(key=lambda hand: hand["box"][0])

    # Kumpulan variasi pose yang terdeteksi
    poses = set(hand["pose"] for hand in hands)

    winner_pose = None

    # LOGIKA MENENTUKAN PEMENANG
    if len(hands) >= 2:

        # Semua pose sama atau ketiga pose muncul
        if len(poses) == 1 or len(poses) == 3:
            for hand in hands:
                hand["status"] = "DRAW"

        # Jika hanya ada dua variasi pose
        elif len(poses) == 2:
            pose1, pose2 = list(poses)

            if wins_against[pose1] == pose2:
                winner_pose = pose1
            else:
                winner_pose = pose2

            for hand in hands:
                if hand["pose"] == winner_pose:
                    hand["status"] = "WINNER"
                else:
                    hand["status"] = "LOSER"

    # TAMPILKAN BOUNDING BOX
    for i, hand in enumerate(hands):
        x1, y1, x2, y2 = hand["box"]
        pose = hand["pose"].upper()
        status = hand["status"]
        confidence = hand["confidence"]

        # Warna berdasarkan hasil
        if status == "WINNER":
            color = (0, 255, 0)       # Hijau
        elif status == "LOSER":
            color = (0, 0, 255)       # Merah
        elif status == "DRAW":
            color = (0, 255, 255)     # Kuning
        else:
            color = (255, 255, 0)     # Cyan

        # Bounding box tangan
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Teks label
        label = f"Hand {i+1}: {pose} {confidence:.2f}"

        if status:
            label += f" - {status}"

        # Posisi teks
        text_y = max(65, y1 - 10)

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # TAMPILKAN HASIL PERMAINAN
    if len(hands) < 2:
        game_result = "Waiting for at least 2 hands..."

    elif len(poses) == 1 or len(poses) == 3:
        game_result = "DRAW"

    else:
        game_result = f"{winner_pose.upper()} WINS!"

    # Background teks hasil
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 55),
        (30, 30, 30),
        -1
    )

    # Tampilkan teks hasil
    cv2.putText(
        frame,
        game_result,
        (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    # Tampilkan kamera
    cv2.imshow("Real-time Rock Paper Scissors", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Matikan kamera
cap.release()
cv2.destroyAllWindows()
