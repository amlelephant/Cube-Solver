import cv2
import numpy as np
from color_classifier import classify_color, COLOR_BGR

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
FACE_NAMES = {
    "U": "Up (white center)",
    "R": "Right (red center)",
    "F": "Front (green center)",
    "D": "Down (yellow center)",
    "L": "Left (orange center)",
    "B": "Back (blue center)",
}
COLOR_TO_FACE = {
    "white":  "U",
    "red":    "R",
    "green":  "F",
    "yellow": "D",
    "orange": "L",
    "blue":   "B",
}

BOX_X, BOX_Y = 150, 100
BOX_SIZE = 300
CELL = BOX_SIZE // 3
SAMPLE_RADIUS = 10

def draw_grid(frame, sticker_labels=None):
    for row in range(3):
        for col in range(3):
            x = BOX_X + col * CELL
            y = BOX_Y + row * CELL
            cv2.rectangle(frame, (x, y), (x + CELL, y + CELL), (200, 200, 200), 1)

            cx = x + CELL // 2
            cy = y + CELL // 2

            if sticker_labels:
                idx = row * 3 + col
                color_name = sticker_labels[idx]
                bgr = COLOR_BGR[color_name]
                cv2.circle(frame, (cx, cy), 18, bgr, -1)
                cv2.circle(frame, (cx, cy), 18, (50, 50, 50), 1)
            else:
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

def read_face(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []

    for row in range(3):
        for col in range(3):
            cx = BOX_X + col * CELL + CELL // 2
            cy = BOX_Y + row * CELL + CELL // 2
            r = SAMPLE_RADIUS

            region = hsv[cy-r:cy+r, cx-r:cx+r]
            avg = region.mean(axis=(0, 1))
            h, s, v = int(avg[0]), int(avg[1]), int(avg[2])

            color = classify_color(h, s, v)
            results.append(color)

    return results

def face_colors_to_letters(color_list):
    letters = []
    for color in color_list:
        letter = COLOR_TO_FACE.get(color, "?")
        letters.append(letter)
    return letters


def main():
    cap = cv2.VideoCapture(0)
    last_reading = None

    print("Hold a cube face inside the grid.")
    print("Press SPACE to capture. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)


        cv2.putText(frame, "Press SPACE to scan face", (BOX_X, BOX_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            last_reading = read_face(frame)
            print("Scanned:", last_reading)

        draw_grid(frame, last_reading)
        cv2.imshow("Face Scanner", frame)
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()