import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read from webcam")
        break
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    region = frame[cy-10:cy+10, cx-10:cx+10]
    avg_color = region.mean(axis=(0, 1))
    print(f"Avg BGR in center region: {avg_color.astype(int)}")
    cv2.imshow("Webcam Feed", frame)

    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()