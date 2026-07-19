import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Standard grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Stronger blur to melt the inner sticker details into the frame lines
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Adaptive thresholding to extract fine structural lines
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 4. AGGRESSIVE MORPHOLOGY (The Key for Angles)
    # We use a larger kernel and iterate closing to force the 9 stickers 
    # to merge into one single, massive solid shape.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Find contours on the merged master shapes
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8000:  # Must be large enough to be the full face near the camera
            continue

        # Approximate the polygon. For a tilted cube, we use a slightly 
        # higher epsilon (0.05 - 0.06) to flatten out edge noise caused by angles.
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)

        # A cube face under perspective warp will always have exactly 4 major corners
        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                
                # Check solidity to ensure it's a solid block (not a weird background artifact)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                if solidity > 0.85:
                    # DRAW THE DETECTED FACE BOUNDARY
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    
                    # Optional: Print corner points for debugging perspective warp
                    for pt in approx:
                        cv2.circle(frame, tuple(pt[0]), 5, (0, 0, 255), -1)

    # Output windows
    cv2.imshow("Original Frame (Face Detection)", frame)
    cv2.imshow("Melted Structure (Debug)", closed)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()