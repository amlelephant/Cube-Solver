# HSV ranges: each color is defined by (H_min, H_max, S_min, V_min)
# OpenCV HSV: H is 0-179, S is 0-255, V is 0-255

COLOR_RANGES = {
    
    "yellow": {"h": (26, 35),  "s": (100, 255), "v": (100, 255)},
    "red":    {"h": (0, 10),   "s": (100, 255), "v": (100, 255)},
    "orange": {"h": (11, 25),  "s": (100, 255), "v": (100, 255)},
    "blue":   {"h": (86, 130), "s": (110, 255),  "v": (100, 255)},
    "green":  {"h": (36, 85),  "s": (50, 255),  "v": (50, 255)},
    "white":  {"h": (0, 179),  "s": (0, 40),   "v": (150, 255)},
}

COLOR_BGR = {
    "white":   (255, 255, 255),
    "yellow":  (0, 255, 255),
    "red":     (0, 0, 200),
    "orange":  (0, 128, 255),
    "blue":    (200, 50, 0),
    "green":   (0, 180, 0),
    "unknown": (128, 128, 128),
}

def classify_color(h, s, v):
    for color_name, ranges in COLOR_RANGES.items():
        h_min, h_max = ranges["h"]
        s_min = ranges["s"][0]
        v_min = ranges["v"][0]

        if h_min <= h <= h_max and s >= s_min and v >= v_min:
            return color_name

    return "unknown"

if __name__ == "__main__":
    test_cases = [
        (5, 200, 180, "red"),
        (25, 0, 240, "white"),
        (110, 150, 200, "blue"),
        (60, 180, 180, "green"),
        (28, 180, 200, "yellow"),
        (15, 160, 180, "orange"),
    ]

    for h, s, v, expected in test_cases:
        result = classify_color(h, s, v)
        status = "✓" if result == expected else "✗"
        print(f"{status} H={h} S={s} V={v} → got '{result}', expected '{expected}'")