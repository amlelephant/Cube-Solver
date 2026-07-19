"""Quick image review tool for labeling sessions.

Scroll through a folder of images with Left/Right arrows, press D to pull a
bad image (and its matching YOLO label .txt, if present) out of the set.
Nothing is permanently deleted -- both get moved into a `deleted/` subfolder
next to the images so a mis-press is recoverable.

Usage (run from inside cv/, per this repo's convention):
    python label_review.py face_dataset/train/images
    python label_review.py cube_dataset/train/images
"""

import shutil
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_DISPLAY = 900


class ReviewApp:
    def __init__(self, images_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = images_dir.parent / "labels"
        self.deleted_images_dir = images_dir / "deleted"
        self.deleted_labels_dir = self.labels_dir / "deleted"

        self.files = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.files:
            print(f"No images found in {images_dir}")
            sys.exit(1)

        self.index = 0

        self.root = tk.Tk()
        self.root.title("Label Review")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.Label(self.root, font=("Segoe UI", 10))
        self.status.pack()

        self.root.bind("<Left>", lambda e: self.show(self.index - 1))
        self.root.bind("<Right>", lambda e: self.show(self.index + 1))
        self.root.bind("d", lambda e: self.delete_current())
        self.root.bind("D", lambda e: self.delete_current())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.show(0)
        self.root.mainloop()

    def show(self, index):
        if not self.files:
            self.status.config(text="No images left.")
            self.label.config(image="")
            return
        self.index = index % len(self.files)
        path = self.files[self.index]

        img = Image.open(path)
        img.thumbnail((MAX_DISPLAY, MAX_DISPLAY))
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)
        self.status.config(
            text=f"{path.name}   [{self.index + 1}/{len(self.files)}]"
            "   ←/→ scroll   D delete   Q quit"
        )

    def delete_current(self):
        if not self.files:
            return
        path = self.files[self.index]

        self.deleted_images_dir.mkdir(exist_ok=True)
        shutil.move(str(path), str(self.deleted_images_dir / path.name))

        label_path = self.labels_dir / (path.stem + ".txt")
        if label_path.exists():
            self.deleted_labels_dir.mkdir(exist_ok=True)
            shutil.move(str(label_path), str(self.deleted_labels_dir / label_path.name))

        del self.files[self.index]
        self.show(self.index)


def main():
    if len(sys.argv) != 2:
        print("Usage: python label_review.py <path_to_images_folder>")
        sys.exit(1)

    images_dir = Path(sys.argv[1])
    if not images_dir.is_dir():
        print(f"Not a directory: {images_dir}")
        sys.exit(1)

    ReviewApp(images_dir)


if __name__ == "__main__":
    main()
