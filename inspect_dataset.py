
from pathlib import Path
import random
import cv2
import numpy as np

root = Path("data/fake_dataset")
random.seed(42)

groups = [
    ("REAL", root / "real"),
    ("FAKE", root / "fake"),
]

thumb_w, thumb_h = 240, 200
cols, rows = 4, 3
gap = 12
panel_w = cols * thumb_w + (cols + 1) * gap
panel_h = rows * (thumb_h + 30) + (rows + 1) * gap

panels = []

for label, folder in groups:
    files = [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    samples = random.sample(files, min(12, len(files)))

    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
    cv2.putText(
        panel, f"{label} - {len(samples)} samples",
        (gap, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (20, 20, 20), 2
    )

    for i, path in enumerate(samples):
        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.resize(img, (thumb_w, thumb_h))
        x = gap + (i % cols) * (thumb_w + gap)
        y = 45 + (i // cols) * (thumb_h + 30)

        panel[y:y + thumb_h, x:x + thumb_w] = img
        cv2.putText(
            panel, path.name[:28],
            (x, y + thumb_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (20, 20, 20), 1
        )

    panels.append(panel)

preview = np.vstack(panels)
output = Path("dataset_preview.jpg")
cv2.imwrite(str(output), preview)
print(f"Preview saved to: {output.resolve()}")