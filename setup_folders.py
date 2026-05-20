import os

ROOT = "."  # current directory = emotion_video_summarizer/

folders = [
    "data/fer2013",
    "data/ravdess",
    "data/meld",
    "videos/input",
    "videos/output",
    "preprocessed/frames",
    "preprocessed/faces",
    "preprocessed/audio",
    "preprocessed/transcripts",
    "models/cnn",
    "models/lstm",
    "models/bert",
    "fusion",
    "highlights",
    "reports",
]

for folder in folders:
    path = os.path.join(ROOT, folder)
    os.makedirs(path, exist_ok=True)
    print(f"  Created: {path}")

print("\nFolder structure ready.")