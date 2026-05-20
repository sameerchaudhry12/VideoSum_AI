import importlib, sys

packages = [
    ("cv2",            "OpenCV"),
    ("moviepy.editor", "MoviePy"),
    ("librosa",        "Librosa"),
    ("soundfile",      "SoundFile"),
    ("mtcnn",          "MTCNN"),
    ("mediapipe",      "MediaPipe"),
    ("tensorflow",     "TensorFlow"),
    ("torch",          "PyTorch"),
    ("transformers",   "HuggingFace Transformers"),
    ("datasets",       "HuggingFace Datasets"),
    ("whisper",        "OpenAI Whisper"),
    ("pandas",         "Pandas"),
    ("numpy",          "NumPy"),
    ("sklearn",        "Scikit-learn"),
    ("matplotlib",     "Matplotlib"),
    ("gradio",         "Gradio"),
]

all_good = True
for module, name in packages:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "ok")
        print(f"  OK  {name} ({version})")
    except ImportError as e:
        print(f"  FAIL  {name} — {e}")
        all_good = False

print()
if all_good:
    print("All imports OK. Environment is ready.")
else:
    print("Some imports failed. See above.")