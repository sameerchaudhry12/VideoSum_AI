# config.py — Global config. Import this in every script.
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
FER2013_DIR      = os.path.join(ROOT, "data", "fer2013")
RAVDESS_DIR      = os.path.join(ROOT, "data", "ravdess")
MELD_DIR         = os.path.join(ROOT, "data", "meld")

# Preprocessed
FRAMES_DIR       = os.path.join(ROOT, "preprocessed", "frames")
FACES_DIR        = os.path.join(ROOT, "preprocessed", "faces")
AUDIO_DIR        = os.path.join(ROOT, "preprocessed", "audio")
TRANSCRIPTS_DIR  = os.path.join(ROOT, "preprocessed", "transcripts")

# Videos
VIDEO_INPUT_DIR  = os.path.join(ROOT, "videos", "input")
VIDEO_OUTPUT_DIR = os.path.join(ROOT, "videos", "output")
HIGHLIGHTS_DIR   = os.path.join(ROOT, "highlights")

# Model weights
CNN_MODEL_PATH  = os.path.join(ROOT, "models", "cnn",  "mobilenetv2_fer2013_savedmodel")
LSTM_MODEL_PATH = os.path.join(ROOT, "models", "lstm", "lstm_ravdess_savedmodel")
BERT_MODEL_DIR   = os.path.join(ROOT, "models", "bert", "bert_meld")

# Output dirs
FUSION_DIR       = os.path.join(ROOT, "fusion")
REPORTS_DIR      = os.path.join(ROOT, "reports")

# Preprocessing
FRAME_RATE       = 1          # 1 frame per second
FACE_SIZE        = (48, 48)
MFCC_N_COEFF     = 40
SEGMENT_SEC      = 8         # seconds per segment

# Training (used in Colab for CNN and LSTM)
CNN_EPOCHS       = 20
CNN_BATCH_SIZE   = 64
CNN_LR           = 1e-4

LSTM_EPOCHS      = 30
LSTM_BATCH_SIZE  = 32
LSTM_LR          = 1e-3

BERT_EPOCHS      = 3
BERT_BATCH_SIZE  = 16
BERT_LR          = 2e-5
BERT_MAX_LEN     = 128

# Emotion labels
FER_EMOTIONS     = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
RAVDESS_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
MELD_EMOTIONS    = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Fusion weights (must sum to 1.0)
FUSION_WEIGHTS = {
    "cnn":       0.30,
    "lstm":      0.20,
    "bert":      0.20,
    "audio_fe":  0.15,
    "visual_fe": 0.15,
}

# Highlight reel
TOP_N_SEGMENTS   = 6
MIN_SEGMENT_GAP  = 3   # seconds

# ==============================================================================
# --- LONG-FORM LECTURE / PODCAST CONFIGURATIONS ---
# ==============================================================================

LECTURE_COMPRESSION_RATIO = 0.30  # Keeps the top 30% most important content of ANY video length
LECTURE_BLOCK_IDEAL_SEC = 45  # Aim for ~45-second blocks, allowing stretch to finish sentences

# Adjust weights: Prioritize semantic text value and audio presence over raw facial expressions
LECTURE_FUSION_WEIGHTS = {
    "cnn": 0.05,        # Faces matter less in a static/long-form setup
    "lstm": 0.15,       # Vocal tone tracking
    "bert": 0.50,       # High priority: Semantic importance of the words spoken
    "audio_fe": 0.20,   # High priority: Filters out dead air or low-energy mumbles
    "visual_fe": 0.10   # Detects camera cuts or presentation slide changes
}