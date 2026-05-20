# fusion.py
# Phase 4 — Inference + Fusion Layer
#
# Usage:
#   python fusion.py --video videos/input/your_video.mp4
#   python fusion.py --video videos/input/your_video.mp4 --mode lecture

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtcnn import MTCNN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CNN_MODEL_PATH, LSTM_MODEL_PATH, BERT_MODEL_DIR,
    AUDIO_DIR, TRANSCRIPTS_DIR, FUSION_DIR, REPORTS_DIR,
    SEGMENT_SEC, MFCC_N_COEFF, FACE_SIZE,
    FER_EMOTIONS, RAVDESS_EMOTIONS, MELD_EMOTIONS,
    FUSION_WEIGHTS, TOP_N_SEGMENTS,
    LECTURE_BLOCK_IDEAL_SEC, LECTURE_FUSION_WEIGHTS
)

# Emotion intensity weights used internally for scoring — not shown to user
EMOTION_INTENSITY = {
    "angry":    0.85, "disgust":  0.60, "fear":     0.80,
    "happy":    0.90, "sad":      0.70, "surprise": 0.95,
    "neutral":  0.10, "calm":     0.10, "fearful":  0.80,
    "surprised":0.95, "joy":      0.90, "sadness":  0.70,
    "anger":    0.85,
}

# ── Load Models ───────────────────────────────────────────────────────────────

def load_cnn_model():
    print("  Loading CNN model...", end=" ", flush=True)
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"NOT FOUND at {CNN_MODEL_PATH}")
        return None
    import tensorflow as tf
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print("OK")
    return model

def load_lstm_model():
    print("  Loading LSTM model...", end=" ", flush=True)
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"NOT FOUND at {LSTM_MODEL_PATH}")
        return None
    import tensorflow as tf
    model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    print("OK")
    return model

def load_bert_model():
    print("  Loading BERT model...", end=" ", flush=True)
    if not os.path.exists(BERT_MODEL_DIR):
        print(f"NOT FOUND at {BERT_MODEL_DIR}")
        return None, None
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    print("OK")
    return tokenizer, model

# ── Dynamic Sentence Chunking ─────────────────────────────────────────────────

def group_by_complete_sentences(transcript_df, ideal_duration=45):
    """
    Groups Whisper rows so that clips never break mid-sentence.
    Each block will start at the beginning of a sentence and end at the close of one.
    """
    grouped_chunks = []
    current_sentences = []
    current_start = None
    
    for idx, row in transcript_df.iterrows():
        if current_start is None:
            current_start = float(row["start_sec"])
            
        current_sentences.append(str(row["text"]).strip())
        current_end = float(row["end_sec"])
        
        duration = current_end - current_start
        text_str = " ".join(current_sentences).strip()
        
        # Check if we've met the ideal length AND the speaker finished their thought
        is_sentence_end = text_str.endswith('.') or text_str.endswith('!') or text_str.endswith('?')
        
        if duration >= ideal_duration and is_sentence_end:
            grouped_chunks.append({
                "start_sec": round(current_start, 2),
                "end_sec": round(current_end, 2),
                "transcript_text": text_str
            })
            # Reset trackers for the next conversational block
            current_sentences = []
            current_start = None
            
    # Sweep up any remaining sentences left over at the very end of the video
    if current_sentences and current_start is not None:
        grouped_chunks.append({
            "start_sec": round(current_start, 2),
            "end_sec": round(transcript_df["end_sec"].iloc[-1], 2),
            "transcript_text": " ".join(current_sentences).strip()
        })
        
    return pd.DataFrame(grouped_chunks)


# ── CNN Inference ─────────────────────────────────────────────────────────────

def predict_cnn_segment(frames_bgr, cnn_model):
    if cnn_model is None:
        return np.ones(len(FER_EMOTIONS)) / len(FER_EMOTIONS)

    detector = MTCNN()
    preds = []

    for frame in frames_bgr:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        if not faces:
            continue
        best = max(faces, key=lambda f: f["confidence"])
        x, y, w, h = best["box"]
        x, y = max(0, x), max(0, y)
        face = img_rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face_resized = cv2.resize(face, FACE_SIZE)
        face_input = face_resized.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_input, axis=0)
        pred = cnn_model.predict(face_input, verbose=0)[0]
        preds.append(pred)

    if not preds:
        return np.ones(len(FER_EMOTIONS)) / len(FER_EMOTIONS)
    return np.mean(preds, axis=0)

# ── LSTM Inference ────────────────────────────────────────────────────────────

MAX_LSTM_LEN = 200

def extract_segment_audio_features(y_seg, sr):
    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=MFCC_N_COEFF)
    delta_mfcc = librosa.feature.delta(mfcc)
    f0 = librosa.yin(y_seg, fmin=50, fmax=500, frame_length=2048)
    f0 = f0[np.newaxis, :]
    rms = librosa.feature.rms(y=y_seg)
    features = np.vstack([mfcc, delta_mfcc, f0, rms]).T
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std
    if features.shape[0] < MAX_LSTM_LEN:
        pad = np.zeros((MAX_LSTM_LEN - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:MAX_LSTM_LEN]
    return features

def predict_lstm_segment(y_seg, sr, lstm_model):
    if lstm_model is None or len(y_seg) < sr * 0.1:
        return np.ones(len(RAVDESS_EMOTIONS)) / len(RAVDESS_EMOTIONS)
    features = extract_segment_audio_features(y_seg, sr)
    x = np.expand_dims(features, axis=0).astype(np.float32)
    return lstm_model.predict(x, verbose=0)[0]

# ── BERT Inference ────────────────────────────────────────────────────────────

def predict_bert_text(text, bert_tokenizer, bert_model):
    if bert_tokenizer is None or not text.strip():
        return np.ones(len(MELD_EMOTIONS)) / len(MELD_EMOTIONS)
    import torch
    encoding = bert_tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        outputs = bert_model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    return probs

# ── Feature Scores ────────────────────────────────────────────────────────────

def audio_feature_score(y_seg, sr):
    if len(y_seg) < sr * 0.1:
        return 0.0
    rms = float(np.mean(librosa.feature.rms(y=y_seg)))
    rms_score = min(rms / 0.1, 1.0)
    f0 = librosa.yin(y_seg, fmin=50, fmax=500)
    voiced = f0[f0 > 0]
    if len(voiced) > 1:
        pitch_score = min(float(np.std(voiced) / (np.mean(voiced) + 1e-8)) / 0.3, 1.0)
    else:
        pitch_score = 0.0
    onsets = librosa.onset.onset_detect(y=y_seg, sr=sr)
    rate_score = min(len(onsets) / (SEGMENT_SEC * 3), 1.0)
    return 0.4 * rms_score + 0.4 * pitch_score + 0.2 * rate_score

def visual_feature_score(frames_bgr):
    if not frames_bgr:
        return 0.0
    detector = MTCNN()
    face_count = 0
    motion_scores = []
    prev_gray = None
    for frame in frames_bgr:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if detector.detect_faces(img_rgb):
            face_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motion_scores.append(float(cv2.absdiff(gray, prev_gray).mean()))
        prev_gray = gray
    face_rate    = face_count / len(frames_bgr)
    motion_score = min(float(np.mean(motion_scores)) if motion_scores else 0.0, 20.0) / 20.0
    return 0.6 * face_rate + 0.4 * motion_score

def emotion_intensity_score(prob_vector, emotion_labels):
    return float(sum(
        prob * EMOTION_INTENSITY.get(label.lower(), 0.5)
        for prob, label in zip(prob_vector, emotion_labels)
    ))

# ── Full Fusion Pipeline ──────────────────────────────────────────────────────

def run_fusion(video_path, cnn_model, lstm_model, bert_tokenizer, bert_model,
               transcript_csv=None, mode="highlight"):
    print(f"\nRunning fusion pipeline on: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    print(f"  Duration: {duration_sec:.1f}s at {video_fps:.1f} fps")

    audio_path = os.path.join(AUDIO_DIR, f"{video_name}.wav")
    if os.path.exists(audio_path):
        y_audio, sr = librosa.load(audio_path, sr=None)
        print(f"  Audio loaded: {len(y_audio)/sr:.1f}s at {sr}Hz")
    else:
        y_audio, sr = None, 22050
        print("  WARNING: Audio file not found. Run preprocess.py first.")

    # --- ADVANCED TRANSCRIPT LOADING (CSV + JSON SUPPORT) ---
    transcript_df = None
    auto_csv = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_segments.csv")
    auto_json = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_transcript.json")
    auto_json_alt = os.path.join("data", "processed", f"{video_name}_transcript.json")
    
    if transcript_csv and os.path.exists(transcript_csv):
        transcript_df = pd.read_csv(transcript_csv)
    elif os.path.exists(auto_csv):
        transcript_df = pd.read_csv(auto_csv)
    elif os.path.exists(auto_json) or os.path.exists(auto_json_alt):
        json_to_load = auto_json if os.path.exists(auto_json) else auto_json_alt
        with open(json_to_load, 'r') as f:
            t_data = json.load(f)
            segs = t_data.get('segments', t_data) if isinstance(t_data, dict) else t_data
            transcript_df = pd.DataFrame(segs)
            if 'start' in transcript_df.columns:
                transcript_df = transcript_df.rename(columns={'start': 'start_sec', 'end': 'end_sec'})
    
    if transcript_df is not None and not transcript_df.empty:
        transcript_df["start_sec"] = pd.to_numeric(transcript_df["start_sec"], errors='coerce')
        transcript_df["end_sec"] = pd.to_numeric(transcript_df["end_sec"], errors='coerce')
        print(f"  Transcript successfully loaded: {len(transcript_df)} spoken segments found.")
    else:
        print("  WARNING: No transcript found or format unsupported. BERT scores will be uniform (0.70).")

    # --- DYNAMIC SEGMENT GENERATOR ---
    fusion_blocks = []
    
    if mode == "lecture" and transcript_df is not None and not transcript_df.empty:
        print("  -> Grouping video frames by complete structural sentences...")
        blocks_df = group_by_complete_sentences(transcript_df, ideal_duration=LECTURE_BLOCK_IDEAL_SEC)
        for _, row in blocks_df.iterrows():
            fusion_blocks.append({
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "text": row["transcript_text"]
            })
        weights = LECTURE_FUSION_WEIGHTS
    else:
        if mode == "lecture":
            print("  WARNING: Lecture mode requires a transcript. Falling back to time-based highlight mode.")
        num_segments = int(np.ceil(duration_sec / SEGMENT_SEC))
        for seg_idx in range(num_segments):
            fusion_blocks.append({
                "start_sec": seg_idx * SEGMENT_SEC,
                "end_sec": min((seg_idx + 1) * SEGMENT_SEC, duration_sec),
                "text": None # Will pull from overlap
            })
        weights = FUSION_WEIGHTS

    print(f"  Processing {len(fusion_blocks)} segments in {mode.upper()} mode...")

    segment_scores = []

    for seg_idx, block in enumerate(tqdm(fusion_blocks, desc="  Fusion")):
        t_start = block["start_sec"]
        t_end   = block["end_sec"]
        
        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * video_fps))
        frames = []
        for _ in range(int((t_end - t_start) * video_fps)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if len(frames) > 5:
            idxs = np.linspace(0, len(frames)-1, 5, dtype=int)
            frames = [frames[i] for i in idxs]

        # CNN
        cnn_probs     = predict_cnn_segment(frames, cnn_model)
        cnn_intensity = emotion_intensity_score(cnn_probs, FER_EMOTIONS)

        # LSTM
        if y_audio is not None:
            y_seg          = y_audio[int(t_start * sr):int(t_end * sr)]
            lstm_probs     = predict_lstm_segment(y_seg, sr, lstm_model)
            lstm_intensity = emotion_intensity_score(lstm_probs, RAVDESS_EMOTIONS)
            audio_fe       = audio_feature_score(y_seg, sr)
        else:
            lstm_intensity = 0.5
            audio_fe       = 0.0

        # BERT
        text = block["text"]
        if text is None:
            if transcript_df is not None and not transcript_df.empty:
                mask = (transcript_df["start_sec"] < t_end) & (transcript_df["end_sec"] > t_start)
                text = " ".join(transcript_df[mask]["text"].astype(str).tolist()).strip()
            else:
                text = ""
                
        bert_probs     = predict_bert_text(text, bert_tokenizer, bert_model)
        bert_intensity = emotion_intensity_score(bert_probs, MELD_EMOTIONS)

        # Visual features
        vis_fe = visual_feature_score(frames)

        # Weighted fusion (Dynamically switched based on mode)
        fusion_score = (
            weights["cnn"]       * cnn_intensity  +
            weights["lstm"]      * lstm_intensity +
            weights["bert"]      * bert_intensity +
            weights["audio_fe"]  * audio_fe       +
            weights["visual_fe"] * vis_fe
        )

        segment_scores.append({
            "segment_idx":     seg_idx,
            "start_sec":       round(t_start, 2),
            "end_sec":         round(t_end, 2),
            "fusion_score":    round(fusion_score, 4),
            "cnn_score":       round(cnn_intensity, 4),
            "lstm_score":      round(lstm_intensity, 4),
            "bert_score":      round(bert_intensity, 4),
            "audio_fe_score":  round(audio_fe, 4),
            "visual_fe_score": round(vis_fe, 4),
            "transcript_text": text[:250],
        })

    cap.release()

    os.makedirs(FUSION_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    json_path = os.path.join(FUSION_DIR, f"{video_name}_{mode}_scores.json")
    csv_path  = os.path.join(FUSION_DIR, f"{video_name}_{mode}_scores.csv")
    with open(json_path, "w") as f:
        json.dump(segment_scores, f, indent=2)
    df = pd.DataFrame(segment_scores)
    df.to_csv(csv_path, index=False)

    print(f"\n  Scores saved:")
    print(f"    JSON: {json_path}")
    print(f"    CSV:  {csv_path}")

    plot_fusion_scores(df, video_name)
    return df, json_path

def plot_fusion_scores(df, video_name):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    x = df["start_sec"]
    # Dynamically calculate block widths for the chart since they now vary in lecture mode
    widths = df["end_sec"] - df["start_sec"]

    top_threshold = df["fusion_score"].nlargest(TOP_N_SEGMENTS).min()
    colors = ["#e74c3c" if s >= top_threshold else "#3498db" for s in df["fusion_score"]]

    axes[0].bar(x, df["fusion_score"], width=widths * 0.8, color=colors, alpha=0.85, align='edge')
    axes[0].axhline(top_threshold, color="red", linestyle="--", linewidth=1,
                    label=f"Top {TOP_N_SEGMENTS} threshold")
    axes[0].set_ylabel("Fusion Score")
    axes[0].set_title(f"Segment Significance Scores — {video_name}")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].plot(x, df["cnn_score"],  label="CNN (Visual)", marker="o", ms=3)
    axes[1].plot(x, df["lstm_score"], label="LSTM (Audio)", marker="s", ms=3)
    axes[1].plot(x, df["bert_score"], label="BERT (Text)",  marker="^", ms=3)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Individual Model Scores")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    axes[2].plot(x, df["audio_fe_score"],  label="Audio Features", color="orange")
    axes[2].plot(x, df["visual_fe_score"], label="Visual Features", color="green")
    axes[2].set_ylabel("Score")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Handcrafted Feature Scores")
    axes[2].legend()
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, f"{video_name}_fusion_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"    Chart: {chart_path}")

    top_segs = df.nlargest(TOP_N_SEGMENTS, "fusion_score")[
        ["segment_idx", "start_sec", "end_sec", "fusion_score"]
    ]
    print(f"\n  Top {TOP_N_SEGMENTS} segments:")
    print(top_segs.to_string(index=False))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str, required=True)
    parser.add_argument("--transcript", type=str, default=None)
    parser.add_argument("--skip_cnn",  action="store_true")
    parser.add_argument("--skip_lstm", action="store_true")
    parser.add_argument("--skip_bert", action="store_true")
    parser.add_argument("--mode",       type=str, default="highlight", choices=["highlight", "lecture"])
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    print("=" * 60)
    print("  AI-Driven Video Summarization System — Fusion Layer")
    print("=" * 60)
    print("\nLoading models...")

    cnn_model  = None if args.skip_cnn  else load_cnn_model()
    lstm_model = None if args.skip_lstm else load_lstm_model()
    bert_tokenizer, bert_model = (None, None) if args.skip_bert else load_bert_model()

    df, json_path = run_fusion(
        args.video, cnn_model, lstm_model,
        bert_tokenizer, bert_model,
        transcript_csv=args.transcript,
        mode=args.mode
    )

    print("\n" + "=" * 60)
    print("  Fusion complete.")
    print(f"  Scores: {json_path}")
    print("  Next: python highlight.py --video <path> --scores <json_path>")
    print("=" * 60)

if __name__ == "__main__":
    main()