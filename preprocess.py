# preprocess.py
# Full preprocessing pipeline — run after download_datasets.py
#
# What this does:
#   1. Verifies datasets are present
#   2. Extracts frames from a test video (1 fps)
#   3. Detects and crops faces from frames using MTCNN
#   4. Separates audio from video using moviepy
#   5. Extracts MFCCs, pitch (F0), and RMS from audio
#   6. Transcribes audio using Whisper
#   7. Previews MELD CSV structure
#
# Usage:
#   python preprocess.py --video videos/input/your_video.mp4
#   python preprocess.py --video videos/input/your_video.mp4 --whisper_model small

import os
import imageio_ffmpeg

# Add the local ffmpeg binary to the system PATH so Whisper can find it
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtcnn import MTCNN
from moviepy.editor import VideoFileClip

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    FER2013_DIR, RAVDESS_DIR, MELD_DIR,
    FRAMES_DIR, FACES_DIR, AUDIO_DIR, TRANSCRIPTS_DIR,
    REPORTS_DIR, FRAME_RATE, FACE_SIZE, MFCC_N_COEFF, SEGMENT_SEC,
    FER_EMOTIONS
)


# ── 1. Dataset Verification ───────────────────────────────────────────────────

def verify_datasets():
    print("\n[Step 1] Verifying datasets...")

    checks = {
        "FER-2013": {
            "path": FER2013_DIR,
            "look_for": ["train", "test"],  # folders inside
        },
        "RAVDESS": {
            "path": RAVDESS_DIR,
            "look_for": [".wav"],           # file extensions
        },
        "MELD": {
            "path": MELD_DIR,
            "look_for": ["train_sent_emo.csv", "test_sent_emo.csv"],
        },
    }

    all_ok = True
    for name, cfg in checks.items():
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"  MISSING: {name} — folder not found: {path}")
            all_ok = False
            continue

        found = False
        for item in cfg["look_for"]:
            if item.startswith("."):
                # Check for any file with this extension
                for root, _, files in os.walk(path):
                    if any(f.endswith(item) for f in files):
                        found = True
                        break
            else:
                if os.path.exists(os.path.join(path, item)):
                    found = True
                    break

        if found:
            print(f"  OK: {name}")
        else:
            print(f"  WARNING: {name} — folder exists but expected files not found")
            print(f"           Expected: {cfg['look_for']}")
            all_ok = False

    return all_ok


# ── 2. Frame Extraction ───────────────────────────────────────────────────────

def extract_frames(video_path, output_dir, fps=FRAME_RATE):
    """Extract frames from video at given FPS rate."""
    print(f"\n[Step 2] Extracting frames from: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return None, []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    frame_interval = int(video_fps / fps)  # extract every Nth frame

    print(f"  Video FPS: {video_fps:.1f}, Duration: {duration_sec:.1f}s")
    print(f"  Extracting 1 frame every {frame_interval} frames ({fps} fps)")

    saved_frames = []
    frame_idx = 0
    saved_idx = 0

    with tqdm(total=int(duration_sec), unit="sec", desc="  Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / video_fps
                filename = f"frame_{saved_idx:05d}_t{timestamp_sec:.2f}s.jpg"
                filepath = os.path.join(frame_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_frames.append({
                    "path": filepath,
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "saved_idx": saved_idx,
                })
                saved_idx += 1
                pbar.update(1)

            frame_idx += 1

    cap.release()
    print(f"  Saved {len(saved_frames)} frames to: {frame_dir}")
    return frame_dir, saved_frames


# ── 3. Face Detection & Cropping ─────────────────────────────────────────────

def crop_faces(saved_frames, output_dir, face_size=FACE_SIZE):
    """Detect and crop faces from extracted frames using MTCNN."""
    print(f"\n[Step 3] Detecting and cropping faces with MTCNN...")

    detector = MTCNN()
    face_records = []
    failed = 0

    os.makedirs(output_dir, exist_ok=True)

    for record in tqdm(saved_frames, desc="  Face detection"):
        img_bgr = cv2.imread(record["path"])
        if img_bgr is None:
            failed += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        if not faces:
            # No face found — save a black placeholder so timeline stays intact
            face_records.append({
                **record,
                "face_path": None,
                "face_found": False,
                "confidence": 0.0,
            })
            continue

        # Take the highest-confidence face
        best_face = max(faces, key=lambda f: f["confidence"])
        x, y, w, h = best_face["box"]

        # Clamp to image bounds
        x, y = max(0, x), max(0, y)
        x2, y2 = min(img_rgb.shape[1], x + w), min(img_rgb.shape[0], y + h)

        face_crop = img_rgb[y:y2, x:x2]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, face_size)

        # Save face crop
        base = os.path.splitext(os.path.basename(record["path"]))[0]
        face_filename = f"face_{base}.jpg"
        face_path = os.path.join(output_dir, face_filename)
        cv2.imwrite(face_path, face_resized)

        face_records.append({
            **record,
            "face_path": face_path,
            "face_found": True,
            "confidence": best_face["confidence"],
        })

    found = sum(1 for r in face_records if r["face_found"])
    print(f"  Faces found: {found}/{len(saved_frames)} frames")
    if failed:
        print(f"  Failed to read: {failed} frames")

    return face_records


# ── 4. Audio Extraction ───────────────────────────────────────────────────────

def extract_audio(video_path, output_dir):
    """Extract audio track from video using moviepy."""
    print(f"\n[Step 4] Extracting audio from video...")

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{video_name}.wav")

    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            print("  WARNING: Video has no audio track.")
            clip.close()
            return None

        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        clip.close()
        print(f"  Audio saved to: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"  ERROR extracting audio: {e}")
        return None


# ── 5. Audio Feature Extraction ───────────────────────────────────────────────

def extract_audio_features(audio_path, segment_sec=SEGMENT_SEC, n_mfcc=MFCC_N_COEFF):
    """
    Extract per-segment audio features:
    - MFCCs (40 coefficients, mean per segment)
    - Pitch F0 (mean per segment)
    - RMS loudness (mean per segment)
    """
    print(f"\n[Step 5] Extracting audio features (MFCCs, Pitch, RMS)...")

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Audio: {duration:.1f}s at {sr}Hz")

    segment_samples = int(segment_sec * sr)
    num_segments = int(np.ceil(len(y) / segment_samples))

    segments = []

    for i in range(num_segments):
        start = i * segment_samples
        end = min(start + segment_samples, len(y))
        seg = y[start:end]

        if len(seg) < sr * 0.1:  # skip segments shorter than 0.1s
            continue

        # MFCCs
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)  # shape: (40,)

        # Pitch (F0) via YIN algorithm
        f0 = librosa.yin(seg, fmin=50, fmax=500)
        f0_mean = float(np.nanmean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0

        # RMS loudness
        rms = librosa.feature.rms(y=seg)[0]
        rms_mean = float(np.mean(rms))

        # Speech rate (onset density)
        onsets = librosa.onset.onset_detect(y=seg, sr=sr)
        speech_rate = len(onsets) / segment_sec

        segments.append({
            "segment_idx": i,
            "start_sec": i * segment_sec,
            "end_sec": min((i + 1) * segment_sec, duration),
            "mfcc_mean": mfcc_mean,
            "pitch_mean": f0_mean,
            "rms_mean": rms_mean,
            "speech_rate": speech_rate,
        })

    print(f"  Extracted features for {len(segments)} segments ({segment_sec}s each)")
    return segments


def save_audio_features(segments, output_path):
    """Save audio features to a .npz file."""
    # Separate mfcc arrays from scalar features
    mfcc_array = np.array([s["mfcc_mean"] for s in segments])
    scalars = [{k: v for k, v in s.items() if k != "mfcc_mean"} for s in segments]
    df = pd.DataFrame(scalars)

    npz_path = output_path.replace(".csv", ".npz")
    csv_path = output_path

    np.savez(npz_path, mfcc=mfcc_array)
    df.to_csv(csv_path, index=False)

    print(f"  MFCCs saved to: {npz_path}")
    print(f"  Scalar features saved to: {csv_path}")
    return csv_path, npz_path


# ── 6. Whisper Transcription ──────────────────────────────────────────────────

def transcribe_audio(audio_path, output_dir, model_size="base"):
    """Transcribe audio using OpenAI Whisper."""
    print(f"\n[Step 6] Transcribing audio with Whisper ({model_size} model)...")
    print("  First run downloads the model weights (~140MB for base). Wait for it.")

    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Save full transcript
    transcript_path = os.path.join(output_dir, f"{video_name}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Save per-segment transcript (with timestamps)
    segments_path = os.path.join(output_dir, f"{video_name}_segments.csv")
    seg_rows = []
    for seg in result["segments"]:
        seg_rows.append({
            "segment_id": seg["id"],
            "start_sec": round(seg["start"], 2),
            "end_sec": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    df = pd.DataFrame(seg_rows)
    df.to_csv(segments_path, index=False)

    print(f"  Transcript saved: {transcript_path}")
    print(f"  Segments CSV saved: {segments_path}")
    print(f"\n  Preview (first 3 segments):")
    for _, row in df.head(3).iterrows():
        print(f"    [{row['start_sec']:.1f}s - {row['end_sec']:.1f}s] {row['text']}")

    return transcript_path, segments_path, df


# ── 7. MELD Dataset Preview ───────────────────────────────────────────────────

def preview_meld():
    """Load and preview the MELD dataset structure."""
    print(f"\n[Step 7] Previewing MELD dataset...")

    csv_path = os.path.join(MELD_DIR, "train_sent_emo.csv")
    if not os.path.exists(csv_path):
        print(f"  MELD CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"\n  Emotion distribution:")
    print(df["Emotion"].value_counts().to_string())
    print(f"\n  Sample rows:")
    print(df[["Utterance", "Emotion", "Sentiment"]].head(5).to_string(index=False))

    # Save emotion distribution chart
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Emotion"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("MELD Training Set — Emotion Distribution")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "meld_emotion_distribution.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\n  Chart saved: {chart_path}")

    return df


# ── 8. Save Face Records ──────────────────────────────────────────────────────

def save_face_records(face_records, video_name, output_dir):
    """Save face detection results to CSV."""
    rows = []
    for r in face_records:
        rows.append({
            "timestamp_sec": r["timestamp_sec"],
            "frame_path": r["path"],
            "face_path": r.get("face_path"),
            "face_found": r.get("face_found", False),
            "confidence": r.get("confidence", 0.0),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{video_name}_face_records.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Face records saved: {csv_path}")
    return csv_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video file (e.g. videos/input/sample.mp4)"
    )
    parser.add_argument(
        "--whisper_model", type=str, default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: base, ~140MB)"
    )
    parser.add_argument(
        "--skip_whisper", action="store_true",
        help="Skip Whisper transcription (saves time during testing)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Preprocessing Pipeline — Emotion Video Summarization")
    print("=" * 60)

    # Step 1: Verify datasets
    verify_datasets()

    # Step 7: MELD preview (no video needed)
    preview_meld()

    # If no video provided, stop here
    if args.video is None:
        print("\n  No --video provided. Dataset verification and MELD preview done.")
        print("  To run full preprocessing:")
        print("    python preprocess.py --video videos/input/your_video.mp4")
        return

    if not os.path.exists(args.video):
        print(f"\n  ERROR: Video file not found: {args.video}")
        return

    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # Step 2: Extract frames
    frame_dir, saved_frames = extract_frames(args.video, FRAMES_DIR)
    if not saved_frames:
        print("  No frames extracted. Exiting.")
        return

    # Step 3: Crop faces
    face_output_dir = os.path.join(FACES_DIR, video_name)
    face_records = crop_faces(saved_frames, face_output_dir)
    save_face_records(face_records, video_name, FACES_DIR)

    # Step 4: Extract audio
    audio_path = extract_audio(args.video, AUDIO_DIR)

    if audio_path:
        # Step 5: Audio features
        segments = extract_audio_features(audio_path)
        audio_feat_csv = os.path.join(AUDIO_DIR, f"{video_name}_audio_features.csv")
        save_audio_features(segments, audio_feat_csv)

        # Step 6: Whisper transcription
        if not args.skip_whisper:
            transcribe_audio(audio_path, TRANSCRIPTS_DIR, model_size=args.whisper_model)
        else:
            print("\n[Step 6] Whisper skipped (--skip_whisper flag set)")

    print("\n" + "=" * 60)
    print("  Phase 2 Complete!")
    print(f"  Frames:       {FRAMES_DIR}/{video_name}/")
    print(f"  Faces:        {FACES_DIR}/{video_name}/")
    print(f"  Audio:        {AUDIO_DIR}/")
    print(f"  Transcripts:  {TRANSCRIPTS_DIR}/")
    print(f"  Reports:      {REPORTS_DIR}/")
    print("\n  Next: Run the model training notebooks in Google Colab.")
    print("=" * 60)


if __name__ == "__main__":
    main()