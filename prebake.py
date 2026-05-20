# prebake.py
# Pre-bakes the full pipeline for demo videos so results load instantly during demo.
#
# Run this ONCE tonight for each video:
#   python prebake.py --video videos/input/tedtalk.mp4 --mode lecture
#   python prebake.py --video videos/input/interview.mp4 --mode highlight
#
import os
import imageio_ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    AUDIO_DIR, TRANSCRIPTS_DIR, FUSION_DIR, HIGHLIGHTS_DIR,
    TOP_N_SEGMENTS, LECTURE_COMPRESSION_RATIO
)


def fmt_time(seconds):
    m, s = int(seconds) // 60, int(seconds) % 60
    return f"{m}m {s:02d}s"


def fmt_elapsed(start):
    return f"{time.time() - start:.1f}s"


def prebake(video_path, whisper_model="base", top_n=TOP_N_SEGMENTS,
            skip_bert=False, skip_whisper=False, mode="highlight"):

    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    total_start = time.time()

    print("=" * 60)
    print(f"  Pre-baking: {video_name} [{mode.upper()} MODE]")
    print("=" * 60)

    # ── Check if already baked ────────────────────────────────────────────────
    json_path      = os.path.join(FUSION_DIR, f"{video_name}_{mode}_scores.json")
    highlight_path = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_{mode}.mp4")

    if os.path.exists(json_path) and os.path.exists(highlight_path):
        print(f"\n  Already baked! Results exist:")
        print(f"    Scores:    {json_path}")
        print(f"    Highlight: {highlight_path}")
        print(f"\n  Delete those files and re-run if you want to redo.")
        return

    # ── Step 1: Audio extraction ──────────────────────────────────────────────
    print(f"\n[1/5] Extracting audio...")
    t = time.time()
    from preprocess import extract_audio
    audio_path = extract_audio(video_path, AUDIO_DIR)
    if audio_path:
        print(f"      Done in {fmt_elapsed(t)}")
    else:
        print("      WARNING: No audio track found.")

    # ── Step 2: Whisper transcription ─────────────────────────────────────────
    transcript_df = None
    if not skip_whisper and audio_path:
        print(f"\n[2/5] Transcribing with Whisper ({whisper_model})...")
        print(f"      This takes 2-5 min for a 10-min video. Please wait.")
        t = time.time()
        from preprocess import transcribe_audio
        try:
            _, segments_csv, transcript_df = transcribe_audio(
                audio_path, TRANSCRIPTS_DIR, model_size=whisper_model
            )
            print(f"      Done in {fmt_elapsed(t)} — {len(transcript_df)} segments")
        except Exception as e:
            print(f"      Whisper failed: {e}")
            print("      Continuing without transcript (BERT will use uniform scores).")
    else:
        print(f"\n[2/5] Whisper skipped.")
        # Check if transcript already exists
        auto_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}_segments.csv")
        if os.path.exists(auto_path):
            import pandas as pd
            transcript_df = pd.read_csv(auto_path)
            print(f"      Found existing transcript: {len(transcript_df)} segments")

    # ── Step 3: Load models ───────────────────────────────────────────────────
    print(f"\n[3/5] Loading models...")
    t = time.time()
    from fusion import load_cnn_model, load_lstm_model, load_bert_model
    cnn_model  = load_cnn_model()
    lstm_model = load_lstm_model()
    if skip_bert or transcript_df is None:
        bert_tokenizer, bert_model = None, None
        print("      BERT skipped (no transcript).")
    else:
        bert_tokenizer, bert_model = load_bert_model()
    print(f"      Models loaded in {fmt_elapsed(t)}")

    # ── Step 4: Fusion ────────────────────────────────────────────────────────
    print(f"\n[4/5] Running fusion inference ({mode} mode)...")
    print(f"      This takes 4-6 min for a 6-min video on CPU. Please wait.")
    t = time.time()
    from fusion import run_fusion
    scores_df, json_path = run_fusion(
        video_path, cnn_model, lstm_model,
        bert_tokenizer, bert_model,
        mode=mode
    )
    print(f"      Fusion done in {fmt_elapsed(t)}")

    # ── Step 5: Highlight reel ────────────────────────────────────────────────
    print(f"\n[5/5] Building highlight reel...")
    t = time.time()
    from highlight import select_segments, select_lecture_chunks, build_highlight_reel, save_summary

    if mode == "lecture":
        selected_df = select_lecture_chunks(scores_df, compression_ratio=LECTURE_COMPRESSION_RATIO)
    else:
        selected_df = select_segments(scores_df, top_n=top_n)
        
    os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)
    highlight_path = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_{mode}.mp4")

    result = build_highlight_reel(video_path, selected_df, highlight_path, add_labels=True)
    if result:
        save_summary(selected_df, highlight_path, video_name)
        print(f"      Reel built in {fmt_elapsed(t)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    import cv2
    cap = cv2.VideoCapture(video_path)
    orig_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if not selected_df.empty:
        highlight_dur = selected_df["end_sec"].sub(selected_df["start_sec"]).sum()
        compression   = (1 - highlight_dur / orig_dur) * 100
    else:
        highlight_dur = 0
        compression = 0

    print("\n" + "=" * 60)
    print(f"  Pre-bake complete for: {video_name}")
    print(f"  Mode:           {mode.upper()}")
    print(f"  Total time:     {fmt_elapsed(total_start)}")
    print(f"  Original:       {fmt_time(orig_dur)}")
    print(f"  Final Output:   {fmt_time(highlight_dur)}")
    print(f"  Compressed:     {compression:.0f}%")
    print(f"  Scores JSON:    {json_path}")
    print(f"  Highlight reel: {highlight_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-bake pipeline for demo videos")
    parser.add_argument("--video",         type=str, required=True)
    parser.add_argument("--whisper_model", type=str, default="base",
                        choices=["tiny", "base", "small"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--top_n",         type=int, default=TOP_N_SEGMENTS)
    parser.add_argument("--skip_whisper",  action="store_true",
                        help="Skip Whisper (faster, BERT uses uniform scores)")
    parser.add_argument("--skip_bert",     action="store_true",
                        help="Skip BERT model entirely")
    parser.add_argument("--mode",          type=str, default="highlight", 
                        choices=["highlight", "lecture"],
                        help="Choose between emotional highlights or long-form lecture summary")
    args = parser.parse_args()

    prebake(
        args.video,
        whisper_model=args.whisper_model,
        top_n=args.top_n,
        skip_bert=args.skip_bert,
        skip_whisper=args.skip_whisper,
        mode=args.mode
    )