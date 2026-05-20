# highlight.py
# Phase 5a — Highlight Reel Generator with Subtitles
#
# Usage:
#   python highlight.py --video videos/input/sample_video.mp4 --scores fusion/sample_video_scores.json
#   python highlight.py --video videos/input/sample_video.mp4 --scores fusion/sample_video_scores.json --mode lecture

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips,
    TextClip, CompositeVideoClip
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    HIGHLIGHTS_DIR, TOP_N_SEGMENTS, MIN_SEGMENT_GAP, SEGMENT_SEC,
    LECTURE_COMPRESSION_RATIO
)


# ── Segment Selection (Short Highlights) ──────────────────────────────────────

def select_segments(scores_df, top_n=TOP_N_SEGMENTS, min_gap=MIN_SEGMENT_GAP):
    """Original mode: Picks the top N highest emotional spikes with gap filtering."""
    sorted_df = scores_df.sort_values("fusion_score", ascending=False).reset_index(drop=True)
    selected = []
    for _, row in sorted_df.iterrows():
        too_close = any(abs(row["start_sec"] - s["start_sec"]) < min_gap for s in selected)
        if not too_close:
            selected.append(row)
        if len(selected) == top_n:
            break
    return pd.DataFrame(selected).sort_values("start_sec").reset_index(drop=True)


# ── Segment Selection (Long-Form Lectures/Podcasts) ───────────────────────────

def select_lecture_chunks(scores_df, compression_ratio=0.30):
    """
    Lecture mode: Selects top semantic blocks until the dynamic target duration 
    is reached (e.g., top 30% of the video), maintaining chronological order.
    """
    total_video_duration = scores_df["end_sec"].max()
    target_duration_sec = total_video_duration * compression_ratio
    
    print(f"  Total Video Length: {total_video_duration / 60:.1f} minutes")
    print(f"  Compression Target: {target_duration_sec / 60:.1f} minutes ({compression_ratio*100:.0f}%)")
    
    # 1. Sort by the highest semantic importance
    sorted_by_value = scores_df.sort_values("fusion_score", ascending=False)
    
    selected_blocks = []
    current_total_duration_sec = 0
    
    for _, row in sorted_by_value.iterrows():
        block_dur = row["end_sec"] - row["start_sec"]
        
        # Keep picking blocks until we fill our target time bucket
        if current_total_duration_sec + block_dur <= target_duration_sec:
            selected_blocks.append(row)
            current_total_duration_sec += block_dur
            
    # 2. CRUCIAL: Sort chronologically by start time so the lecture flows logically
    if not selected_blocks:
        return pd.DataFrame()
        
    final_ordered_df = pd.DataFrame(selected_blocks).sort_values("start_sec").reset_index(drop=True)
    return final_ordered_df


# ── Subtitles & Score Overlays ────────────────────────────────────────────────

def add_render_overlays(clip, start_sec, score, transcript, duration):
    """Combines minimal significance metrics and spoken word subtitles."""
    layers = [clip]
    w, h = clip.w, clip.h

    # 1. Spoken Word Subtitles
    if transcript and str(transcript).strip() and str(transcript).strip() != "Segment transcription unavailable.":
        try:
            raw_text = str(transcript).strip()
            
            # Smart line-wrapping logic (keeps chunks under 45 characters)
            words = raw_text.split()
            wrapped_lines = []
            current_line = []
            for word in words:
                if len(" ".join(current_line + [word])) <= 45:
                    current_line.append(word)
                else:
                    wrapped_lines.append(" ".join(current_line))
                    current_line = [word]
            if current_line:
                wrapped_lines.append(" ".join(current_line))
            wrapped_text = "\n".join(wrapped_lines)

            sub_txt = TextClip(
                wrapped_text,
                fontsize=min(24, int(h * 0.05)),
                color="white",
                font="Arial-Bold",
                stroke_color="black",
                stroke_width=1.5,
                method="caption",
                size=(int(w * 0.85), None)
            ).set_duration(duration).set_position(("center", int(h * 0.82)))
            
            layers.append(sub_txt)
        except Exception as e:
            pass

    # 2. Minimal Significance Metric Overlay
    try:
        label = f"[{start_sec:.0f}s]  Significance: {score:.2f}"
        metric_txt = TextClip(
            label,
            fontsize=min(16, int(h * 0.035)),
            color="rgba(255,255,255,0.8)",
            font="Arial",
            bg_color="rgba(0,0,0,0.4)"
        ).set_duration(duration).set_position(("center", int(h * 0.05)))
        
        layers.append(metric_txt)
    except Exception:
        pass

    if len(layers) > 1:
        return CompositeVideoClip(layers)
    return clip


# ── Highlight Reel Compilation ────────────────────────────────────────────────

def build_highlight_reel(video_path, selected_df, output_path, add_labels=True):
    print(f"\nBuilding highlight reel with subtitles...")
    print(f"  Source: {video_path}")
    print(f"  Segments: {len(selected_df)}")

    source = VideoFileClip(video_path)
    clips  = []

    for idx, row in selected_df.iterrows():
        start = float(row["start_sec"])
        end   = min(float(row["end_sec"]), source.duration)
        score = float(row["fusion_score"])
        
        transcript = row.get("transcript_text", row.get("transcript", ""))

        if start >= source.duration:
            print(f"  Skipping segment {idx} — beyond video duration")
            continue

        clip = source.subclip(start, end).fadein(0.2).fadeout(0.2)

        if add_labels:
            clip = add_render_overlays(clip, start, score, transcript, clip.duration)

        clips.append(clip)
        print(f"  Clip {idx+1}: {start:.1f}s - {end:.1f}s | Score: {score:.4f}")

    if not clips:
        print("  ERROR: No clips to stitch.")
        source.close()
        return None

    print(f"\n  Concatenating {len(clips)} multi-layered video streams...")
    final = concatenate_videoclips(clips, method="compose")

    print(f"  Writing file to output destination: {output_path}")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=source.fps,
        verbose=False,
        logger=None,
    )

    source.close()
    final.close()
    for c in clips:
        c.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done. File size: {size_mb:.1f} MB")
    print(f"  Saved: {output_path}")
    return output_path


# ── Summary Log Compilation ───────────────────────────────────────────────────

def save_summary(selected_df, output_path, video_name):
    total_dur = selected_df["end_sec"].sub(selected_df["start_sec"]).sum()
    lines = [
        f"Highlight Reel Summary — {video_name}",
        "=" * 50,
        f"Total clips:    {len(selected_df)}",
        f"Total duration: {total_dur:.1f}s",
        "",
        "Selected Segments (Chronological Order):",
        "-" * 50,
    ]
    for idx, row in selected_df.iterrows():
        dur = row["end_sec"] - row["start_sec"]
        lines.append(
            f"  Clip {idx+1}: {row['start_sec']:.1f}s - {row['end_sec']:.1f}s "
            f"({dur:.1f}s) | Score: {row['fusion_score']:.4f}"
        )
        transcript = row.get("transcript_text", row.get("transcript", ""))
        if transcript:
            lines.append(f"    Text: \"{str(transcript)[:100]}\"")
        lines.append("")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = output_path.replace(".mp4", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Summary saved: {summary_path}")
    return summary_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str, required=True)
    parser.add_argument("--scores",     type=str, required=True)
    parser.add_argument("--top_n",      type=int, default=TOP_N_SEGMENTS)
    parser.add_argument("--no_labels",  action="store_true")
    parser.add_argument("--mode",       type=str, default="highlight", choices=["highlight", "lecture"])
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.scores):
        print(f"ERROR: Scores JSON not found: {args.scores}")
        sys.exit(1)

    print("=" * 60)
    print("  AI-Driven Video Summarization System — Highlight Reel")
    print("=" * 60)

    with open(args.scores) as f:
        scores_df = pd.DataFrame(json.load(f))

    # Route to the correct logic based on mode
    if args.mode == "lecture":
        selected_df = select_lecture_chunks(scores_df, compression_ratio=LECTURE_COMPRESSION_RATIO)
        print(f"\nSelected {len(selected_df)} chronological segments based on the {LECTURE_COMPRESSION_RATIO*100}% compression target.")
    else:
        selected_df = select_segments(scores_df, top_n=args.top_n)
        print(f"\nSelected {len(selected_df)} segments after gap filtering.")

    os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)
    video_name  = os.path.splitext(os.path.basename(args.video))[0]
    output_path = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_{args.mode}.mp4")

    result = build_highlight_reel(
        args.video, selected_df, output_path,
        add_labels=not args.no_labels
    )

    if result:
        save_summary(selected_df, output_path, video_name)
        print("\n" + "=" * 60)
        print("  Highlight reel ready.")
        print(f"  File: {output_path}")
        print("  Next: python app.py")
        print("=" * 60)


if __name__ == "__main__":
    main()