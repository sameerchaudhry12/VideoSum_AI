# app.py
# AI-Driven Video Summarization System — Gradio Web App
# Includes demo mode: if pre-baked results exist, loads them instantly.
#
# Usage: python app.py

import os
import sys
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    CNN_MODEL_PATH, LSTM_MODEL_PATH, BERT_MODEL_DIR,
    AUDIO_DIR, TRANSCRIPTS_DIR, FUSION_DIR, HIGHLIGHTS_DIR,
    SEGMENT_SEC, TOP_N_SEGMENTS, MIN_SEGMENT_GAP,
    FER_EMOTIONS, RAVDESS_EMOTIONS, MELD_EMOTIONS, FUSION_WEIGHTS,
    LECTURE_COMPRESSION_RATIO
)
from preprocess import extract_audio
from fusion import load_cnn_model, load_lstm_model, load_bert_model, run_fusion
from highlight import select_segments, select_lecture_chunks, build_highlight_reel


# ── Load models once at startup ───────────────────────────────────────────────

print("Loading models at startup...")
CNN_MODEL = load_cnn_model()
LSTM_MODEL = load_lstm_model()
BERT_TOKENIZER, BERT_MODEL = load_bert_model()
print("Models ready.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_time(s):
    m, sec = int(s) // 60, int(s) % 60
    return f"{m}m {sec:02d}s"


def get_video_duration(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0


def build_stats_html(orig_dur, highlight_dur, n_clips):
    compression = (1 - highlight_dur / orig_dur) * 100 if orig_dur > 0 else 0
    return f"""
<div class="stats-row">
  <div class="stat-card">
    <div class="stat-label">ORIGINAL</div>
    <div class="stat-value">{fmt_time(orig_dur)}</div>
  </div>
  <div class="stat-arrow">→</div>
  <div class="stat-card highlight-card">
    <div class="stat-label">FINAL OUTPUT</div>
    <div class="stat-value">{fmt_time(highlight_dur)}</div>
  </div>
  <div class="stat-card compression-card">
    <div class="stat-label">COMPRESSED</div>
    <div class="stat-value">{compression:.0f}%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">CLIPS</div>
    <div class="stat-value">{n_clips}</div>
  </div>
</div>
"""


def build_chart(scores_df, work_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), facecolor="#0f0f13")
    x = scores_df["start_sec"]
    
    # Dynamic widths based on whether it's standard segments or variable lecture chunks
    widths = scores_df["end_sec"] - scores_df["start_sec"]
    
    ax1 = axes[0]
    ax1.set_facecolor("#0f0f13")
    ax1.bar(x, scores_df["fusion_score"], width=widths * 0.82, color="#e05252", alpha=0.9, zorder=3, align='edge')
    ax1.set_ylabel("Significance Score", color="#aaaaaa", fontsize=9)
    ax1.set_title("Segment Analysis", color="#eeeeee", fontsize=10, pad=8)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors="#666666", labelsize=8)
    ax1.spines[:].set_color("#222233")
    ax1.grid(axis="y", color="#1e1e2e", linewidth=0.5, zorder=0)

    ax2 = axes[1]
    ax2.set_facecolor("#0f0f13")
    ax2.plot(x, scores_df["cnn_score"],  color="#4fc3f7", linewidth=1.4, label="CNN — Visual", alpha=0.9)
    ax2.plot(x, scores_df["lstm_score"], color="#81c784", linewidth=1.4, label="LSTM — Audio", alpha=0.9)
    ax2.plot(x, scores_df["bert_score"], color="#ce93d8", linewidth=1.4, label="BERT — Text",  alpha=0.9)
    ax2.set_ylabel("Model Score", color="#aaaaaa", fontsize=9)
    ax2.set_xlabel("Time (seconds)", color="#aaaaaa", fontsize=9)
    ax2.set_title("Individual Model Contributions", color="#eeeeee", fontsize=10, pad=8)
    ax2.legend(facecolor="#1a1a22", edgecolor="#333", labelcolor="#aaaaaa", fontsize=8, ncol=3)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors="#666666", labelsize=8)
    ax2.spines[:].set_color("#222233")
    ax2.grid(axis="y", color="#1e1e2e", linewidth=0.5, zorder=0)

    plt.tight_layout(pad=2.0)
    chart_path = os.path.join(work_dir, "scores_chart.png")
    plt.savefig(chart_path, dpi=130, facecolor="#0f0f13")
    plt.close()
    return chart_path


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_video(video_file, top_n_clips, use_cnn, use_lstm, use_bert, show_labels, mode="highlight"):
    if video_file is None:
        return None, None, "⚠ Please upload a video first.", None, ""

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    top_n      = int(top_n_clips)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    work_dir   = tempfile.mkdtemp(prefix="evs_")

    orig_dur = get_video_duration(video_file)

    # ── Demo mode: check if pre-baked results exist ───────────────────────────
    prebaked_scores    = os.path.join(FUSION_DIR,     f"{video_name}_{mode}_scores.json")
    prebaked_highlight = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_{mode}.mp4")

    if os.path.exists(prebaked_scores) and os.path.exists(prebaked_highlight):
        log(f"⚡ Pre-baked results found for '{video_name}' — loading instantly!")
        
        with open(prebaked_scores) as f:
            scores_df = pd.DataFrame(json.load(f))

        if mode == "lecture":
            import config as cfg
            selected_df = select_lecture_chunks(scores_df, compression_ratio=cfg.LECTURE_COMPRESSION_RATIO)
        else:
            selected_df = select_segments(scores_df, top_n=top_n, min_gap=MIN_SEGMENT_GAP)
            
        highlight_dur = selected_df["end_sec"].sub(selected_df["start_sec"]).sum()

        display_df = selected_df[["segment_idx", "start_sec", "end_sec", "fusion_score"]].copy()
        display_df.columns = ["Clip #", "Start (s)", "End (s)", "Significance Score"]
        display_df["Significance Score"] = display_df["Significance Score"].round(3)
        display_df["Clip #"] = display_df["Clip #"] + 1

        chart_path = build_chart(scores_df, work_dir)
        stats_html = build_stats_html(orig_dur, highlight_dur, len(selected_df))

        log(f"  Original: {fmt_time(orig_dur)} → Output: {fmt_time(highlight_dur)}")
        log(f"✓ Loaded instantly from pre-baked results.")

        return prebaked_highlight, chart_path, "\n".join(logs), display_df, stats_html

    # ── Full pipeline (no pre-baked results) ──────────────────────────────────
    log(f"▶ No pre-baked results found. Running full pipeline in {mode.upper()} mode...")
    log(f"  Source: {video_name} ({fmt_time(orig_dur)})")

    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        log("  [1/4] Extracting audio...")
        audio_path = extract_audio(video_file, audio_dir)
        if audio_path is None:
            log("  WARNING: No audio track found.")

        log(f"  [2/4] Running Whisper transcription...")
        transcript_df = None
        if audio_path:
            try:
                from preprocess import transcribe_audio
                _, segments_csv, transcript_df = transcribe_audio(
                    audio_path, TRANSCRIPTS_DIR, model_size="base"
                )
                log(f"         Whisper done — {len(transcript_df)} segments")
            except Exception as e:
                log(f"         Whisper failed: {e} — BERT will use uniform scores.")

        log(f"  [3/4] Running multi-modal fusion...")
        import config as cfg
        orig_audio_dir = cfg.AUDIO_DIR
        cfg.AUDIO_DIR  = audio_dir

        scores_df, json_path = run_fusion(
            video_file,
            CNN_MODEL  if use_cnn  else None,
            LSTM_MODEL if use_lstm else None,
            BERT_TOKENIZER if (use_bert and BERT_MODEL is not None) else None,
            BERT_MODEL     if (use_bert and BERT_MODEL is not None) else None,
            mode=mode
        )

        cfg.AUDIO_DIR = orig_audio_dir
        log(f"         Fusion done — {len(scores_df)} segments scored.")

        if mode == "lecture":
            selected_df = select_lecture_chunks(scores_df, compression_ratio=cfg.LECTURE_COMPRESSION_RATIO)
        else:
            selected_df = select_segments(scores_df, top_n=top_n, min_gap=MIN_SEGMENT_GAP)

        log(f"  [4/4] Building final video output...")
        output_path = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_{mode}.mp4")
        os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)
        result = build_highlight_reel(video_file, selected_df, output_path,
                                      add_labels=show_labels)

        if result is None:
            return None, None, "✗ Failed to build video.", None, ""

        highlight_dur = selected_df["end_sec"].sub(selected_df["start_sec"]).sum()

        display_df = selected_df[["segment_idx", "start_sec", "end_sec", "fusion_score"]].copy()
        display_df.columns = ["Clip #", "Start (s)", "End (s)", "Significance Score"]
        display_df["Significance Score"] = display_df["Significance Score"].round(3)
        display_df["Clip #"] = display_df["Clip #"] + 1

        chart_path = build_chart(scores_df, work_dir)
        stats_html = build_stats_html(orig_dur, highlight_dur, len(selected_df))

        log(f"  Original: {fmt_time(orig_dur)} → Output: {fmt_time(highlight_dur)}")
        log("✓ Done!")

        return output_path, chart_path, "\n".join(logs), display_df, stats_html

    except Exception as e:
        import traceback
        log(f"✗ ERROR: {e}\n{traceback.format_exc()}")
        return None, None, "\n".join(logs), None, ""

# Wrapper functions for the Gradio buttons
def run_highlight_mode(video, top_n, cnn, lstm, bert, labels):
    return process_video(video, top_n, cnn, lstm, bert, labels, mode="highlight")

def run_lecture_mode(video, cnn, lstm, bert, labels):
    return process_video(video, 0, cnn, lstm, bert, labels, mode="lecture")


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0b0b10;
    --surface:   #13131a;
    --surface2:  #1a1a24;
    --border:    #25253a;
    --accent:    #e05252;
    --accent2:   #4fc3f7;
    --text:      #e8e8f0;
    --muted:     #7070a0;
    --success:   #81c784;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --radius:    12px;
    --radius-lg: 20px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}
footer { display: none !important; }

.hero-wrap {
    text-align: center;
    padding: 48px 24px 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(224,82,82,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    font-family: var(--font-body);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid rgba(224,82,82,0.35);
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 18px;
}
.hero-title {
    font-family: var(--font-head) !important;
    font-size: clamp(28px, 5vw, 52px) !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px !important;
    line-height: 1.1 !important;
    color: var(--text) !important;
    margin: 0 0 16px !important;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-family: var(--font-body);
    font-size: 15px;
    color: var(--muted);
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.7;
    font-weight: 300;
}
.model-pills {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 20px 0 0;
}
.pill {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 5px 12px;
    border-radius: 100px;
    border: 1px solid;
}
.pill-cnn  { color: #4fc3f7; border-color: rgba(79,195,247,0.3);  background: rgba(79,195,247,0.06); }
.pill-lstm { color: #81c784; border-color: rgba(129,199,132,0.3); background: rgba(129,199,132,0.06); }
.pill-bert { color: #ce93d8; border-color: rgba(206,147,216,0.3); background: rgba(206,147,216,0.06); }

.divider { height: 1px; background: var(--border); margin: 20px 0; }

.section-label {
    font-family: var(--font-head);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

.stats-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 80px;
    text-align: center;
    padding: 10px 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
}
.stat-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.stat-value {
    font-family: var(--font-head);
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
}
.highlight-card  { border-color: rgba(224,82,82,0.4); }
.highlight-card .stat-value  { color: var(--accent); }
.compression-card .stat-value { color: var(--success); }
.stat-arrow { font-size: 20px; color: var(--muted); flex: 0 0 auto; }

label {
    font-family: var(--font-body) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

textarea, input[type=text] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}

.gr-dataframe table { background: var(--surface2) !important; border-collapse: collapse !important; }
.gr-dataframe th {
    background: var(--surface) !important;
    color: var(--muted) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}
.gr-dataframe td {
    color: var(--text) !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}

#run-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: white !important;
    padding: 14px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(224,82,82,0.25) !important;
}
#run-btn:hover {
    background: #c94444 !important;
    box-shadow: 0 6px 32px rgba(224,82,82,0.4) !important;
    transform: translateY(-1px) !important;
}

.tips-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: var(--radius);
    padding: 14px 18px;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.7;
    margin-top: 8px;
}
.tips-box strong { color: var(--text); }
"""


# ── Build pre-baked video list for display ────────────────────────────────────

def get_prebaked_list():
    if not os.path.exists(FUSION_DIR):
        return ""
    videos = []
    for f in os.listdir(FUSION_DIR):
        if f.endswith("_scores.json"):
            name_parts = f.replace("_scores.json", "").split("_")
            name = "_".join(name_parts[:-1]) # get video name
            mode = name_parts[-1] # get mode
            
            highlight = os.path.join(HIGHLIGHTS_DIR, f"{name}_{mode}.mp4")
            status = "⚡ ready" if os.path.exists(highlight) else "scores only"
            videos.append(f"<li><strong>{name}</strong> <em>({mode})</em> — {status}</li>")
    if not videos:
        return ""
    items = "\n".join(videos)
    return f"""
    <div class="tips-box" style="margin-top:12px;">
        <strong>⚡ Pre-baked videos (instant load):</strong>
        <ul style="margin:6px 0 0; padding-left:18px; line-height:2;">{items}</ul>
        Upload any of these videos to get instant results.
    </div>
    """


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui():
    prebaked_html = get_prebaked_list()

    with gr.Blocks(css=CSS, title="VideoSum AI") as demo:

        gr.HTML("""
        <div class="hero-wrap">
            <div class="hero-tag">Deep Learning · Multi-Modal AI</div>
            <h1 class="hero-title">Video<span>Sum</span> AI</h1>
            <p class="hero-sub">
                Upload any video. The system analyzes visual, audio, and textual signals
                simultaneously to extract the most significant moments — automatically.
            </p>
            <div class="model-pills">
                <span class="pill pill-cnn">CNN · Facial Features</span>
                <span class="pill pill-lstm">LSTM · Speech Prosody</span>
                <span class="pill pill-bert">BERT · Transcript Meaning</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left: controls ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<div class="section-label">Input Video</div>')
                video_input = gr.Video(label="Upload Video", height=220)

                if prebaked_html:
                    gr.HTML(prebaked_html)

                gr.HTML('<div class="divider"></div>')
                gr.HTML('<div class="section-label">Active Models</div>')

                with gr.Row():
                    use_cnn  = gr.Checkbox(value=True,  label="CNN")
                    use_lstm = gr.Checkbox(value=True,  label="LSTM")
                    use_bert = gr.Checkbox(
                        value=BERT_MODEL is not None,
                        interactive=BERT_MODEL is not None,
                        label="BERT"
                    )

                show_labels = gr.Checkbox(value=True, label="Score overlays on clips")

                gr.HTML('<div class="divider"></div>')
                
                # ── The New Dual-Tab Interface ──
                with gr.Tabs():
                    # Tab 1: Short Highlights
                    with gr.TabItem("🎬 Short Highlights"):
                        top_n_slider = gr.Slider(
                            minimum=3, maximum=10, value=5, step=1,
                            label="Highlight clips to generate",
                        )
                        run_btn_highlight = gr.Button(
                            "Generate Highlight Reel →",
                            variant="primary",
                            elem_id="run-btn"
                        )
                        gr.HTML("""
                        <div class="tips-box" style="margin-top:15px;">
                            <strong>Best for:</strong> Finding explosive, high-emotion reactions 
                            like a sudden laugh, shout, or gasp in short content.
                        </div>
                        """)

                    # Tab 2: Long-Form Lecture Summary
                    with gr.TabItem("📚 Lecture Summary"):
                        gr.HTML(f"""
                        <div class="tips-box" style="margin-bottom:15px; margin-top:5px;">
                            <strong>Best for:</strong> Podcasts, Lectures, and Long Interviews. <br><br>
                            This engine reads complete sentences and chronologically stitches together the top 
                            <strong>{int(LECTURE_COMPRESSION_RATIO*100)}%</strong> most valuable conversational blocks.
                        </div>
                        """)
                        run_btn_lecture = gr.Button(
                            "Generate Lecture Digest →",
                            variant="primary",
                            elem_id="run-btn"
                        )

            # ── Right: outputs ────────────────────────────────────────────────
            with gr.Column(scale=2):

                gr.HTML('<div class="section-label">Summary</div>')
                stats_output = gr.HTML(
                    value='<div class="stats-row" style="justify-content:center;'
                          'color:#7070a0;font-size:13px;padding:20px;">'
                          'Upload a video and click Generate to see results</div>'
                )

                gr.HTML('<div class="section-label">Final Video Output</div>')
                video_output = gr.Video(label="", height=240)

                gr.HTML('<div class="section-label">Significance Analysis</div>')
                chart_output = gr.Image(label="", type="filepath")

                gr.HTML('<div class="section-label">Selected Segments</div>')
                table_output = gr.Dataframe(
                    headers=["Clip #", "Start (s)", "End (s)", "Significance Score"],
                    label="",
                    interactive=False,
                    wrap=True,
                )

                gr.HTML('<div class="section-label">Pipeline Log</div>')
                log_output = gr.Textbox(
                    label="",
                    lines=5,
                    max_lines=12,
                    interactive=False,
                    placeholder="Logs will appear here...",
                )

        # ── Button Triggers ──
        run_btn_highlight.click(
            fn=run_highlight_mode,
            inputs=[video_input, top_n_slider, use_cnn, use_lstm, use_bert, show_labels],
            outputs=[video_output, chart_output, log_output, table_output, stats_output],
        )
        
        run_btn_lecture.click(
            fn=run_lecture_mode,
            inputs=[video_input, use_cnn, use_lstm, use_bert, show_labels],
            outputs=[video_output, chart_output, log_output, table_output, stats_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )