import os

import gradio as gr
import plotly.graph_objects as go

from pipeline_utils import PipelineRunner
import warnings
warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CHECKPOINT_PATH = "./ckpts/vjepa-ori-ft20.pt"
CHECKPOINT_PATH = "./ckpts/vjepa_full/best_vjepa_model9639(paper).pt"
ASSETS_DIR = "./assets"
PORT = 9530

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = PipelineRunner(CHECKPOINT_PATH)
    return pipeline


# ---------------------------------------------------------------------------
# Flowchart renderer
# ---------------------------------------------------------------------------

FLOWCHART_CSS = """
<style>
.pipeline-container {
    display: flex; align-items: center; justify-content: center;
    gap: 0; padding: 24px 12px; flex-wrap: wrap;
}
.stage-box {
    display: flex; flex-direction: column; align-items: center;
    padding: 14px 22px; border: 2.5px solid #d1d5db; border-radius: 14px;
    text-align: center; min-width: 155px; background: #fafafa;
    transition: all 0.45s ease; position: relative;
}
.stage-box .stage-icon { font-size: 26px; margin-bottom: 4px; }
.stage-box .stage-title { font-weight: 700; font-size: 14px; color: #374151; }
.stage-box .stage-model { font-size: 11px; color: #6b7280; margin-top: 2px; }
.stage-box .stage-status { font-size: 12px; font-weight: 600; margin-top: 6px; }

.stage-box.pending { opacity: 0.45; border-color: #d1d5db; }
.stage-box.pending .stage-status { color: #9ca3af; }

.stage-box.running { border-color: #3b82f6; background: #eff6ff;
    animation: stagePulse 1.6s ease-in-out infinite; }
.stage-box.running .stage-status { color: #2563eb; }

.stage-box.pass { border-color: #22c55e; background: #f0fdf4; }
.stage-box.pass .stage-status { color: #16a34a; }

.stage-box.stop { border-color: #ef4444; background: #fef2f2; }
.stage-box.stop .stage-status { color: #dc2626; }

.stage-box.complete { border-color: #22c55e; background: #f0fdf4;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.18); }
.stage-box.complete .stage-status { color: #16a34a; }

.arrow-connector {
    display: flex; align-items: center; font-size: 28px;
    color: #d1d5db; margin: 0 6px; transition: color 0.4s ease;
    user-select: none;
}
.arrow-connector.active { color: #22c55e; }
.arrow-connector.rejected { color: #ef4444; }

@keyframes stagePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.35); }
    50% { box-shadow: 0 0 18px 6px rgba(59,130,246,0.15); }
}
</style>
"""

STAGE_INFO = {
    "stage1": {"icon": "\U0001f50d", "title": "Edge QC Model", "model": ""},
    "stage2": {"icon": "\U0001f9ec", "title": "Cloud Risk Grading", "model": ""},
}

STATUS_LABELS = {
    "pending": "Waiting...",
    "running": "Processing...",
    "pass": "✅ Passed",
    "stop": "❌ Stopped",
    "complete": "✅ Complete",
}


def render_pipeline_flowchart(stage_results):
    stages = ["stage1", "stage2"]
    boxes = []
    arrows = []

    for i, key in enumerate(stages):
        state = stage_results.get(key, "pending")
        info = STAGE_INFO[key]
        label = STATUS_LABELS.get(state, "")
        boxes.append(
            f'<div class="stage-box {state}">'
            f'<div class="stage-icon">{info["icon"]}</div>'
            f'<div class="stage-title">{info["title"]}</div>'
            f'<div class="stage-status">{label}</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            next_key = stages[i + 1]
            cur = stage_results.get(key, "pending")
            nxt = stage_results.get(next_key, "pending")
            if cur in ("pass", "complete") and nxt != "pending":
                cls = "active"
            elif cur == "stop":
                cls = "rejected"
            else:
                cls = ""
            arrows.append(f'<div class="arrow-connector {cls}">➡</div>')

    parts = []
    for i, box in enumerate(boxes):
        parts.append(box)
        if i < len(arrows):
            parts.append(arrows[i])
    return FLOWCHART_CSS + '<div class="pipeline-container">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Low Risk\n(Class 0)", "Medium Risk\n(Class 1)", "High Risk\n(Class 2)"]
CLASS_COLORS = ["#3b82f6", "#f59e0b", "#ef4444"]


def create_bar_chart(prediction):
    probs = prediction["probs"]
    pred_class = prediction["predicted_class"]
    colors = [CLASS_COLORS[i] if i != pred_class else "#22c55e" for i in range(3)]

    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=[probs["class_0"], probs["class_1"], probs["class_2"]],
            marker_color=colors,
            text=[f"{probs[f'class_{i}']:.1%}" for i in range(3)],
            textposition="auto",
            textfont=dict(size=13, color="white"),
        )
    ])
    fig.update_layout(
        title=dict(text="Top Clip: 3-Class Probabilities", font=dict(size=15)),
        yaxis=dict(title="Probability", range=[0, 1], gridcolor="#e5e7eb"),
        xaxis=dict(gridcolor="#e5e7eb"),
        showlegend=False,
        height=280,
        margin=dict(t=40, b=30, l=50, r=20),
        plot_bgcolor="white",
    )
    return fig


def create_per_clip_bar_chart(clip_results):
    """Grouped bar chart: one group per clip, predicted class highlighted green."""
    if not clip_results:
        return go.Figure()
    n = len(clip_results)
    fig = go.Figure()
    for i, cr in enumerate(clip_results):
        pred = cr["prediction"]
        colors = [
            CLASS_COLORS[c] if c != pred["predicted_class"] else "#22c55e"
            for c in range(3)
        ]
        for c in range(3):
            fig.add_trace(go.Bar(
                x=[f"Clip {i + 1}"],
                y=[pred["probs"][f"class_{c}"]],
                marker_color=colors[c],
                text=[f"{pred['probs'][f'class_{c}']:.1%}"],
                textposition="inside",
                textfont=dict(size=11, color="white"),
                showlegend=False,
            ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Per-Clip Classification", font=dict(size=14)),
        yaxis=dict(title="Probability", range=[0, 1], gridcolor="#e5e7eb"),
        xaxis=dict(gridcolor="#e5e7eb"),
        height=240,
        margin=dict(t=36, b=30, l=50, r=20),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Detail formatters
# ---------------------------------------------------------------------------

def format_stage1_detail(result):
    cls = result["classification"]
    lines = [
        f"**Result:** {'Lesions detected' if cls['has_lesions'] else 'No lesions'}",
        f"**Confidence:** {cls['confidence']:.1%}",
    ]
    if result.get("detection") and result["detection"].get("boxes_found"):
        det = result["detection"]
        lines.append(
            f"**Boxes found:** {det['num_boxes']} across "
            f"{det.get('frames_with_boxes', '?')} frame(s)")
    if result.get("clip_regions"):
        lines.append(f"**Clips detected:** {len(result['clip_regions'])}")
        for cr in result["clip_regions"]:
            lines.append(
                f"  - `{cr['label']}`: frames "
                f"{cr['start_frame'] + 1}–{cr['end_frame'] + 1}")
    return "\n".join(lines)


def format_stage2_detail(result):
    pred = result["prediction"]
    lines = [
        f"**Predicted class:** {pred['predicted_class']}",
        f"**Confidence:** {pred['confidence']:.1%}",
        "",
    ]
    for c in range(3):
        marker = " ⬅" if c == pred["predicted_class"] else ""
        lines.append(f"- Class {c}: {pred['probs'][f'class_{c}']:.1%}{marker}")
    return "\n".join(lines)


def format_clip_summary(clip_results, top_idx):
    if not clip_results:
        return ""
    lines = ["### Clip Results\n"]
    for i, cr in enumerate(clip_results):
        pred = cr["prediction"]
        star = " ⬅ top risk" if i == top_idx else ""
        lines.append(
            f"- **Clip {i + 1}**: Class {pred['predicted_class']} "
            f"({pred['confidence']:.1%}){star}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline callback
# ---------------------------------------------------------------------------

def _empty_outputs(clips):
    """Return a dict that resets / hides all 16 output components."""
    base = {
        flowchart: render_pipeline_flowchart({}),
        s1_status: "",
        s1_detail: "",
        s1_montage_image: gr.update(visible=False, value=None),
        s2_status: "",
        s2_detail: "",
        s2_clip_summary: "",
        s2_clip_bar_chart: gr.update(visible=False, value=None),
        s2_bar_chart: gr.update(visible=False, value=None),
        s2_gradcam_video: gr.update(visible=False, value=None),
        s2_gradcam_summary: gr.update(visible=False, value=None),
    }
    for slot in clips:
        base[slot] = gr.update(visible=False, value=None)
    return base


def _set_clip_outputs(clip_paths, outputs, clips):
    """Populate the 5 clip Video slots based on clip_paths length."""
    n = len(clip_paths) if clip_paths else 0
    for i, slot in enumerate(clips):
        outputs[slot] = gr.update(
            visible=i < n, value=clip_paths[i] if i < n else None)


def run_pipeline(video_path):
    clips = [s1_clip_0, s1_clip_1, s1_clip_2, s1_clip_3, s1_clip_4]

    if video_path is None:
        yield {
            **_empty_outputs(clips),
            s1_status: "Please upload a video first.",
        }
        return

    pipe = get_pipeline()
    stage_state = {}

    # Reset
    yield _empty_outputs(clips)

    # === Stage 1: Edge QC Model ===
    stage_state["stage1"] = "running"
    yield {
        **_empty_outputs(clips),
        flowchart: render_pipeline_flowchart(stage_state),
        s1_status: "⏳ Running non-valid detection + lesion detection...",
    }

    s1 = pipe.run_stage1(video_path)
    stage_state["stage1"] = s1["status"]

    if s1["status"] == "stop":
        stage_state["stage2"] = "stop"
        out = {
            **_empty_outputs(clips),
            flowchart: render_pipeline_flowchart(stage_state),
            s1_status: f"❌ STOPPED — {s1['message']}",
            s1_detail: format_stage1_detail(s1),
        }
        if s1.get("montage_image") is not None:
            out[s1_montage_image] = gr.update(visible=True, value=s1["montage_image"])
        yield out
        return

    # Stage 1 passed
    s1_pass = {
        **_empty_outputs(clips),
        flowchart: render_pipeline_flowchart(stage_state),
        s1_status: f"✅ PASSED — {s1['message']}",
        s1_detail: format_stage1_detail(s1),
        s1_montage_image: gr.update(visible=True, value=s1["montage_image"]),
    }
    _set_clip_outputs(s1.get("clip_video_paths"), s1_pass, clips)
    yield s1_pass

    # === Stage 2: Cloud Risk Grading Foundation Model ===
    stage_state["stage2"] = "running"
    s2_running = {
        **_empty_outputs(clips),
        flowchart: render_pipeline_flowchart(stage_state),
        # Preserve Stage 1 outputs
        s1_status: s1_pass[s1_status],
        s1_detail: s1_pass[s1_detail],
        s1_montage_image: s1_pass[s1_montage_image],
        s1_clip_0: s1_pass[s1_clip_0],
        s1_clip_1: s1_pass[s1_clip_1],
        s1_clip_2: s1_pass[s1_clip_2],
        s1_clip_3: s1_pass[s1_clip_3],
        s1_clip_4: s1_pass[s1_clip_4],
        s2_status: "⏳ Running risk grading + Grad-CAM...",
    }
    yield s2_running

    clip_paths = s1.get("clip_video_paths") or []
    s2 = pipe.run_stage3(video_path, clip_paths=clip_paths if clip_paths else None)
    stage_state["stage2"] = s2["status"]

    clip_results = s2.get("clip_results")
    has_clips = clip_results is not None and len(clip_results) > 0

    yield {
        **_empty_outputs(clips),
        flowchart: render_pipeline_flowchart(stage_state),
        # Preserve Stage 1 outputs
        s1_status: s1_pass[s1_status],
        s1_detail: s1_pass[s1_detail],
        s1_montage_image: s1_pass[s1_montage_image],
        s1_clip_0: s1_pass[s1_clip_0],
        s1_clip_1: s1_pass[s1_clip_1],
        s1_clip_2: s1_pass[s1_clip_2],
        s1_clip_3: s1_pass[s1_clip_3],
        s1_clip_4: s1_pass[s1_clip_4],
        # Stage 2 results
        s2_status: f"✅ COMPLETE — Predicted: Class {s2['prediction']['predicted_class']}",
        s2_detail: format_stage2_detail(s2),
        s2_clip_summary: format_clip_summary(clip_results, s2.get("top_clip_index", -1)),
        s2_clip_bar_chart: gr.update(
            visible=has_clips,
            value=create_per_clip_bar_chart(clip_results) if has_clips else None),
        s2_bar_chart: gr.update(visible=True, value=create_bar_chart(s2["prediction"])),
        s2_gradcam_video: gr.update(visible=True, value=s2["gradcam_compare_path"]),
        s2_gradcam_summary: gr.update(visible=True, value=s2["gradcam_summary"]),
    }


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#main-title { text-align: center; margin-bottom: 4px; }
#main-subtitle { text-align: center; color: #6b7280; margin-bottom: 16px; font-size: 14px; }
video { max-width: 100%; max-height: 360px; object-fit: contain; }
"""

with gr.Blocks(title="Medical Video Classification Pipeline") as demo:
    gr.Markdown("# Medical Video Classification Pipeline", elem_id="main-title")
    gr.Markdown(
        "2-stage cascaded inference: Edge QC Model → Cloud Risk Grading Foundation Model",
        elem_id="main-subtitle",
    )

    flowchart = gr.HTML(value=render_pipeline_flowchart({}))

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Video", sources=["upload"])
        with gr.Column(scale=1):
            gr.Markdown("### Example Videos")
            example_videos = {}
            for cls_label, cls_name in [("Class 0 — Low Risk", 0), ("Class 1 — Medium Risk", 1), ("Class 2 — High Risk", 2), ("Class NO — No Lesion", "NO")]:
                gr.Markdown(f"**{cls_label}**")
                with gr.Row():
                    for i in range(5):
                        fname = f"CLASS{cls_name}_{i}.mp4"
                        fpath = os.path.join(ASSETS_DIR, fname)
                        btn = gr.Button(fname, size="sm", variant="secondary")
                        example_videos[btn] = fpath

            for btn, fpath in example_videos.items():
                btn.click(fn=lambda p=fpath: p, inputs=[], outputs=[input_video])

    run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

    with gr.Row():
        # ========== Stage 1 ==========
        with gr.Column(scale=1):
            gr.Markdown("### Stage 1: Edge QC Model")
            s1_status = gr.Markdown("")
            s1_detail = gr.Markdown("")
            s1_montage_image = gr.Image(label="16-Frame Lesion Detection Montage", visible=False)

            gr.Markdown("#### Detected Lesion Clips")
            with gr.Row():
                s1_clip_0 = gr.Video(label="Clip 1", visible=False)
                s1_clip_1 = gr.Video(label="Clip 2", visible=False)
            with gr.Row():
                s1_clip_2 = gr.Video(label="Clip 3", visible=False)
                s1_clip_3 = gr.Video(label="Clip 4", visible=False)
            s1_clip_4 = gr.Video(label="Clip 5", visible=False)

        # ========== Stage 2 ==========
        with gr.Column(scale=1):
            gr.Markdown("### Stage 2: Cloud Risk Grading")
            s2_status = gr.Markdown("")
            s2_detail = gr.Markdown("")
            s2_clip_summary = gr.Markdown("")
            s2_clip_bar_chart = gr.Plot(label="Per-Clip Probabilities", visible=False)
            s2_bar_chart = gr.Plot(label="Top Clip: 3-Class Probabilities", visible=False)
            s2_gradcam_video = gr.Video(label="Grad-CAM Overlay", visible=False)
            s2_gradcam_summary = gr.Image(label="Grad-CAM Summary", visible=False)

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_video],
        outputs=[
            flowchart,
            s1_status, s1_detail, s1_montage_image,
            s1_clip_0, s1_clip_1, s1_clip_2, s1_clip_3, s1_clip_4,
            s2_status, s2_detail, s2_clip_summary, s2_clip_bar_chart,
            s2_bar_chart, s2_gradcam_video, s2_gradcam_summary,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        max_file_size=100 * 1024 * 1024,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
        css=CUSTOM_CSS,
    )
