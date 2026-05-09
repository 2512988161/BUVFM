import os
import warnings

import gradio as gr
import plotly.graph_objects as go

from pipeline_utils import PipelineRunner

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
# Bar charts
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Class 0\n(Low Risk / No Lesion)", "Medium Risk\n(Class 1)", "High Risk\n(Class 2)"]
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
        title=dict(text="3-Class Probabilities", font=dict(size=15)),
        yaxis=dict(title="Probability", range=[0, 1], gridcolor="#e5e7eb"),
        xaxis=dict(gridcolor="#e5e7eb"),
        showlegend=False,
        height=300,
        margin=dict(t=40, b=30, l=50, r=20),
        plot_bgcolor="white",
    )
    return fig


def create_stage1_frame_chart(result):
    frame_results = result["frame_results"]
    x = [item["frame_idx"] for item in frame_results]
    yolo_scores = [item["yolo_score"] for item in frame_results]
    valid_scores = [1.0 - item["mobilenet_score"] for item in frame_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=valid_scores,
        mode="lines",
        name="Valid",
        line=dict(color="#2563eb", width=2.5),
        hovertemplate="Frame %{x}<br>Valid %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=yolo_scores,
        mode="lines",
        name="YOLO confidence",
        line=dict(color="#10b981", width=2.5),
        hovertemplate="Frame %{x}<br>YOLO %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Frame-wise Stage 1 Scores", font=dict(size=15)),
        height=230,
        margin=dict(t=40, b=30, l=45, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(title="Frame", gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(title="Score", range=[-0.05, 1.05], gridcolor="#e5e7eb", zeroline=False),
        hovermode="x unified",
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=0.5,
                y1=0.5,
                line=dict(color="rgba(100, 116, 139, 0.35)", width=1, dash="dash"),
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Detail formatters
# ---------------------------------------------------------------------------


def format_stage1_detail(result):
    cls = result["classification"]
    det = result["detection"]
    lines = [
        f"**Result:** {'Lesions detected' if cls['has_lesions'] else 'No lesions detected'}",
        f"**Mean valid-frame score:** {cls['confidence']:.1%}",
        f"**Valid frames:** {cls['valid_frames']} / {cls['total_frames']}",
        f"**Detected frames:** {det['frames_with_boxes']}",
        f"**Total boxes:** {det['num_boxes']}",
        f"**Max YOLO confidence:** {det['max_confidence']:.3f}",
    ]
    return "\n".join(lines)



def format_stage2_detail(result):
    pred = result["prediction"]
    predicted_label = "Class 0 (Low Risk / No Lesion)" if pred["predicted_class"] == 0 else f"Class {pred['predicted_class']}"
    lines = [
        f"**Predicted class:** {predicted_label}",
        f"**Confidence:** {pred['confidence']:.1%}",
        "",
    ]
    for c in range(3):
        label = "Class 0 (Low Risk / No Lesion)" if c == 0 else f"Class {c}"
        marker = " ⬅" if c == pred["predicted_class"] else ""
        lines.append(f"- {label}: {pred['probs'][f'class_{c}']:.1%}{marker}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline callback
# ---------------------------------------------------------------------------


def format_status_badge(text):
    if text.startswith("✅"):
        status_class = "status-success"
    elif text.startswith("❌"):
        status_class = "status-error"
    elif text.startswith("⏳"):
        status_class = "status-running"
    else:
        status_class = "status-idle"
    return f'<div class="status-badge {status_class}">{text}</div>'


def _empty_outputs():
    return {
        s1_panel: gr.update(visible=True),
        s1_status: format_status_badge("Waiting to run stage 1."),
        s1_detail: "Upload a video and click **Run Pipeline** to generate the annotated screening result.",
        s1_video: gr.update(visible=False, value=None),
        s1_frame_chart: gr.update(visible=False, value=None),
        s2_panel: gr.update(visible=True),
        s2_status: format_status_badge("Waiting for stage 1."),
        s2_detail: "Stage 2 risk grading and Grad-CAM results will appear here after stage 1 passes.",
        s2_bar_chart: gr.update(visible=False, value=None),
        s2_gradcam_video: gr.update(visible=False, value=None),
    }



def run_pipeline(video_path):
    if video_path is None:
        yield {
            **_empty_outputs(),
            s1_panel: gr.update(visible=True),
            s1_status: format_status_badge("Please upload a video first."),
        }
        return

    pipe = get_pipeline()

    yield _empty_outputs()

    yield {
        **_empty_outputs(),
        s1_panel: gr.update(visible=True),
        s1_status: format_status_badge("⏳ Running stage 1 screening with MobileNet + YOLO..."),
    }

    s1 = pipe.run_stage1(video_path)
    s1_outputs = {
        **_empty_outputs(),
        s1_panel: gr.update(visible=True),
        s1_status: format_status_badge(f"{'✅ PASSED' if s1['status'] == 'pass' else '❌ STOPPED'} — {s1['message']}"),
        s1_detail: format_stage1_detail(s1),
        s1_video: gr.update(visible=True, value=s1["annotated_video_path"]),
        s1_frame_chart: gr.update(visible=True, value=create_stage1_frame_chart(s1)),
    }
    yield s1_outputs

    if s1["status"] == "stop":
        return

    yield {
        **s1_outputs,
        s2_panel: gr.update(visible=True),
        s2_status: format_status_badge("⏳ Running stage 2 risk grading + Grad-CAM..."),
    }

    s2 = pipe.run_stage3(video_path)

    yield {
        **s1_outputs,
        s2_panel: gr.update(visible=True),
        s2_status: format_status_badge("✅ COMPLETE — Predicted: " + (
            "Class 0 (Low Risk / No Lesion)"
            if s2["prediction"]["predicted_class"] == 0
            else f"Class {s2['prediction']['predicted_class']}"
        )),
        s2_detail: format_stage2_detail(s2),
        s2_bar_chart: gr.update(visible=True, value=create_bar_chart(s2["prediction"])),
        s2_gradcam_video: gr.update(visible=True, value=s2["gradcam_compare_path"]),
    }


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#main-title { text-align: center; margin-bottom: 18px; }
video { max-width: 100%; max-height: 420px; object-fit: contain; }
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
}
.contain {
    max-width: 100% !important;
}
.result-card {
    width: 100%;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    margin-top: 16px;
    background: #ffffff;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
}
.status-badge {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 10px 0 14px 0;
    font-size: 18px;
    font-weight: 700;
    line-height: 1.45;
    border: 1px solid transparent;
}
.status-idle {
    background: #f8fafc;
    color: #334155;
    border-color: #cbd5e1;
}
.status-running {
    background: #eff6ff;
    color: #1d4ed8;
    border-color: #93c5fd;
}
.status-success {
    background: #f0fdf4;
    color: #15803d;
    border-color: #86efac;
}
.status-error {
    background: #fef2f2;
    color: #dc2626;
    border-color: #fca5a5;
}
"""

with gr.Blocks(
    title="Large-scale real-world validation of an AI-driven ultrasound breast cancer screening system in rural settings"
) as demo:
    gr.Markdown(
        "# Large-scale real-world validation of an AI-driven ultrasound breast cancer screening system in rural settings",
        elem_id="main-title",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Video", sources=["upload"], autoplay=True, loop=True)
        with gr.Column(scale=1):
            gr.Markdown("### Example Videos")
            example_videos = {}
            for cls_label, cls_name in [
                ("Class 0 — Low Risk", 0),
                ("Class 0 — No Lesion", "NO"),
                ("Class 1 — Medium Risk", 1),
                ("Class 2 — High Risk", 2),
            ]:
                gr.Markdown(f"**{cls_label}**")
                with gr.Row():
                    for i in range(4):
                        fname = f"CLASS{cls_name}_{i}.mp4"
                        fpath = os.path.join(ASSETS_DIR, fname)
                        btn = gr.Button(fname, size="sm", variant="secondary")
                        example_videos[btn] = fpath

            for btn, fpath in example_videos.items():
                btn.click(fn=lambda p=fpath: p, inputs=[], outputs=[input_video])

    run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

    with gr.Column(visible=True, elem_classes=["result-card"]) as s1_panel:
        gr.Markdown("### Stage 1 Screening")
        s1_status = gr.Markdown("")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                s1_detail = gr.Markdown("")
                s1_frame_chart = gr.Plot(label="Frame-wise Stage 1 Scores", visible=False)
            with gr.Column(scale=1):
                s1_video = gr.Video(label="Stage 1 Annotated Video", visible=False, autoplay=True, loop=True)

    with gr.Column(visible=True, elem_classes=["result-card"]) as s2_panel:
        gr.Markdown("### Stage 2 Risk Grading")
        s2_status = gr.Markdown("")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                s2_detail = gr.Markdown("")
                s2_bar_chart = gr.Plot(label="3-Class Probabilities", visible=False)
            with gr.Column(scale=1):
                s2_gradcam_video = gr.Video(label="Grad-CAM Comparison", visible=False, autoplay=True, loop=True)

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_video],
        outputs=[
            s1_panel,
            s1_status,
            s1_detail,
            s1_video,
            s1_frame_chart,
            s2_panel,
            s2_status,
            s2_detail,
            s2_bar_chart,
            s2_gradcam_video,
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
