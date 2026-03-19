"""
Gradio demo for VLM-Guided Anomaly Detection.
Run: python demo/app.py
Author: Abhay A
"""

import gradio as gr
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import sys
sys.path.append("..")

from pipeline import VLMAnomalyPipeline, DEFAULT_CONFIG, MVTEC_CATEGORIES

pipeline = VLMAnomalyPipeline(DEFAULT_CONFIG)

def run_inference(image: Image.Image, category: str, ckpt_dir: str = "checkpoints"):
    try:
        pipeline.scorer.load(f"{ckpt_dir}/{category}_scorer.pkl")
    except FileNotFoundError:
        return None, None, f"⚠ No checkpoint found for '{category}'. Run training first."

    img = image.convert("RGB").resize((224, 224))
    img_np = np.array(img)

    patches = pipeline.extractor.extract_patches(img)
    score, heatmap = pipeline.scorer.score(patches, spatial_shape=(224, 224))

    verdict = "⚠ ANOMALY DETECTED" if score > 0.5 else "✓ NORMAL"
    label = f"{verdict} | Score: {score:.4f}"

    if heatmap is not None:
        heat_color = (cm.jet(heatmap)[..., :3] * 255).astype(np.uint8)
        overlay = (0.6 * img_np + 0.4 * heat_color).clip(0, 255).astype(np.uint8)
        return Image.fromarray(heat_color), Image.fromarray(overlay), label

    return None, None, label


with gr.Blocks(title="VLM Anomaly Detection — Abhay A") as demo:
    gr.Markdown("# VLM-Guided Unsupervised Anomaly Detection")
    gr.Markdown(
        "Upload an image, select a category. No labeled anomaly data was used in training. "
        "[[GitHub]](https://github.com/abhay) | [[LinkedIn]](https://linkedin.com/in/abhay)"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input image")
            category = gr.Dropdown(
                choices=MVTEC_CATEGORIES,
                value="bottle",
                label="Product category",
            )
            run_btn = gr.Button("Detect anomalies", variant="primary")
        with gr.Column():
            heatmap_out = gr.Image(label="Anomaly heatmap")
            overlay_out = gr.Image(label="Overlay")
            score_out = gr.Textbox(label="Result")

    run_btn.click(
        fn=run_inference,
        inputs=[input_image, category],
        outputs=[heatmap_out, overlay_out, score_out],
    )

    gr.Examples(
        examples=[["assets/sample_bottle.jpg", "bottle"]],
        inputs=[input_image, category],
    )

if __name__ == "__main__":
    demo.launch(share=True)
