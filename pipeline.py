"""
VLM-Guided Unsupervised Anomaly Detection Pipeline
===================================================
Combines CLIP patch embeddings + VLM normality priors + PatchCore scoring.
Benchmarked on MVTec AD. Edge-optimized via OpenVINO / TensorRT.

Usage:
    # Train on a category
    python pipeline.py --mode train --category bottle --data_root ./data/mvtec

    # Evaluate
    python pipeline.py --mode eval --category bottle --data_root ./data/mvtec

    # Run demo on a single image
    python pipeline.py --mode demo --category bottle --image path/to/image.jpg

    # Full benchmark (all 15 categories)
    python pipeline.py --mode benchmark --data_root ./data/mvtec

Author: Abhay A | github.com/abhay
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")

from src.feature_extractor import CLIPPatchExtractor
from src.vlm_descriptor import VLMNormalityDescriptor
from src.anomaly_scorer import PatchCoreScorer


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "clip_model": "ViT-B-16",
    "clip_pretrained": "openai",
    "coreset_ratio": 0.25,
    "n_neighbors": 9,
    "vlm_backend": "static",       # "static" | "llava" | "gpt4v"
    "vlm_prior_weight": 0.2,
    "image_size": 224,
    "max_patches": 50000,          # cap before coreset to keep training fast
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class VLMAnomalyPipeline:
    def __init__(self, config: dict = DEFAULT_CONFIG):
        self.cfg = config
        device = config["device"]

        print(f"[Init] Device: {device}")

        self.extractor = CLIPPatchExtractor(
            model_name=config["clip_model"],
            pretrained=config["clip_pretrained"],
            device=device,
        )
        self.vlm = VLMNormalityDescriptor(
            clip_model_name=config["clip_model"],
            pretrained=config["clip_pretrained"],
            device=device,
            backend=config["vlm_backend"],
        )
        self.scorer = PatchCoreScorer(
            coreset_ratio=config["coreset_ratio"],
            n_neighbors=config["n_neighbors"],
            vlm_prior_weight=config["vlm_prior_weight"],
            device=device,
        )

    def train(self, category: str, data_root: str, save_dir: str = "checkpoints"):
        """Build memory bank from normal training images."""
        train_dir = Path(data_root) / category / "train" / "good"
        assert train_dir.exists(), f"Training dir not found: {train_dir}"

        image_paths = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))
        print(f"[Train] {len(image_paths)} normal images for '{category}'")

        # Get VLM text prior for this category
        vlm_prior = self.vlm.get_text_prior(category)

        # Extract patch embeddings from all training images
        all_patches = []
        for path in tqdm(image_paths, desc="Extracting features"):
            img = Image.open(path).convert("RGB")
            patches = self.extractor.extract_patches(img)
            all_patches.append(patches)

        all_patches = torch.cat(all_patches, dim=0)
        print(f"[Train] Total patches: {all_patches.shape[0]:,}")

        # Cap patches before coreset to keep training fast
        max_patches = self.cfg.get("max_patches", 50000)
        if len(all_patches) > max_patches:
            idx = torch.randperm(len(all_patches))[:max_patches]
            all_patches = all_patches[idx]
            print(f"[Train] Subsampled to {max_patches:,} patches before coreset")

        # Build VLM-guided memory bank
        self.scorer.build_memory_bank(all_patches, vlm_prior=vlm_prior)

        # Save
        os.makedirs(save_dir, exist_ok=True)
        self.scorer.save(f"{save_dir}/{category}_scorer.pkl")
        print(f"[✓] Training complete for '{category}'")

    def evaluate(self, category: str, data_root: str, ckpt_dir: str = "checkpoints"):
        """Evaluate on MVTec AD test set. Returns image-level AUROC."""
        self.scorer.load(f"{ckpt_dir}/{category}_scorer.pkl")

        test_dir = Path(data_root) / category / "test"

        scores, labels = [], []

        for defect_type in sorted(os.listdir(test_dir)):
            defect_path = test_dir / defect_type
            is_anomaly = defect_type != "good"

            for img_path in sorted(defect_path.glob("*.png")):
                img = Image.open(img_path).convert("RGB")
                patches = self.extractor.extract_patches(img)
                score, _ = self.scorer.score(patches)
                scores.append(score)
                labels.append(1 if is_anomaly else 0)

        auroc = roc_auc_score(labels, scores)
        print(f"[Eval] {category}: Image-AUROC = {auroc:.4f}")
        return auroc

    def run_demo(
        self,
        category: str,
        image_path: str,
        ckpt_dir: str = "checkpoints",
        output_path: str = "demo_output.png",
    ):
        """Run inference on a single image and save heatmap overlay."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        self.scorer.load(f"{ckpt_dir}/{category}_scorer.pkl")

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img.resize((224, 224)))

        patches = self.extractor.extract_patches(img)
        score, heatmap = self.scorer.score(patches, spatial_shape=(224, 224))

        print(f"[Demo] Anomaly score: {score:.4f}")
        print(f"[Demo] Verdict: {'ANOMALY' if score > 0.5 else 'NORMAL'}")

        if heatmap is not None:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(
                f"Category: {category} | Score: {score:.4f} | "
                + ("⚠ ANOMALY" if score > 0.5 else "✓ NORMAL"),
                fontsize=13
            )

            axes[0].imshow(img_np)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Anomaly heatmap")
            axes[1].axis("off")

            overlay = img_np.copy().astype(np.float32) / 255.0
            heat_color = cm.jet(heatmap)[..., :3]
            blended = 0.6 * overlay + 0.4 * heat_color
            axes[2].imshow(np.clip(blended, 0, 1))
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[✓] Saved to {output_path}")

        return score, heatmap


# ── Benchmark runner ──────────────────────────────────────────────────────────

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

def run_full_benchmark(data_root: str, ckpt_dir: str = "checkpoints"):
    """Train and evaluate on all 15 MVTec AD categories."""
    pipeline = VLMAnomalyPipeline()
    results = {}

    for cat in MVTEC_CATEGORIES:
        print(f"\n{'='*50}")
        print(f"Category: {cat}")
        print(f"{'='*50}")
        pipeline.train(cat, data_root, ckpt_dir)
        auroc = pipeline.evaluate(cat, data_root, ckpt_dir)
        results[cat] = round(auroc * 100, 2)

    mean_auroc = np.mean(list(results.values()))
    results["mean"] = round(mean_auroc, 2)

    print(f"\n{'='*50}")
    print(f"Mean AUROC across all categories: {mean_auroc:.2f}%")
    print(f"{'='*50}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Anomaly Detection Pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "demo", "benchmark"], required=True)
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--data_root", type=str, default="./data/mvtec")
    parser.add_argument("--image", type=str, help="Image path for demo mode")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--output_path", type=str, default="demo_output.png")
    parser.add_argument("--vlm_backend", type=str, default="static",
                        choices=["static", "llava", "gpt4v"])
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "vlm_backend": args.vlm_backend}
    pipeline = VLMAnomalyPipeline(config)

    if args.mode == "train":
        pipeline.train(args.category, args.data_root, args.ckpt_dir)
    elif args.mode == "eval":
        pipeline.evaluate(args.category, args.data_root, args.ckpt_dir)
    elif args.mode == "demo":
        assert args.image, "--image required for demo mode"
        pipeline.run_demo(args.category, args.image, args.ckpt_dir, args.output_path)
    elif args.mode == "benchmark":
        run_full_benchmark(args.data_root, args.ckpt_dir)