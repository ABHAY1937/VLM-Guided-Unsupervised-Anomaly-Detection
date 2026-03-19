"""
PatchCore anomaly scorer with VLM-guided memory bank.

Key innovation: VLM text priors act as a soft filter during memory bank
construction, upweighting patches whose visual features align with the
normality description.

Author: Abhay A | github.com/abhay
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple
import pickle
import os


class PatchCoreScorer:
    """
    Anomaly scorer based on PatchCore (Roth et al. 2022) extended with
    VLM-guided coreset construction.

    Pipeline:
      1. Build memory bank from normal training patches (+ VLM prior weighting)
      2. Apply greedy coreset subsampling to reduce memory (k-center greedy)
      3. At inference: compute max patch distance as anomaly score
      4. Reshape patch scores into spatial heatmap
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        n_neighbors: int = 9,
        vlm_prior_weight: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.vlm_prior_weight = vlm_prior_weight
        self.device = device

        self.memory_bank: Optional[torch.Tensor] = None
        self.nn_index: Optional[NearestNeighbors] = None

    # ── Memory bank construction ───────────────────────────────────────────────

    def build_memory_bank(
        self,
        patch_embeddings: torch.Tensor,   # (N_total_patches, D)
        vlm_prior: Optional[torch.Tensor] = None,  # (1, D)
    ):
        """
        Build and subsample memory bank from normal training patches.

        Args:
            patch_embeddings: All patch features from normal training images.
            vlm_prior: CLIP text embedding of normality description.
                       When provided, patches are weighted by cosine similarity
                       to the prior before coreset selection.
        """
        patches = patch_embeddings.to(self.device)

        if vlm_prior is not None and self.vlm_prior_weight > 0:
            patches = self._apply_vlm_weighting(patches, vlm_prior.to(self.device))

        # Greedy k-center coreset subsampling
        n_select = max(1, int(len(patches) * self.coreset_ratio))
        coreset_idx = self._greedy_coreset(patches, n_select)
        self.memory_bank = patches[coreset_idx].cpu()

        print(
            f"[PatchCore] Memory bank: {len(patches):,} → {len(coreset_idx):,} patches "
            f"({self.coreset_ratio*100:.0f}% coreset)"
        )

        # Build approximate NN index (CPU — fast enough for coreset size)
        self.nn_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="ball_tree",
            metric="minkowski",
            n_jobs=-1,
        )
        self.nn_index.fit(self.memory_bank.numpy())

    def _apply_vlm_weighting(
        self,
        patches: torch.Tensor,
        vlm_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft-weight patches by cosine similarity to VLM text prior.
        Handles dimension mismatch between visual (768-dim) and text (512-dim)
        embeddings via linear interpolation projection.
        """
        v_dim = patches.shape[-1]
        t_dim = vlm_prior.shape[-1]

        if v_dim != t_dim:
            prior = F.normalize(
                torch.nn.functional.interpolate(
                    vlm_prior.unsqueeze(0),
                    size=v_dim,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0),
                dim=-1,
            )
        else:
            prior = vlm_prior

        sims = (patches @ prior.T).squeeze(-1)
        weights = (sims * self.vlm_prior_weight + 1.0).unsqueeze(-1)
        mean = patches.mean(dim=0, keepdim=True)
        weighted = patches * weights + mean * (1 - weights)
        return F.normalize(weighted, dim=-1)

    def _greedy_coreset(self, features: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        Greedy k-center coreset: iteratively pick the point farthest
        from the current selected set. O(n × k).
        """
        n = len(features)
        selected = [torch.randint(n, (1,)).item()]
        min_dists = torch.full((n,), float("inf"), device=self.device)

        for _ in range(n_select - 1):
            last = features[selected[-1]].unsqueeze(0)
            dists = torch.cdist(features, last).squeeze(-1)
            min_dists = torch.minimum(min_dists, dists)
            selected.append(min_dists.argmax().item())

        return torch.tensor(selected)

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(
        self,
        patch_embeddings: torch.Tensor,  # (N_patches, D)
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute image-level anomaly score and pixel-level heatmap.

        Returns:
            score (float): Image anomaly score (higher = more anomalous).
            heatmap (np.ndarray | None): (H, W) spatial anomaly map.
        """
        assert self.nn_index is not None, "Call build_memory_bank() first."

        patches_np = patch_embeddings.cpu().numpy()
        dists, _ = self.nn_index.kneighbors(patches_np)

        # Patch-level score: mean distance to k nearest neighbors
        patch_scores = dists.mean(axis=1)

        # Image-level score: max patch score (PatchCore paper)
        image_score = float(patch_scores.max())

        heatmap = None
        if spatial_shape is not None:
            h, w = spatial_shape
            n_patches = patch_scores.shape[0]
            grid_size = int(n_patches ** 0.5)
            if grid_size * grid_size == n_patches:
                import cv2
                raw = patch_scores.reshape(grid_size, grid_size).astype(np.float32)
                heatmap = cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC)
                # Normalize to [0, 1]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return image_score, heatmap

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"memory_bank": self.memory_bank, "nn_index": self.nn_index}, f)
        print(f"[✓] Scorer saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.memory_bank = data["memory_bank"]
        self.nn_index = data["nn_index"]
        print(f"[✓] Scorer loaded from {path}")
