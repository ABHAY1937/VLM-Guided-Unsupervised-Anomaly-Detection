"""
VLM-based normality descriptor — the novel contribution.
Uses LLaVA (local) or GPT-4V (API) to describe what 'normal' looks like
for a given product category. These text priors guide memory bank construction.

Author: Abhay A | github.com/abhay
"""

import torch
import torch.nn.functional as F
import open_clip
from typing import Optional
import base64
from io import BytesIO
from PIL import Image


# ── Default normality prompts per MVTec category ──────────────────────────────
CATEGORY_PRIORS = {
    "bottle":     "A normal bottle has a smooth, uniform surface with no cracks, chips, or contamination.",
    "cable":      "A normal cable has intact insulation, uniform color, and no fraying or missing strands.",
    "capsule":    "A normal capsule is smooth, uniformly colored, and has no cracks or surface defects.",
    "carpet":     "A normal carpet has a uniform weave pattern with consistent texture and no holes or stains.",
    "grid":       "A normal grid has uniform, evenly spaced lines with no breaks or missing segments.",
    "hazelnut":   "A normal hazelnut has a smooth, unblemished shell with no cracks or holes.",
    "leather":    "Normal leather has a uniform grain texture with no cuts, folds, or color inconsistencies.",
    "metal_nut":  "A normal metal nut has clean, sharp edges with uniform surface finish and no corrosion.",
    "pill":       "A normal pill has a smooth, uniformly colored surface with no chips or contamination.",
    "screw":      "A normal screw has clean, uniformly spaced threads with no damage or debris.",
    "tile":       "A normal tile has a flat, uniform surface with consistent color and no cracks.",
    "toothbrush": "A normal toothbrush has evenly distributed bristles with no bending or missing tufts.",
    "transistor": "A normal transistor has intact leads and a clean, undamaged component body.",
    "wood":       "Normal wood has a consistent grain pattern with no knots, holes, or surface damage.",
    "zipper":     "A normal zipper has evenly spaced teeth with no missing, bent, or misaligned elements.",
}


class VLMNormalityDescriptor:
    """
    Generates text-based normality descriptions and encodes them as
    CLIP text embeddings to serve as soft priors for anomaly scoring.

    Supports:
      - Static priors (fast, no API needed)
      - LLaVA local inference (via Ollama)
      - GPT-4V API (best quality, requires key)
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        backend: str = "static",  # "static" | "llava" | "gpt4v"
        openai_api_key: Optional[str] = None,
        ollama_model: str = "llava:13b",
    ):
        self.device = device
        self.backend = backend
        self.openai_api_key = openai_api_key
        self.ollama_model = ollama_model

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.clip_model = self.clip_model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

    def get_text_prior(
        self,
        category: str,
        reference_image: Optional[Image.Image] = None,
    ) -> torch.Tensor:
        """
        Returns a CLIP text embedding encoding normality for the given category.
        Shape: (1, embed_dim)
        """
        description = self._generate_description(category, reference_image)
        return self._encode_text(description)

    def _generate_description(
        self,
        category: str,
        reference_image: Optional[Image.Image] = None,
    ) -> str:
        if self.backend == "static":
            desc = CATEGORY_PRIORS.get(
                category,
                f"A normal {category} with no visible defects or anomalies."
            )
            print(f"[VLM] Static prior for '{category}': {desc[:60]}...")
            return desc

        elif self.backend == "llava":
            return self._query_llava(category, reference_image)

        elif self.backend == "gpt4v":
            return self._query_gpt4v(category, reference_image)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _query_llava(
        self,
        category: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """Query LLaVA via Ollama (local inference, no API key needed)."""
        try:
            import requests, json

            prompt = (
                f"Describe in one sentence what a defect-free, normal {category} "
                f"looks like. Be specific about texture, color, and surface properties."
            )

            payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}

            if image is not None:
                buf = BytesIO()
                image.save(buf, format="JPEG")
                payload["images"] = [base64.b64encode(buf.getvalue()).decode()]

            resp = requests.post("http://localhost:11434/api/generate", json=payload)
            result = resp.json()["response"].strip()
            print(f"[VLM-LLaVA] '{category}': {result[:80]}...")
            return result

        except Exception as e:
            print(f"[VLM-LLaVA] Fallback to static prior. Error: {e}")
            return CATEGORY_PRIORS.get(category, f"A normal {category}.")

    def _query_gpt4v(
        self,
        category: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """Query GPT-4V via OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"You are a quality control expert. Describe in exactly one "
                            f"sentence what a defect-free normal {category} looks like. "
                            f"Be specific about surface texture, color uniformity, and structure."
                        ),
                    }
                ],
            }]

            if image is not None:
                buf = BytesIO()
                image.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                messages[0]["content"].insert(0, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=100,
            )
            result = resp.choices[0].message.content.strip()
            print(f"[VLM-GPT4V] '{category}': {result[:80]}...")
            return result

        except Exception as e:
            print(f"[VLM-GPT4V] Fallback to static prior. Error: {e}")
            return CATEGORY_PRIORS.get(category, f"A normal {category}.")

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.clip_model.encode_text(tokens)
        return F.normalize(feat, dim=-1)
