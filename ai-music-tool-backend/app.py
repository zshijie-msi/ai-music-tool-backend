from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

# --------------------------------------------------
# Paths
# --------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_ASSETS_DIR = BACKEND_DIR / "model_assets"
MODEL_PATH = MODEL_ASSETS_DIR / "a4_cnn_multilabel_best.pt"
LABEL_MAP_PATH = MODEL_ASSETS_DIR / "a4_label_maps_v1.json"

# --------------------------------------------------
# Constants
# --------------------------------------------------
PITCH_MIN = 48
PITCH_MAX = 84
STEPS_PER_SEGMENT = 32
N_PITCH_BINS = PITCH_MAX - PITCH_MIN + 1

DENSITY_INV_MAP = {0: "sparse", 1: "medium", 2: "dense"}
REGISTER_INV_MAP = {0: "low", 1: "mid", 2: "high"}
MOTION_INV_MAP = {0: "smooth", 1: "mixed", 2: "leapy"}

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="AI Music Tool API", version="1.0.0")

cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [x.strip() for x in cors_origins_env.split(",") if x.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeMelodyRequest(BaseModel):
    melody_steps: List[int] = Field(..., description="Length-32 melody using MIDI pitches, 0 for rests")
    complexity: float = 0.35
    tone: float = 0.30
    energy: float = 0.25


class SmallCNNMultiHead(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_density = nn.Linear(64, num_classes)
        self.head_register = nn.Linear(64, num_classes)
        self.head_motion = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        return self.head_density(x), self.head_register(x), self.head_motion(x)


device = "cpu"
_model: Optional[SmallCNNMultiHead] = None
_label_maps = None


def load_label_maps():
    global _label_maps
    if _label_maps is None and LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            _label_maps = json.load(f)
    return _label_maps


def load_model() -> Optional[SmallCNNMultiHead]:
    global _model
    if _model is not None:
        return _model
    if torch is None or nn is None:
        return None
    if not MODEL_PATH.exists():
        return None

    model = SmallCNNMultiHead()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _model = model
    return _model


def normalize_melody_steps(melody_steps: List[int]) -> List[int]:
    cleaned: List[int] = []
    for pitch in melody_steps:
        try:
            value = int(pitch)
        except Exception:
            value = 0
        if value == 0:
            cleaned.append(0)
        elif PITCH_MIN <= value <= PITCH_MAX:
            cleaned.append(value)
        else:
            cleaned.append(0)

    if len(cleaned) < STEPS_PER_SEGMENT:
        cleaned += [0] * (STEPS_PER_SEGMENT - len(cleaned))
    else:
        cleaned = cleaned[:STEPS_PER_SEGMENT]
    return cleaned


def melody_steps_to_matrix(melody_steps: List[int]) -> np.ndarray:
    matrix = np.zeros((STEPS_PER_SEGMENT, N_PITCH_BINS), dtype=np.float32)
    for t, pitch in enumerate(melody_steps[:STEPS_PER_SEGMENT]):
        if pitch == 0:
            continue
        if PITCH_MIN <= pitch <= PITCH_MAX:
            matrix[t, pitch - PITCH_MIN] = 1.0
    return matrix


def heuristic_predictions(melody_steps: List[int]):
    notes = [p for p in melody_steps if p > 0]
    note_count = len(notes)
    density_ratio = note_count / STEPS_PER_SEGMENT
    avg_pitch = sum(notes) / len(notes) if notes else 60.0

    leaps = 0
    intervals = []
    prev = None
    for p in notes:
        if prev is not None:
            iv = abs(p - prev)
            intervals.append(iv)
            if iv >= 5:
                leaps += 1
        prev = p
    leap_ratio = leaps / max(1, len(intervals))

    density = "sparse" if density_ratio < 0.30 else "medium" if density_ratio < 0.60 else "dense"
    register = "low" if avg_pitch < 60 else "mid" if avg_pitch < 72 else "high"
    motion = "smooth" if leap_ratio < 0.18 else "mixed" if leap_ratio < 0.38 else "leapy"
    return density, register, motion


def model_predictions(melody_steps: List[int]):
    model = load_model()
    if model is None or torch is None:
        return None
    matrix = melody_steps_to_matrix(melody_steps)
    x = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out_density, out_register, out_motion = model(x)
    density_idx = int(torch.argmax(out_density, dim=1).item())
    register_idx = int(torch.argmax(out_register, dim=1).item())
    motion_idx = int(torch.argmax(out_motion, dim=1).item())
    return (
        DENSITY_INV_MAP[density_idx],
        REGISTER_INV_MAP[register_idx],
        MOTION_INV_MAP[motion_idx],
    )


def build_fingerprint(density_label: str, register_label: str, motion_label: str):
    register_text = {
        "high": "Your melody sits higher and feels lighter.",
        "mid": "Your melody stays in a balanced middle range.",
        "low": "Your melody sits lower and feels more grounded.",
    }
    density_text = {
        "sparse": "It has more space between note events.",
        "medium": "It has a balanced rhythmic density.",
        "dense": "It has a denser and more active rhythm.",
    }
    motion_text = {
        "smooth": "Its movement feels smooth and connected.",
        "mixed": "Its movement mixes gentle flow with some variation.",
        "leapy": "Its movement includes larger jumps and feels more active.",
    }

    summary_parts = []
    summary_parts.append({"high": "high and airy", "mid": "balanced and centered", "low": "low and grounded"}[register_label])
    summary_parts.append({"sparse": "with spacious rhythm", "medium": "with balanced rhythm", "dense": "with dense rhythmic activity"}[density_label])
    summary_parts.append({"smooth": "and smooth movement", "mixed": "and mixed movement", "leapy": "and stronger melodic jumps"}[motion_label])

    return {
        "register_line": register_text[register_label],
        "density_line": density_text[density_label],
        "motion_line": motion_text[motion_label],
        "summary": f"This melody feels {summary_parts[0]}, {summary_parts[1]} {summary_parts[2]}.",
    }


SUGGESTIONS = {
    "dreamy_calm": [
        {
            "variant_name": "Airy Bell",
            "preset": "Soft Bell + Pad",
            "rhythm_advice": "Keep it simple",
            "tone_hint": "light, spacious, floating",
        },
        {
            "variant_name": "Glass Cloud",
            "preset": "Glass Keys + Air Pad",
            "rhythm_advice": "Let notes breathe",
            "tone_hint": "clean, airy, delicate",
        },
    ],
    "gentle_reflective": [
        {
            "variant_name": "Felt Quiet",
            "preset": "Felt Piano + Ambient Pad",
            "rhythm_advice": "Leave more space between phrases",
            "tone_hint": "quiet, intimate, reflective",
        },
        {
            "variant_name": "Soft Room",
            "preset": "Warm Piano + Reverb",
            "rhythm_advice": "Use gentle phrasing",
            "tone_hint": "warm, personal, calm",
        },
    ],
    "neutral_lyrical": [
        {
            "variant_name": "Soft Hybrid",
            "preset": "Piano + Light Texture Layer",
            "rhythm_advice": "Keep the phrasing smooth",
            "tone_hint": "gentle, balanced, lyrical",
        },
        {
            "variant_name": "Clear Line",
            "preset": "Mellow Keys + Pad",
            "rhythm_advice": "Preserve melodic contour",
            "tone_hint": "clean, even, singable",
        },
    ],
    "driving_rhythmic": [
        {
            "variant_name": "Pulse Motion",
            "preset": "Pluck + Tight Bass",
            "rhythm_advice": "Lean into repetition",
            "tone_hint": "steady, active, rhythmic",
        },
        {
            "variant_name": "Night Run",
            "preset": "Synth Pluck + Soft Kit",
            "rhythm_advice": "Keep accents consistent",
            "tone_hint": "focused, forward, moving",
        },
    ],
    "bold_expressive": [
        {
            "variant_name": "Bright Leap",
            "preset": "Lead Synth + Support Pad",
            "rhythm_advice": "Emphasize larger jumps",
            "tone_hint": "bright, expressive, vivid",
        },
        {
            "variant_name": "Open Gesture",
            "preset": "Layered Lead + Atmos Pad",
            "rhythm_advice": "Let phrase peaks stand out",
            "tone_hint": "expansive, emotional, open",
        },
    ],
}


def choose_families(density_label: str, register_label: str, motion_label: str) -> List[str]:
    if register_label == "high" and density_label == "sparse" and motion_label == "smooth":
        return ["dreamy_calm", "gentle_reflective", "neutral_lyrical"]
    if density_label == "dense" and motion_label in {"mixed", "leapy"}:
        return ["driving_rhythmic", "bold_expressive", "neutral_lyrical"]
    if motion_label == "leapy":
        return ["bold_expressive", "driving_rhythmic", "neutral_lyrical"]
    if density_label == "medium":
        return ["neutral_lyrical", "gentle_reflective", "dreamy_calm"]
    return ["gentle_reflective", "neutral_lyrical", "dreamy_calm"]


def build_cards(density_label: str, register_label: str, motion_label: str, fingerprint_summary: str):
    family_order = choose_families(density_label, register_label, motion_label)
    cards = []
    for family_key in family_order[:3]:
        variant = SUGGESTIONS[family_key][0]
        style_label = family_key.replace("_", " / ").title()
        explanation = (
            f"The melody is {register_label} in register, {density_label} in density, "
            f"and {motion_label} in motion. {fingerprint_summary} "
            f"This makes {style_label} — {variant['variant_name']} a strong direction to explore."
        )
        cards.append(
            {
                "family_key": family_key,
                "style": style_label,
                "variant_name": variant["variant_name"],
                "preset": variant["preset"],
                "rhythm_advice": variant["rhythm_advice"],
                "tone_hint": variant["tone_hint"],
                "explanation": explanation,
            }
        )
    return cards


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file_found": MODEL_PATH.exists(),
        "label_map_found": LABEL_MAP_PATH.exists(),
        "torch_available": torch is not None,
        "cors_origins": allow_origins,
    }


@app.post("/analyze_melody")
def analyze_melody(payload: AnalyzeMelodyRequest):
    melody_steps = normalize_melody_steps(payload.melody_steps)

    preds = model_predictions(melody_steps)
    backend_mode = "model" if preds is not None else "heuristic_fallback"
    if preds is None:
        preds = heuristic_predictions(melody_steps)

    density_label, register_label, motion_label = preds
    fingerprint = build_fingerprint(density_label, register_label, motion_label)
    cards = build_cards(density_label, register_label, motion_label, fingerprint["summary"])

    return {
        "input_melody_steps": melody_steps,
        "predictions": {
            "density": density_label,
            "register": register_label,
            "motion": motion_label,
        },
        "fingerprint": fingerprint,
        "cards": cards,
        "refinement": {
            "complexity": payload.complexity,
            "tone": payload.tone,
            "energy": payload.energy,
        },
        "backend_mode": backend_mode,
    }
