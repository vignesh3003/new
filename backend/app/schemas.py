"""Pydantic schemas for FastAPI responses."""

from typing import List

from pydantic import BaseModel, Field


class PlotPayload(BaseModel):
    """Container for base64 encoded plot images."""

    title: str
    image_base64: str = Field(..., description="PNG image encoded as base64 data URI.")


class FormulaPayload(BaseModel):
    """Math formula metadata."""

    title: str
    latex: str


class StepPayload(BaseModel):
    """Step-by-step descriptions of the math pipeline."""

    title: str
    details: str


class AnalysisResponse(BaseModel):
    """Full response returned after processing an ECG upload."""

    fft_frequencies: List[float]
    fft_magnitudes: List[float]
    pca_components: List[List[float]]
    explained_variance: List[float]
    plots: List[PlotPayload]
    formulas: List[FormulaPayload]
    steps: List[StepPayload]
    heartbeat_count: int
    r_peak_indices: List[int]
    method_summary: str


