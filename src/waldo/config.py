from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    model_config = {"frozen": True}

    input_dir: Path
    refs_dir: Path | None = None
    output_dir: Path
    fps: float = Field(default=2.0, gt=0)
    threshold: float = Field(default=0.5, ge=0, le=1)
    gap: float = Field(default=2.0, ge=0)
    min_len: float = Field(default=0.5, ge=0)
    pad: float = Field(default=15.0, ge=0)
    workers: int = Field(default=1, ge=1)
