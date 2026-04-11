from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# Supported file extensions.
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class Config(BaseModel):
    model_config = {"frozen": True}

    input_dir: Path
    refs_dir: Path | None = None
    output_dir: Path | None = None
    fps: float = Field(default=2.0, gt=0)
    threshold: float = Field(default=0.5, ge=0, le=1)
    gap: float = Field(default=2.0, ge=0)
    min_len: float = Field(default=0.5, ge=0)
    pad: float = Field(default=15.0, ge=0)
    workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_dirs(self) -> "Config":
        if self.output_dir is not None and self.input_dir.resolve() == self.output_dir.resolve():
            raise ValueError("--input and --output must be different directories")
        return self
