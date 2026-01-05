"""Job status schemas for async operations."""

from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


class JobStatus(BaseModel):
    """Status of an async job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = Field(
        ..., description="Current job status"
    )
    progress: Optional[float] = Field(
        default=None, ge=0, le=1, description="Progress (0.0-1.0)"
    )
    message: Optional[str] = Field(
        default=None, description="Status message"
    )
    created_at: str = Field(..., description="ISO timestamp when job was created")
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp when job started"
    )
    completed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when job completed"
    )
    result: Optional[Any] = Field(
        default=None, description="Full result when completed"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
