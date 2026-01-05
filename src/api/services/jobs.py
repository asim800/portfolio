"""Async job management for long-running simulations."""

import uuid
from datetime import datetime
from typing import Dict, Optional, Any

from ..schemas.jobs import JobStatus


class JobManager:
    """
    In-memory job manager for async simulation tasks.

    For production, replace with Redis or database-backed storage.
    """

    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}

    def create_job(self) -> str:
        """Create a new pending job and return its ID."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobStatus(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Get job status by ID."""
        return self._jobs.get(job_id)

    def start_job(self, job_id: str, message: str = None):
        """Mark job as running."""
        if job_id in self._jobs:
            self._jobs[job_id].status = "running"
            self._jobs[job_id].started_at = datetime.utcnow().isoformat() + "Z"
            self._jobs[job_id].progress = 0.0
            if message:
                self._jobs[job_id].message = message

    def update_progress(self, job_id: str, progress: float, message: str = None):
        """Update job progress (0.0 to 1.0)."""
        if job_id in self._jobs:
            self._jobs[job_id].progress = min(1.0, max(0.0, progress))
            if message:
                self._jobs[job_id].message = message

    def complete_job(self, job_id: str, result: Any):
        """Mark job as completed with result."""
        if job_id in self._jobs:
            self._jobs[job_id].status = "completed"
            self._jobs[job_id].completed_at = datetime.utcnow().isoformat() + "Z"
            self._jobs[job_id].progress = 1.0
            self._jobs[job_id].result = result
            self._jobs[job_id].message = "Completed successfully"

    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        if job_id in self._jobs:
            self._jobs[job_id].status = "failed"
            self._jobs[job_id].completed_at = datetime.utcnow().isoformat() + "Z"
            self._jobs[job_id].error = error
            self._jobs[job_id].message = f"Failed: {error}"

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Returns True if job was cancelled, False if not found or already completed.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in ["pending", "running"]:
            job.status = "cancelled"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.message = "Cancelled by user"
            return True

        return False

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove completed jobs older than max_age_hours."""
        now = datetime.utcnow()
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.completed_at:
                completed = datetime.fromisoformat(
                    job.completed_at.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                age_hours = (now - completed).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]

        return len(to_remove)

    def list_jobs(self, status: str = None) -> list:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)


# Singleton instance
job_manager = JobManager()
