"""
Pydantic schemas for apply.ai job application copilot.

This module defines the core data models used throughout the multi-agent architecture
for job posting analysis, resume parsing, research, and application generation.
"""

from .core import (
    JobPosting,
    Resume,
    ResumeBullet,
    ResumeSection,
    Fact,
    FactSheet,
    TailoredBullet,
    CoverLetter,
    CompBand,
    Metrics,
)

__all__ = [
    "JobPosting",
    "Resume",
    "ResumeBullet",
    "ResumeSection",
    "Fact",
    "FactSheet",
    "TailoredBullet",
    "CoverLetter",
    "CompBand",
    "Metrics",
]
