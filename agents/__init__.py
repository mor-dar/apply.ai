"""
Agents package for apply.ai job application copilot.

This package contains agents that orchestrate workflows and manage state
within the LangGraph architecture. Agents use tools for actual processing
while handling workflow orchestration, error handling, and state management.
"""

from .job_post_parser_agent import JobPostParserAgent, JobPostParsingState
from .resume_parser_agent import ResumeParserAgent, ResumeParsingState
from .company_research_agent import CompanyResearchAgent, CompanyResearchState
from .compensation_analyst_agent import CompensationAnalystAgent, CompensationAnalysisState

__all__ = [
    "JobPostParserAgent",
    "JobPostParsingState",
    "ResumeParserAgent",
    "ResumeParsingState",
    "CompanyResearchAgent",
    "CompanyResearchState",
    "CompensationAnalystAgent",
    "CompensationAnalysisState",
]
