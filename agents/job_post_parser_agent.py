"""
JobPostParserAgent for orchestrating job posting parsing within LangGraph workflow.

This agent handles the workflow orchestration, error handling, and state management
for job posting parsing. It uses the JobPostingParser tool internally for the 
actual NLP processing while managing retries, validation, and LangGraph state updates.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from tools.job_posting_parser import JobPostingParser


# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
CONFIDENCE_MIN = 0.65
REQUIRED_FIELDS = ("title", "company")


class JobPostParsingState(TypedDict):
    """
    State dictionary for job posting parsing workflow.

    This state is passed between nodes in the LangGraph and tracks
    the progress and results of job posting parsing operations.
    """

    # Input data
    job_text: str
    job_title: Optional[str]
    company_name: Optional[str]
    job_location: Optional[str]

    # Processing status
    status: str
    error_message: Optional[str]
    retry_count: int

    # Results
    parsed_job: Optional[Dict[str, Any]]  # Serialized JobPosting
    parser_report: Optional[Dict[str, Any]]  # Serialized ParserReport
    processing_metadata: Dict[str, Any]

    # Timestamps (as ISO strings for JSON serialization)
    started_at: Optional[str]
    completed_at: Optional[str]


class JobPostParserAgent:
    """
    Agent for orchestrating job posting parsing workflows.

    This agent manages the complete workflow for parsing job descriptions,
    including error handling, retries, and LangGraph state management.
    It uses the JobPostingParser tool for the actual parsing work.
    """

    def __init__(
        self,
        max_keywords: int = 30,
        min_keyword_length: int = 2,
        max_retries: int = 3,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the JobPostParserAgent.

        Args:
            max_keywords: Maximum keywords to extract (passed to tool)
            min_keyword_length: Minimum keyword length (passed to tool)
            max_retries: Maximum retry attempts for failed operations
            enable_checkpoints: Whether to enable LangGraph checkpointing
        """
        self.max_retries = max_retries

        # Initialize the parsing tool
        self.parser = JobPostingParser(
            max_keywords=max_keywords, min_keyword_length=min_keyword_length
        )

        # Build the LangGraph workflow
        self.graph = self._build_graph()

        # Configure checkpointing if enabled
        self.checkpointer = MemorySaver() if enable_checkpoints else None

        # Compile the graph
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpointer, debug=True
        )

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for job posting parsing.

        Returns:
            Configured StateGraph for the parsing workflow
        """
        # Create the graph with our state model
        workflow = StateGraph(JobPostParsingState)

        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_parsing)
        workflow.add_node("parse_job_posting", self._parse_job_posting)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_results)

        # Define the workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "parse_job_posting")

        # Conditional routing from parsing
        workflow.add_conditional_edges(
            "parse_job_posting",
            self._should_retry,
            {"validate": "validate_results", "retry": "handle_error", "failed": END},
        )

        # From validation
        workflow.add_conditional_edges(
            "validate_results",
            self._validation_complete,
            {"success": "finalize", "retry": "handle_error"},
        )

        # Error handling flow
        workflow.add_conditional_edges(
            "handle_error",
            self._should_continue_after_error,
            {"retry": "parse_job_posting", "failed": END},
        )

        # Finalization
        workflow.add_edge("finalize", END)

        return workflow

    def _initialize_parsing(self, state: JobPostParsingState) -> JobPostParsingState:
        """
        Initialize the parsing workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated state with initialization metadata
        """
        logger.info(
            f"Initializing job posting parsing for: {state.get('job_title', 'Unknown Position')}"
        )

        new_state = state.copy()
        new_state.update(
            {
                "status": "initializing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "processing_metadata": {
                    "parser_config": {
                        "max_keywords": self.parser.max_keywords,
                        "min_keyword_length": self.parser.min_keyword_length,
                    },
                    "workflow_version": "1.0",
                },
            }
        )
        return new_state

    def _parse_job_posting(self, state: JobPostParsingState) -> JobPostParsingState:
        """
        Parse the job posting using the JobPostingParser tool with quality gates.

        This method implements the agent policy:
        1. Call deterministic parser tool
        2. Check confidence against threshold
        3. Apply quality gates (required fields, etc.)
        4. Update state with results and report

        Args:
            state: Current workflow state

        Returns:
            Updated state with parsing results and quality report
        """
        try:
            logger.info(
                f"Parsing job posting (attempt {state.get('retry_count', 0) + 1}/{self.max_retries})"
            )

            # Use the tool to parse the job posting (returns tuple now)
            parsed_job, parser_report = self.parser.parse(
                job_text=state["job_text"],
                title=state.get("job_title", ""),
                company=state.get("company_name", ""),
                location=state.get("job_location", ""),
            )

            logger.info(
                f"Parsed job posting: {len(parsed_job.keywords)} keywords, "
                f"{len(parsed_job.requirements)} requirements, "
                f"confidence: {parser_report.confidence:.3f}"
            )

            # Apply quality gates based on confidence and required fields
            missing = [f for f in REQUIRED_FIELDS if f in parser_report.missing_fields]

            if parser_report.confidence < CONFIDENCE_MIN or missing:
                # Log quality issues
                issues = []
                if parser_report.confidence < CONFIDENCE_MIN:
                    issues.append(
                        f"low confidence ({parser_report.confidence:.3f} < {CONFIDENCE_MIN})"
                    )
                if missing:
                    issues.append(f"missing required fields: {missing}")

                logger.warning(f"Quality gates failed: {', '.join(issues)}")
                # Note: In the reference implementation, this would trigger LLM fallback
                # For now, we'll continue with warning but could add fallback logic here

            new_state = state.copy()
            new_state.update(
                {
                    "status": "parsed",
                    "parsed_job": parsed_job.model_dump(),
                    "parser_report": parser_report.model_dump(),
                    "error_message": None,
                }
            )
            return new_state

        except Exception as e:
            error_msg = f"Parsing failed: {str(e)}"
            logger.error(error_msg)

            new_state = state.copy()
            new_state.update({"status": "error", "error_message": error_msg})
            return new_state

    def _validate_results(self, state: JobPostParsingState) -> JobPostParsingState:
        """
        Validate the parsing results using parser report and quality gates.

        This method now uses the ParserReport for more sophisticated validation
        based on confidence scores and quality metrics.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation results
        """
        parsed_job = state.get("parsed_job")
        parser_report = state.get("parser_report")

        if not parsed_job:
            new_state = state.copy()
            new_state.update(
                {
                    "status": "validation_failed",
                    "error_message": "No parsed job data to validate",
                }
            )
            return new_state

        if not parser_report:
            # Fallback to basic validation if no report available
            return self._basic_validation(state, parsed_job)

        try:
            # Use parser report for validation
            validation_issues = []

            # Check confidence threshold
            confidence = parser_report.get("confidence", 0.0)
            if confidence < CONFIDENCE_MIN:
                validation_issues.append(
                    f"Confidence too low ({confidence:.3f} < {CONFIDENCE_MIN})"
                )

            # Check for missing required fields
            missing_fields = parser_report.get("missing_fields", [])
            required_missing = [f for f in REQUIRED_FIELDS if f in missing_fields]
            if required_missing:
                validation_issues.append(f"Missing required fields: {required_missing}")

            # Check for critical warnings
            warnings = parser_report.get("warnings", [])
            critical_warnings = [
                w for w in warnings if "No keywords" in w or "No requirements" in w
            ]
            if critical_warnings:
                validation_issues.extend(critical_warnings)

            if validation_issues:
                error_msg = f"Validation failed: {'; '.join(validation_issues)}"
                logger.warning(error_msg)

                new_state = state.copy()
                new_state.update(
                    {"status": "validation_failed", "error_message": error_msg}
                )
                return new_state

            logger.info("Job posting validation passed")
            new_state = state.copy()
            new_state.update({"status": "validated"})
            return new_state

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)

            new_state = state.copy()
            new_state.update(
                {"status": "validation_failed", "error_message": error_msg}
            )
            return new_state

    def _basic_validation(
        self, state: JobPostParsingState, parsed_job: Dict[str, Any]
    ) -> JobPostParsingState:
        """
        Fallback basic validation when no parser report is available.

        Args:
            state: Current workflow state
            parsed_job: Parsed job data

        Returns:
            Updated state with validation results
        """
        validation_issues = []

        # Check if we got reasonable results
        if len(parsed_job.get("keywords", [])) == 0:
            validation_issues.append("No keywords extracted")

        if len(parsed_job.get("requirements", [])) == 0:
            validation_issues.append("No requirements extracted")

        # Check for obviously bad data
        if len(parsed_job.get("text", "").strip()) < 50:
            validation_issues.append("Job text appears too short")

        if validation_issues:
            error_msg = f"Validation failed: {'; '.join(validation_issues)}"
            logger.warning(error_msg)

            new_state = state.copy()
            new_state.update(
                {"status": "validation_failed", "error_message": error_msg}
            )
            return new_state

        logger.info("Basic job posting validation passed")
        new_state = state.copy()
        new_state.update({"status": "validated"})
        return new_state

    def _handle_error(self, state: JobPostParsingState) -> JobPostParsingState:
        """
        Handle errors and prepare for potential retry.

        Args:
            state: Current workflow state

        Returns:
            Updated state with incremented retry count
        """
        new_retry_count = state.get("retry_count", 0) + 1
        logger.warning(
            f"Handling error (retry {new_retry_count}/{self.max_retries}): {state.get('error_message')}"
        )

        new_state = state.copy()
        new_state.update(
            {
                "retry_count": new_retry_count,
                "status": (
                    "retrying" if new_retry_count <= self.max_retries else "failed"
                ),
            }
        )
        return new_state

    def _finalize_results(self, state: JobPostParsingState) -> JobPostParsingState:
        """
        Finalize the parsing workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated state with completion metadata
        """
        logger.info("Finalizing job posting parsing results")

        # Update processing metadata with parser report metrics
        metadata = state.get("processing_metadata", {}).copy()
        parsed_job = state.get("parsed_job", {})
        parser_report = state.get("parser_report", {})

        metadata.update(
            {
                "final_status": "completed",
                "total_keywords": len(parsed_job.get("keywords", [])),
                "total_requirements": len(parsed_job.get("requirements", [])),
                "retry_attempts": state.get("retry_count", 0),
                "final_confidence": parser_report.get("confidence", 0.0),
                "parser_warnings_count": len(parser_report.get("warnings", [])),
                "missing_fields_count": len(parser_report.get("missing_fields", [])),
            }
        )

        new_state = state.copy()
        new_state.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "processing_metadata": metadata,
            }
        )
        return new_state

    def _should_retry(self, state: JobPostParsingState) -> str:
        """
        Determine if parsing should be retried based on current state.

        Args:
            state: Current workflow state

        Returns:
            Next workflow step: "validate", "retry", or "failed"
        """
        status = state.get("status")
        retry_count = state.get("retry_count", 0)

        if status == "parsed":
            return "validate"
        elif status == "error" and retry_count < self.max_retries:
            return "retry"
        else:
            return "failed"

    def _validation_complete(self, state: JobPostParsingState) -> str:
        """
        Determine next step after validation.

        Args:
            state: Current workflow state

        Returns:
            Next workflow step: "success" or "retry"
        """
        status = state.get("status")
        retry_count = state.get("retry_count", 0)

        if status == "validated":
            return "success"
        elif status == "validation_failed" and retry_count < self.max_retries:
            return "retry"
        else:
            return "failed"

    def _should_continue_after_error(self, state: JobPostParsingState) -> str:
        """
        Determine if workflow should continue after error handling.

        Args:
            state: Current workflow state

        Returns:
            Next workflow step: "retry" or "failed"
        """
        if state.get("status") == "retrying":
            return "retry"
        else:
            return "failed"

    async def parse_job_posting(
        self,
        job_text: str,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None,
        job_location: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse a job posting using the LangGraph workflow.

        Args:
            job_text: Raw job description text
            job_title: Job title (optional)
            company_name: Company name (optional)
            job_location: Job location (optional)
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Dictionary containing the final state and results
        """
        # Create initial state
        initial_state: JobPostParsingState = {
            "job_text": job_text,
            "job_title": job_title,
            "company_name": company_name,
            "job_location": job_location,
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        # Configure for async execution
        config = (
            {"configurable": {"thread_id": thread_id or "default"}}
            if self.checkpointer
            else {}
        )

        try:
            # Execute the workflow
            final_state = await self.compiled_graph.ainvoke(
                initial_state, config=config
            )

            return {
                "success": final_state.get("status") == "completed",
                "job_posting": final_state.get("parsed_job"),
                "metadata": final_state.get("processing_metadata", {}),
                "error": final_state.get("error_message"),
                "retry_count": final_state.get("retry_count", 0),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "job_posting": None,
                "metadata": {"error": "workflow_execution_failed"},
                "error": str(e),
                "retry_count": 0,
            }

    def parse_job_posting_sync(
        self,
        job_text: str,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None,
        job_location: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous version of parse_job_posting for simpler usage.

        Args:
            job_text: Raw job description text
            job_title: Job title (optional)
            company_name: Company name (optional)
            job_location: Job location (optional)
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Dictionary containing the final state and results
        """
        # Create initial state
        initial_state: JobPostParsingState = {
            "job_text": job_text,
            "job_title": job_title,
            "company_name": company_name,
            "job_location": job_location,
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
            "parsed_job": None,
            "processing_metadata": {},
            "started_at": None,
            "completed_at": None,
        }

        # Configure for sync execution
        config = (
            {"configurable": {"thread_id": thread_id or "default"}}
            if self.checkpointer
            else {}
        )

        try:
            # Execute the workflow synchronously
            final_state = self.compiled_graph.invoke(initial_state, config=config)

            return {
                "success": final_state.get("status") == "completed",
                "job_posting": final_state.get("parsed_job"),
                "metadata": final_state.get("processing_metadata", {}),
                "error": final_state.get("error_message"),
                "retry_count": final_state.get("retry_count", 0),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "job_posting": None,
                "metadata": {"error": "workflow_execution_failed"},
                "error": str(e),
                "retry_count": 0,
            }
