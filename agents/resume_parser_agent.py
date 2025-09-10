"""
ResumeParserAgent for orchestrating resume parsing within LangGraph workflow.

This agent handles the workflow orchestration, error handling, and state management
for resume parsing. It uses the ResumeParser tool internally for the actual
document processing while managing retries, validation, and LangGraph state updates.
"""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from tools.resume_parser import ResumeParser, ResumeParsingError


# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
CONFIDENCE_MIN = 0.3
MIN_SECTIONS = 1
MIN_BULLETS = 1


class ResumeParsingState(TypedDict):
    """
    State dictionary for resume parsing workflow.

    This state is passed between nodes in the LangGraph and tracks
    the progress and results of resume parsing operations.
    """

    # Input data
    file_path: Optional[str]
    file_bytes: Optional[bytes]
    file_extension: Optional[str]
    original_filename: Optional[str]

    # Processing status
    status: str
    error_message: Optional[str]
    retry_count: int

    # Results
    parsed_resume: Optional[Dict[str, Any]]  # Serialized Resume
    parser_report: Optional[Dict[str, Any]]  # Parsing report
    processing_metadata: Dict[str, Any]

    # Workflow tracking
    workflow_start_time: Optional[str]
    workflow_end_time: Optional[str]
    current_node: Optional[str]


class ResumeParserAgent:
    """
    Agent for orchestrating resume parsing using LangGraph workflow.

    This agent manages the complete resume parsing pipeline including:
    - File validation and format detection
    - Document processing using ResumeParser tool
    - Error handling and retry logic
    - Result validation and quality assessment
    - State management and workflow tracking

    The agent follows the established agent/tool separation pattern where
    the agent handles orchestration and the tool handles pure parsing logic.
    """

    def __init__(self, max_retries: int = 3, enable_checkpointing: bool = True):
        """
        Initialize the resume parser agent.

        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            enable_checkpointing: Enable LangGraph checkpointing (default: True)
        """
        self.max_retries = max_retries
        self.parser = ResumeParser()

        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver() if enable_checkpointing else None

        # Compile the graph
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for resume parsing."""
        workflow = StateGraph(ResumeParsingState)

        # Define workflow nodes
        workflow.add_node("initialize", self._initialize_parsing)
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("parse_resume", self._parse_resume)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_parsing)

        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "validate_input")

        # Conditional routing from validate_input
        workflow.add_conditional_edges(
            "validate_input",
            self._route_after_validation,
            {"parse": "parse_resume", "error": "handle_error"},
        )

        workflow.add_edge("parse_resume", "validate_results")

        # Conditional routing from validate_results
        workflow.add_conditional_edges(
            "validate_results",
            self._route_after_results,
            {"finalize": "finalize", "retry": "handle_error", "error": "handle_error"},
        )

        # Conditional routing from handle_error
        workflow.add_conditional_edges(
            "handle_error",
            self._route_after_error,
            {"retry": "parse_resume", "finalize": "finalize"},
        )

        workflow.add_edge("finalize", END)

        return workflow

    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse resume file using the LangGraph workflow.

        Args:
            file_path: Path to resume file

        Returns:
            Dictionary containing parsed resume and metadata
        """
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

        initial_state: ResumeParsingState = {
            "file_path": str(file_path),
            "file_bytes": None,
            "file_extension": file_path.suffix,
            "original_filename": file_path.name,
            "status": "initializing",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        # Execute workflow
        result = self.graph.invoke(initial_state)
        return self._format_result(result)

    def parse_bytes(
        self,
        file_bytes: bytes,
        file_extension: str,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse resume from bytes data using the LangGraph workflow.

        Args:
            file_bytes: Raw file bytes
            file_extension: File extension (with or without dot)
            original_filename: Original filename for metadata

        Returns:
            Dictionary containing parsed resume and metadata
        """
        if not file_extension.startswith("."):
            file_extension = "." + file_extension

        initial_state: ResumeParsingState = {
            "file_path": None,
            "file_bytes": file_bytes,
            "file_extension": file_extension,
            "original_filename": original_filename or f"resume{file_extension}",
            "status": "initializing",
            "error_message": None,
            "retry_count": 0,
            "parsed_resume": None,
            "parser_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }

        # Execute workflow
        result = self.graph.invoke(initial_state)
        return self._format_result(result)

    def _initialize_parsing(self, state: ResumeParsingState) -> ResumeParsingState:
        """Initialize parsing workflow and set up tracking."""
        logger.info("Initializing resume parsing workflow")

        state["workflow_start_time"] = datetime.now(timezone.utc).isoformat()
        state["current_node"] = "initialize"
        state["status"] = "initialized"
        state["processing_metadata"] = {
            "agent_version": "1.0",
            "parser_initialized": True,
            "max_retries": self.max_retries,
        }

        return state

    def _validate_input(self, state: ResumeParsingState) -> ResumeParsingState:
        """Validate input file/data before processing."""
        logger.info("Validating input data")

        state["current_node"] = "validate_input"

        try:
            # Check if we have either file path or bytes
            if not state.get("file_path") and not state.get("file_bytes"):
                raise ValueError("No file path or file bytes provided")

            # Validate file extension
            file_ext = state.get("file_extension", "").lower()
            if file_ext not in [".pdf", ".docx", ".doc"]:
                raise ValueError(f"Unsupported file format: {file_ext}")

            # If file path provided, check it exists
            if state.get("file_path"):
                file_path = Path(state["file_path"])
                if not file_path.exists():
                    raise ValueError(f"File not found: {file_path}")

            state["status"] = "input_validated"
            logger.info("Input validation successful")

        except Exception as e:
            state["error_message"] = str(e)
            state["status"] = "input_validation_failed"
            logger.error(f"Input validation failed: {e}")

        return state

    def _parse_resume(self, state: ResumeParsingState) -> ResumeParsingState:
        """Parse resume using the ResumeParser tool."""
        logger.info("Starting resume parsing")

        state["current_node"] = "parse_resume"
        state["status"] = "parsing"

        try:
            if state.get("file_path"):
                # Parse from file path
                resume, report = self.parser.parse_file(state["file_path"])
            elif state.get("file_bytes"):
                # Parse from bytes
                resume, report = self.parser.parse_bytes(
                    state["file_bytes"], state["file_extension"]
                )
            else:
                raise ValueError("No valid input data available")

            # Convert to serializable format
            state["parsed_resume"] = resume.model_dump()
            state["parser_report"] = report
            state["status"] = "parsed"

            logger.info(
                f"Resume parsing completed successfully. "
                f"Found {len(resume.sections)} sections, "
                f"{len(resume.bullets)} bullets, "
                f"{len(resume.skills)} skills"
            )

        except ResumeParsingError as e:
            state["error_message"] = str(e)
            state["status"] = "parsing_failed"
            logger.error(f"Resume parsing failed: {e}")
        except Exception as e:
            state["error_message"] = f"Unexpected error during parsing: {str(e)}"
            state["status"] = "parsing_failed"
            logger.error(f"Unexpected parsing error: {e}")

        return state

    def _validate_results(self, state: ResumeParsingState) -> ResumeParsingState:
        """Validate parsing results for quality and completeness."""
        logger.info("Validating parsing results")

        state["current_node"] = "validate_results"

        try:
            if not state.get("parsed_resume") or not state.get("parser_report"):
                raise ValueError("No parsing results to validate")

            report = state["parser_report"]

            # Quality checks
            confidence = report.get("confidence", 0)
            sections_count = report.get("sections_found", 0)
            bullets_count = report.get("bullets_found", 0)
            text_length = report.get("text_length", 0)

            warnings = []

            # Confidence check
            if confidence < CONFIDENCE_MIN:
                warnings.append(f"Low parsing confidence: {confidence:.2f}")

            # Structure checks
            if sections_count < MIN_SECTIONS:
                warnings.append(f"Few sections found: {sections_count}")

            if bullets_count < MIN_BULLETS:
                warnings.append(f"Few bullets found: {bullets_count}")

            # Content check
            if text_length < 100:
                warnings.append("Very short resume content")

            state["processing_metadata"]["validation_warnings"] = warnings
            state["processing_metadata"]["quality_score"] = confidence

            # Determine if results are acceptable
            if confidence >= CONFIDENCE_MIN:
                state["status"] = "results_validated"
                logger.info(f"Results validation passed (confidence: {confidence:.2f})")
            else:
                state["status"] = "results_validation_failed"
                state["error_message"] = (
                    f"Results validation failed (confidence: {confidence:.2f})"
                )
                logger.warning(
                    f"Results validation failed (confidence: {confidence:.2f})"
                )

        except Exception as e:
            state["error_message"] = str(e)
            state["status"] = "results_validation_failed"
            logger.error(f"Results validation error: {e}")

        return state

    def _handle_error(self, state: ResumeParsingState) -> ResumeParsingState:
        """Handle errors and determine retry strategy."""
        logger.info("Handling parsing error")

        state["current_node"] = "handle_error"
        state["retry_count"] += 1

        error_msg = state.get("error_message", "Unknown error")
        logger.error(f"Handling error (attempt {state['retry_count']}): {error_msg}")

        # Determine if we should retry
        if state["retry_count"] <= self.max_retries:
            # Check if error is retryable
            if "parsing_failed" in state.get("status", ""):
                state["status"] = "retrying"
                logger.info(f"Retrying parsing (attempt {state['retry_count']})")
            else:
                state["status"] = "max_retries_exceeded"
                logger.error("Max retries exceeded")
        else:
            state["status"] = "max_retries_exceeded"
            logger.error("Max retries exceeded")

        return state

    def _finalize_parsing(self, state: ResumeParsingState) -> ResumeParsingState:
        """Finalize parsing workflow and prepare results."""
        logger.info("Finalizing resume parsing workflow")

        state["current_node"] = "finalize"
        state["workflow_end_time"] = datetime.now(timezone.utc).isoformat()

        # Calculate processing time
        if state.get("workflow_start_time"):
            try:
                start_time = datetime.fromisoformat(
                    state["workflow_start_time"].replace("Z", "+00:00")
                )
                end_time = datetime.fromisoformat(
                    state["workflow_end_time"].replace("Z", "+00:00")
                )
                processing_time = (end_time - start_time).total_seconds()
                state["processing_metadata"][
                    "processing_time_seconds"
                ] = processing_time
            except (ValueError, TypeError):
                # Handle cases where datetime parsing fails or naive datetimes
                state["processing_metadata"]["processing_time_seconds"] = 0.0

        # Set final status based on overall workflow success
        current_status = state.get("status", "")
        if state.get("parsed_resume") and current_status not in [
            "max_retries_exceeded",
            "results_validation_failed",
        ]:
            state["status"] = "completed"
            logger.info("Resume parsing workflow completed successfully")
        else:
            state["status"] = "failed"
            logger.error("Resume parsing workflow failed")

        return state

    def _route_after_validation(self, state: ResumeParsingState) -> str:
        """Route workflow after input validation."""
        status = state.get("status", "")
        return "parse" if status == "input_validated" else "error"

    def _route_after_results(self, state: ResumeParsingState) -> str:
        """Route workflow after results validation."""
        status = state.get("status", "")
        if status == "results_validated":
            return "finalize"
        elif (
            status == "results_validation_failed"
            and state["retry_count"] < self.max_retries
        ):
            return "retry"
        else:
            return "error"

    def _route_after_error(self, state: ResumeParsingState) -> str:
        """Route workflow after error handling."""
        status = state.get("status", "")
        return "retry" if status == "retrying" else "finalize"

    def _format_result(self, state: ResumeParsingState) -> Dict[str, Any]:
        """Format final result for return to caller."""
        return {
            "status": state.get("status"),
            "success": state.get("status") == "completed",
            "resume": state.get("parsed_resume"),
            "report": state.get("parser_report"),
            "metadata": state.get("processing_metadata", {}),
            "error": state.get("error_message"),
            "retry_count": state.get("retry_count", 0),
            "processing_time": state.get("processing_metadata", {}).get(
                "processing_time_seconds"
            ),
        }

    # Async interface methods for compatibility
    async def aparse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Async version of parse_file."""
        return self.parse_file(file_path)

    async def aparse_bytes(
        self,
        file_bytes: bytes,
        file_extension: str,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of parse_bytes."""
        return self.parse_bytes(file_bytes, file_extension, original_filename)
