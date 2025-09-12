"""
ResumeValidatorAgent for orchestrating resume bullet validation within LangGraph workflow.

This agent handles the workflow orchestration, error handling, and state management
for resume bullet validation. It uses the ResumeValidator tool internally for the actual
validation logic while managing evidence indexing, batch processing, and LangGraph state updates.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from tools.resume_validator import (
    ResumeValidator,
    ResumeValidationError,
    ValidationReport,
    ValidationStatus,
)
from tools.evidence_indexer import EvidenceIndexer, EvidenceIndexingError
from src.schemas.core import TailoredBullet, Resume


# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MIN_SUCCESS_RATE = 70.0  # Minimum validation success rate percentage
MIN_EVIDENCE_COVERAGE = 60.0  # Minimum evidence coverage percentage
SIMILARITY_THRESHOLD = 0.8  # Default similarity threshold


class ResumeValidationState(TypedDict):
    """
    State dictionary for resume validation workflow.
    
    This state is passed between nodes in the LangGraph and tracks
    the progress and results of resume bullet validation operations.
    """
    
    # Input data
    tailored_bullets: Optional[List[Dict[str, Any]]]  # Serialized TailoredBullet list
    resume: Optional[Dict[str, Any]]  # Serialized Resume (for evidence indexing)
    similarity_threshold: float
    batch_size: int
    
    # Processing status
    status: str
    error_message: Optional[str]
    retry_count: int
    
    # Evidence indexing
    evidence_indexed: bool
    evidence_stats: Optional[Dict[str, Any]]
    
    # Results
    validation_report: Optional[Dict[str, Any]]  # Serialized ValidationReport
    processing_metadata: Dict[str, Any]
    
    # Workflow tracking
    workflow_start_time: Optional[str]
    workflow_end_time: Optional[str]
    current_node: Optional[str]


class ResumeValidatorAgent:
    """
    Agent for orchestrating resume bullet validation using LangGraph workflow.
    
    This agent manages the complete resume validation pipeline including:
    - Evidence indexing setup and verification
    - Batch processing of tailored bullets for validation
    - Evidence-based similarity checking against original resume content
    - Comprehensive validation reporting with actionable recommendations
    - Error handling and retry logic for evidence indexing issues
    - State management and workflow tracking
    
    The agent follows the established agent/tool separation pattern where
    the agent handles orchestration and the tool handles pure validation logic.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        enable_checkpointing: bool = True,
        evidence_indexer: Optional[EvidenceIndexer] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the resume validator agent.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            enable_checkpointing: Enable LangGraph checkpointing (default: True)
            evidence_indexer: Optional EvidenceIndexer for validation
            similarity_threshold: Default similarity threshold for validation
        """
        self.max_retries = max_retries
        self.similarity_threshold = similarity_threshold
        
        # Initialize evidence indexer and validator
        self.evidence_indexer = evidence_indexer or EvidenceIndexer(
            similarity_threshold=similarity_threshold
        )
        self.validator = ResumeValidator(
            similarity_threshold=similarity_threshold,
            evidence_indexer=self.evidence_indexer,
        )
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        
        # Compile the graph
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for resume validation."""
        workflow = StateGraph(ResumeValidationState)
        
        # Define workflow nodes
        workflow.add_node("initialize", self._initialize_validation)
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("setup_evidence", self._setup_evidence)
        workflow.add_node("validate_bullets", self._validate_bullets)
        workflow.add_node("analyze_results", self._analyze_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_validation)
        
        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "validate_input")
        
        # Conditional routing from validate_input
        workflow.add_conditional_edges(
            "validate_input",
            self._route_after_validation,
            {"setup": "setup_evidence", "error": "handle_error"},
        )
        
        workflow.add_edge("setup_evidence", "validate_bullets")
        workflow.add_edge("validate_bullets", "analyze_results")
        
        # Conditional routing from analyze_results
        workflow.add_conditional_edges(
            "analyze_results",
            self._route_after_analysis,
            {"finalize": "finalize", "retry": "handle_error", "error": "handle_error"},
        )
        
        # Conditional routing from handle_error
        workflow.add_conditional_edges(
            "handle_error",
            self._route_after_error,
            {"retry": "setup_evidence", "finalize": "finalize"},
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def validate_bullets(
        self,
        tailored_bullets: List[TailoredBullet],
        resume: Resume,
        similarity_threshold: Optional[float] = None,
        batch_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Validate tailored bullets using the LangGraph workflow.
        
        Args:
            tailored_bullets: List of TailoredBullet objects to validate
            resume: Resume object for evidence indexing
            similarity_threshold: Optional similarity threshold override
            batch_size: Number of bullets to process in each batch
            
        Returns:
            Dictionary containing validation results and metadata
        """
        threshold = similarity_threshold or self.similarity_threshold
        
        initial_state: ResumeValidationState = {
            "tailored_bullets": [bullet.model_dump() for bullet in tailored_bullets],
            "resume": resume.model_dump(),
            "similarity_threshold": threshold,
            "batch_size": batch_size,
            "status": "initializing",
            "error_message": None,
            "retry_count": 0,
            "evidence_indexed": False,
            "evidence_stats": None,
            "validation_report": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }
        
        # Execute workflow
        result = self.graph.invoke(initial_state)
        return self._format_result(result)
    
    def _initialize_validation(self, state: ResumeValidationState) -> ResumeValidationState:
        """Initialize validation workflow and set up tracking."""
        logger.info("Initializing resume validation workflow")
        
        state["workflow_start_time"] = datetime.now(timezone.utc).isoformat()
        state["current_node"] = "initialize"
        state["status"] = "initialized"
        state["processing_metadata"] = {
            "agent_version": "1.0",
            "validator_initialized": True,
            "max_retries": self.max_retries,
            "similarity_threshold": state.get("similarity_threshold", SIMILARITY_THRESHOLD),
        }
        
        return state
    
    def _validate_input(self, state: ResumeValidationState) -> ResumeValidationState:
        """Validate input tailored bullets and resume data."""
        logger.info("Validating input data for validation")
        
        state["current_node"] = "validate_input"
        
        try:
            # Check if we have required data
            if not state.get("tailored_bullets"):
                raise ValueError("No tailored bullets provided")
            
            if not state.get("resume"):
                raise ValueError("No resume provided for evidence indexing")
            
            # Validate bullets structure
            bullets_data = state["tailored_bullets"]
            if not isinstance(bullets_data, list) or len(bullets_data) == 0:
                raise ValueError("No bullets to validate")
            
            # Check bullet structure
            required_bullet_fields = ["text", "similarity_score", "jd_keywords_covered"]
            for i, bullet_data in enumerate(bullets_data):
                for field in required_bullet_fields:
                    if field not in bullet_data:
                        raise ValueError(f"Bullet {i} missing required field: {field}")
            
            # Validate resume structure
            resume_data = state["resume"]
            required_resume_fields = ["raw_text", "bullets"]
            for field in required_resume_fields:
                if field not in resume_data:
                    raise ValueError(f"Missing required resume field: {field}")
            
            # Check if resume has bullets for evidence
            resume_bullets = resume_data.get("bullets", [])
            if not resume_bullets:
                raise ValueError("Resume has no bullets for evidence indexing")
            
            state["processing_metadata"]["bullets_to_validate"] = len(bullets_data)
            state["processing_metadata"]["resume_bullets_available"] = len(resume_bullets)
            
            state["status"] = "input_validated"
            logger.info(
                f"Input validation successful - {len(bullets_data)} bullets to validate, "
                f"{len(resume_bullets)} evidence bullets available"
            )
            
        except Exception as e:
            state["error_message"] = str(e)
            state["status"] = "input_validation_failed"
            logger.error(f"Input validation failed: {e}")
        
        return state
    
    def _setup_evidence(self, state: ResumeValidationState) -> ResumeValidationState:
        """Set up evidence indexing for validation."""
        logger.info("Setting up evidence indexing")
        
        state["current_node"] = "setup_evidence"
        state["status"] = "setting_up_evidence"
        
        try:
            # Check if evidence is already indexed
            current_stats = self.evidence_indexer.get_collection_stats()
            
            if current_stats.get("total_items", 0) == 0:
                # Need to index the resume
                logger.info("No evidence indexed, indexing resume")
                
                resume_data = state["resume"]
                resume = Resume(**resume_data)
                
                indexing_result = self.evidence_indexer.index_resume(resume)
                state["evidence_stats"] = indexing_result
                
                logger.info(
                    f"Indexed {indexing_result['items_indexed']} evidence items "
                    f"({indexing_result['bullets_indexed']} bullets, "
                    f"{indexing_result['skills_indexed']} skills)"
                )
            else:
                # Evidence already indexed
                state["evidence_stats"] = current_stats
                logger.info(f"Using existing evidence index with {current_stats['total_items']} items")
            
            state["evidence_indexed"] = True
            state["status"] = "evidence_ready"
            
        except EvidenceIndexingError as e:
            state["error_message"] = f"Evidence indexing failed: {str(e)}"
            state["status"] = "evidence_setup_failed"
            logger.error(f"Evidence setup failed: {e}")
        except Exception as e:
            state["error_message"] = f"Unexpected error during evidence setup: {str(e)}"
            state["status"] = "evidence_setup_failed"
            logger.error(f"Unexpected evidence setup error: {e}")
        
        return state
    
    def _validate_bullets(self, state: ResumeValidationState) -> ResumeValidationState:
        """Validate bullets using the ResumeValidator tool."""
        logger.info("Starting bullet validation")
        
        state["current_node"] = "validate_bullets"
        state["status"] = "validating_bullets"
        
        try:
            # Reconstruct TailoredBullet objects
            bullets_data = state["tailored_bullets"]
            bullets = [TailoredBullet(**bullet_data) for bullet_data in bullets_data]
            
            # Reconstruct Resume object (may be needed for additional context)
            resume_data = state["resume"]
            resume = Resume(**resume_data)
            
            # Run validation using tool
            validation_report = self.validator.validate_bullets(bullets, resume)
            
            # Store results
            state["validation_report"] = {
                "total_bullets": validation_report.total_bullets,
                "valid_bullets": validation_report.valid_bullets,
                "rejected_bullets": validation_report.rejected_bullets,
                "needs_edit_bullets": validation_report.needs_edit_bullets,
                "error_bullets": validation_report.error_bullets,
                "overall_evidence_score": validation_report.overall_evidence_score,
                "evidence_coverage_percentage": validation_report.evidence_coverage_percentage,
                "validation_success_rate": validation_report.validation_success_rate,
                "needs_review_count": validation_report.needs_review_count,
                "summary_recommendations": validation_report.summary_recommendations,
                # Store detailed results as serializable data
                "validation_results": [
                    {
                        "bullet_text": result.bullet.text,
                        "status": result.status.value,
                        "best_similarity_score": result.best_similarity_score,
                        "confidence_score": result.confidence_score,
                        "validation_notes": result.validation_notes,
                        "recommended_edits": result.recommended_edits,
                        "evidence_count": len(result.evidence_matches),
                        "keywords_covered": result.bullet.jd_keywords_covered,
                    }
                    for result in validation_report.validation_results
                ],
            }
            
            state["status"] = "bullets_validated"
            
            logger.info(
                f"Validation completed - {validation_report.valid_bullets}/{validation_report.total_bullets} "
                f"bullets valid ({validation_report.validation_success_rate:.1f}% success rate)"
            )
            
        except ResumeValidationError as e:
            state["error_message"] = str(e)
            state["status"] = "bullet_validation_failed"
            logger.error(f"Bullet validation failed: {e}")
        except Exception as e:
            state["error_message"] = f"Unexpected error during bullet validation: {str(e)}"
            state["status"] = "bullet_validation_failed"
            logger.error(f"Unexpected bullet validation error: {e}")
        
        return state
    
    def _analyze_results(self, state: ResumeValidationState) -> ResumeValidationState:
        """Analyze validation results for quality and completeness."""
        logger.info("Analyzing validation results")
        
        state["current_node"] = "analyze_results"
        
        try:
            if not state.get("validation_report"):
                raise ValueError("No validation results to analyze")
            
            report = state["validation_report"]
            
            # Quality checks
            success_rate = report.get("validation_success_rate", 0)
            evidence_coverage = report.get("evidence_coverage_percentage", 0)
            valid_bullets = report.get("valid_bullets", 0)
            total_bullets = report.get("total_bullets", 0)
            needs_review = report.get("needs_review_count", 0)
            
            analysis_notes = []
            
            # Success rate check
            if success_rate < MIN_SUCCESS_RATE:
                analysis_notes.append(f"Low validation success rate: {success_rate:.1f}%")
            
            # Evidence coverage check
            if evidence_coverage < MIN_EVIDENCE_COVERAGE:
                analysis_notes.append(f"Low evidence coverage: {evidence_coverage:.1f}%")
            
            # Absolute validation check
            if valid_bullets == 0 and total_bullets > 0:
                analysis_notes.append("No bullets passed validation")
            
            # High needs-review rate
            review_rate = (needs_review / total_bullets * 100) if total_bullets > 0 else 0
            if review_rate > 50:
                analysis_notes.append(f"High needs-review rate: {review_rate:.1f}%")
            
            state["processing_metadata"]["analysis_notes"] = analysis_notes
            state["processing_metadata"]["quality_metrics"] = {
                "success_rate": success_rate,
                "evidence_coverage": evidence_coverage,
                "review_rate": review_rate,
                "valid_count": valid_bullets,
                "total_count": total_bullets,
            }
            
            # Determine if results are acceptable
            # Accept if we have reasonable success rate and evidence coverage
            if (
                success_rate >= MIN_SUCCESS_RATE * 0.7  # Allow 70% of minimum
                and evidence_coverage >= MIN_EVIDENCE_COVERAGE * 0.7  # Allow 70% of minimum
                and valid_bullets > 0  # At least some bullets passed
            ):
                state["status"] = "results_analyzed"
                logger.info(
                    f"Results analysis passed - {success_rate:.1f}% success rate, "
                    f"{evidence_coverage:.1f}% evidence coverage"
                )
            else:
                state["status"] = "results_analysis_failed"
                state["error_message"] = (
                    f"Results analysis failed - success rate: {success_rate:.1f}%, "
                    f"evidence coverage: {evidence_coverage:.1f}%"
                )
                logger.warning(
                    f"Results analysis failed - success rate: {success_rate:.1f}%, "
                    f"evidence coverage: {evidence_coverage:.1f}%"
                )
            
        except Exception as e:
            state["error_message"] = str(e)
            state["status"] = "results_analysis_failed"
            logger.error(f"Results analysis error: {e}")
        
        return state
    
    def _handle_error(self, state: ResumeValidationState) -> ResumeValidationState:
        """Handle errors and determine retry strategy."""
        logger.info("Handling validation error")
        
        state["current_node"] = "handle_error"
        state["retry_count"] += 1
        
        error_msg = state.get("error_message", "Unknown error")
        logger.error(f"Handling error (attempt {state['retry_count']}): {error_msg}")
        
        # Determine if we should retry
        if state["retry_count"] <= self.max_retries:
            # Check if error is retryable
            retryable_statuses = [
                "evidence_setup_failed",
                "bullet_validation_failed",
                "results_analysis_failed",
            ]
            
            if any(status in state.get("status", "") for status in retryable_statuses):
                state["status"] = "retrying"
                logger.info(f"Retrying validation (attempt {state['retry_count']})")
                
                # Adjust parameters for retry if needed
                if "results_analysis_failed" in state.get("status", ""):
                    # Lower thresholds slightly for retry
                    current_threshold = state.get("similarity_threshold", SIMILARITY_THRESHOLD)
                    state["similarity_threshold"] = max(0.7, current_threshold - 0.05)
                    logger.info(f"Adjusted similarity threshold to {state['similarity_threshold']}")
                    
                    # Update validator threshold
                    self.validator.similarity_threshold = state["similarity_threshold"]
            else:
                state["status"] = "max_retries_exceeded"
                logger.error("Error not retryable")
        else:
            state["status"] = "max_retries_exceeded"
            logger.error("Max retries exceeded")
        
        return state
    
    def _finalize_validation(self, state: ResumeValidationState) -> ResumeValidationState:
        """Finalize validation workflow and prepare results."""
        logger.info("Finalizing resume validation workflow")
        
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
                state["processing_metadata"]["processing_time_seconds"] = processing_time
            except (ValueError, TypeError):
                state["processing_metadata"]["processing_time_seconds"] = 0.0
        
        # Set final status based on overall workflow success
        current_status = state.get("status", "")
        if state.get("validation_report") and current_status not in [
            "max_retries_exceeded",
            "input_validation_failed",
        ]:
            state["status"] = "completed"
            logger.info("Resume validation workflow completed successfully")
        else:
            state["status"] = "failed"
            logger.error("Resume validation workflow failed")
        
        return state
    
    def _route_after_validation(self, state: ResumeValidationState) -> str:
        """Route workflow after input validation."""
        status = state.get("status", "")
        return "setup" if status == "input_validated" else "error"
    
    def _route_after_analysis(self, state: ResumeValidationState) -> str:
        """Route workflow after results analysis."""
        status = state.get("status", "")
        if status == "results_analyzed":
            return "finalize"
        elif (
            status == "results_analysis_failed"
            and state["retry_count"] < self.max_retries
        ):
            return "retry"
        else:
            return "error"
    
    def _route_after_error(self, state: ResumeValidationState) -> str:
        """Route workflow after error handling."""
        status = state.get("status", "")
        return "retry" if status == "retrying" else "finalize"
    
    def _format_result(self, state: ResumeValidationState) -> Dict[str, Any]:
        """Format final result for return to caller."""
        return {
            "status": state.get("status"),
            "success": state.get("status") == "completed",
            "validation_report": state.get("validation_report"),
            "evidence_stats": state.get("evidence_stats"),
            "metadata": state.get("processing_metadata", {}),
            "error": state.get("error_message"),
            "retry_count": state.get("retry_count", 0),
            "processing_time": state.get("processing_metadata", {}).get("processing_time_seconds"),
        }
    
    # Async interface methods for compatibility
    async def avalidate_bullets(
        self,
        tailored_bullets: List[TailoredBullet],
        resume: Resume,
        similarity_threshold: Optional[float] = None,
        batch_size: int = 20,
    ) -> Dict[str, Any]:
        """Async version of validate_bullets."""
        return self.validate_bullets(tailored_bullets, resume, similarity_threshold, batch_size)