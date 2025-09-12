"""
ResumeGeneratorAgent for orchestrating resume bullet generation within LangGraph workflow.

This agent handles the workflow orchestration, error handling, and state management
for resume bullet generation. It uses the ResumeGenerator tool internally for the actual
generation logic while managing retries, validation, and LangGraph state updates.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from tools.resume_generator import ResumeGenerator, ResumeGenerationError
from tools.evidence_indexer import EvidenceIndexer
from src.schemas.core import JobPosting, Resume


# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MIN_COVERAGE_PCT = 20.0  # Minimum keyword coverage percentage
MIN_BULLETS = 5  # Minimum number of bullets to generate
SIMILARITY_THRESHOLD = 0.8  # Evidence similarity threshold


class ResumeGenerationState(TypedDict):
    """
    State dictionary for resume generation workflow.
    
    This state is passed between nodes in the LangGraph and tracks
    the progress and results of resume bullet generation operations.
    """
    
    # Input data
    job_posting: Optional[Dict[str, Any]]  # Serialized JobPosting
    resume: Optional[Dict[str, Any]]  # Serialized Resume
    max_bullets: int
    similarity_threshold: float
    
    # Processing status
    status: str
    error_message: Optional[str]
    retry_count: int
    
    # Results
    tailored_bullets: Optional[List[Dict[str, Any]]]  # Serialized TailoredBullet list
    generation_metrics: Optional[Dict[str, Any]]  # Serialized GenerationMetrics
    diff_summaries: Optional[List[Dict[str, Any]]]
    processing_metadata: Dict[str, Any]
    
    # Workflow tracking
    workflow_start_time: Optional[str]
    workflow_end_time: Optional[str]
    current_node: Optional[str]


class ResumeGeneratorAgent:
    """
    Agent for orchestrating resume bullet generation using LangGraph workflow.
    
    This agent manages the complete resume generation pipeline including:
    - Input validation for job posting and resume
    - Keyword extraction and bullet mapping
    - Tailored bullet generation with evidence validation
    - Quality assessment and metrics calculation
    - Error handling and retry logic with exponential backoff
    - State management and workflow tracking
    
    The agent follows the established agent/tool separation pattern where
    the agent handles orchestration and the tool handles pure generation logic.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        enable_checkpointing: bool = True,
        evidence_indexer: Optional[EvidenceIndexer] = None,
    ):
        """
        Initialize the resume generator agent.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            enable_checkpointing: Enable LangGraph checkpointing (default: True)
            evidence_indexer: Optional EvidenceIndexer for validation
        """
        self.max_retries = max_retries
        self.evidence_indexer = evidence_indexer
        self.generator = ResumeGenerator(evidence_indexer=evidence_indexer)
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        
        # Compile the graph
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for resume generation."""
        workflow = StateGraph(ResumeGenerationState)
        
        # Define workflow nodes
        workflow.add_node("initialize", self._initialize_generation)
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("extract_keywords", self._extract_keywords)
        workflow.add_node("generate_bullets", self._generate_bullets)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_generation)
        
        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "validate_input")
        
        # Conditional routing from validate_input
        workflow.add_conditional_edges(
            "validate_input",
            self._route_after_validation,
            {"extract": "extract_keywords", "error": "handle_error"},
        )
        
        workflow.add_edge("extract_keywords", "generate_bullets")
        workflow.add_edge("generate_bullets", "validate_results")
        
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
            {"retry": "extract_keywords", "finalize": "finalize"},
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def generate_tailored_bullets(
        self,
        job_posting: JobPosting,
        resume: Resume,
        max_bullets: int = 20,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Generate tailored resume bullets using the LangGraph workflow.
        
        Args:
            job_posting: JobPosting object with requirements and keywords
            resume: Resume object with bullets to tailor
            max_bullets: Maximum number of bullets to generate
            similarity_threshold: Minimum similarity threshold for evidence validation
            
        Returns:
            Dictionary containing generated bullets and metadata
        """
        initial_state: ResumeGenerationState = {
            "job_posting": job_posting.model_dump(),
            "resume": resume.model_dump(),
            "max_bullets": max_bullets,
            "similarity_threshold": similarity_threshold,
            "status": "initializing",
            "error_message": None,
            "retry_count": 0,
            "tailored_bullets": None,
            "generation_metrics": None,
            "diff_summaries": None,
            "processing_metadata": {},
            "workflow_start_time": None,
            "workflow_end_time": None,
            "current_node": None,
        }
        
        # Execute workflow
        result = self.graph.invoke(initial_state)
        return self._format_result(result)
    
    def _initialize_generation(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Initialize generation workflow and set up tracking."""
        logger.info("Initializing resume generation workflow")
        
        state["workflow_start_time"] = datetime.now(timezone.utc).isoformat()
        state["current_node"] = "initialize"
        state["status"] = "initialized"
        state["processing_metadata"] = {
            "agent_version": "1.0",
            "generator_initialized": True,
            "max_retries": self.max_retries,
            "similarity_threshold": state.get("similarity_threshold", SIMILARITY_THRESHOLD),
        }
        
        return state
    
    def _validate_input(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Validate input job posting and resume data."""
        logger.info("Validating input data for generation")
        
        state["current_node"] = "validate_input"
        
        try:
            # Check if we have required data
            if not state.get("job_posting"):
                raise ValueError("No job posting provided")
            
            if not state.get("resume"):
                raise ValueError("No resume provided")
            
            # Validate job posting structure
            job_posting_data = state["job_posting"]
            required_job_fields = ["title", "company", "text", "keywords", "requirements"]
            for field in required_job_fields:
                if field not in job_posting_data:
                    raise ValueError(f"Missing required job posting field: {field}")
            
            # Validate resume structure
            resume_data = state["resume"]
            required_resume_fields = ["raw_text", "bullets", "skills"]
            for field in required_resume_fields:
                if field not in resume_data:
                    raise ValueError(f"Missing required resume field: {field}")
            
            # Check if resume has bullets to work with
            bullets = resume_data.get("bullets", [])
            if not bullets:
                raise ValueError("Resume has no bullets to tailor")
            
            if len(bullets) < 3:
                logger.warning(f"Resume has only {len(bullets)} bullets - may limit generation")
            
            # Check if job has keywords
            keywords = job_posting_data.get("keywords", [])
            if not keywords:
                logger.warning("Job posting has no keywords - will extract from text")
            
            state["status"] = "input_validated"
            logger.info("Input validation successful")
            
        except Exception as e:
            state["error_message"] = str(e)
            state["status"] = "input_validation_failed"
            logger.error(f"Input validation failed: {e}")
        
        return state
    
    def _extract_keywords(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Extract and analyze keywords from job posting."""
        logger.info("Extracting keywords from job posting")
        
        state["current_node"] = "extract_keywords"
        state["status"] = "extracting_keywords"
        
        try:
            # Reconstruct JobPosting object
            job_posting_data = state["job_posting"]
            job_posting = JobPosting(**job_posting_data)
            
            # Extract keywords using generator tool
            keywords = self.generator.extract_keywords_from_job(job_posting)
            
            state["processing_metadata"]["extracted_keywords"] = keywords[:20]  # Store top 20
            state["processing_metadata"]["total_keywords_extracted"] = len(keywords)
            
            if not keywords:
                raise ValueError("No keywords could be extracted from job posting")
            
            state["status"] = "keywords_extracted"
            logger.info(f"Extracted {len(keywords)} keywords successfully")
            
        except ResumeGenerationError as e:
            state["error_message"] = str(e)
            state["status"] = "keyword_extraction_failed"
            logger.error(f"Keyword extraction failed: {e}")
        except Exception as e:
            state["error_message"] = f"Unexpected error during keyword extraction: {str(e)}"
            state["status"] = "keyword_extraction_failed"
            logger.error(f"Unexpected keyword extraction error: {e}")
        
        return state
    
    def _generate_bullets(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Generate tailored bullets using the ResumeGenerator tool."""
        logger.info("Starting bullet generation")
        
        state["current_node"] = "generate_bullets"
        state["status"] = "generating_bullets"
        
        try:
            # Reconstruct objects
            job_posting_data = state["job_posting"]
            resume_data = state["resume"]
            
            job_posting = JobPosting(**job_posting_data)
            resume = Resume(**resume_data)
            
            # Generate bullets using tool
            tailored_bullets, metrics, diff_summaries = self.generator.generate_resume_bullets(
                job_posting=job_posting,
                resume=resume,
                max_bullets=state.get("max_bullets", 20),
            )
            
            # Store results
            state["tailored_bullets"] = [bullet.model_dump() for bullet in tailored_bullets]
            state["generation_metrics"] = {
                "total_keywords": metrics.total_keywords,
                "covered_keywords": metrics.covered_keywords,
                "total_bullets_generated": metrics.total_bullets_generated,
                "bullets_above_threshold": metrics.bullets_above_threshold,
                "average_similarity_score": metrics.average_similarity_score,
                "keyword_coverage_percentage": metrics.keyword_coverage_percentage,
            }
            state["diff_summaries"] = diff_summaries
            
            state["status"] = "bullets_generated"
            
            logger.info(
                f"Bullet generation completed. Generated {len(tailored_bullets)} bullets "
                f"with {metrics.keyword_coverage_percentage:.1f}% keyword coverage"
            )
            
        except ResumeGenerationError as e:
            state["error_message"] = str(e)
            state["status"] = "bullet_generation_failed"
            logger.error(f"Bullet generation failed: {e}")
        except Exception as e:
            state["error_message"] = f"Unexpected error during bullet generation: {str(e)}"
            state["status"] = "bullet_generation_failed"
            logger.error(f"Unexpected bullet generation error: {e}")
        
        return state
    
    def _validate_results(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Validate generation results for quality and completeness."""
        logger.info("Validating generation results")
        
        state["current_node"] = "validate_results"
        
        try:
            if not state.get("tailored_bullets") or not state.get("generation_metrics"):
                raise ValueError("No generation results to validate")
            
            metrics = state["generation_metrics"]
            bullets = state["tailored_bullets"]
            
            # Quality checks
            coverage_pct = metrics.get("keyword_coverage_percentage", 0)
            bullets_count = len(bullets)
            bullets_above_threshold = metrics.get("bullets_above_threshold", 0)
            avg_similarity = metrics.get("average_similarity_score", 0)
            
            warnings = []
            
            # Coverage check
            if coverage_pct < MIN_COVERAGE_PCT:
                warnings.append(f"Low keyword coverage: {coverage_pct:.1f}%")
            
            # Bullet count check
            if bullets_count < MIN_BULLETS:
                warnings.append(f"Few bullets generated: {bullets_count}")
            
            # Similarity threshold check
            threshold_ratio = bullets_above_threshold / bullets_count if bullets_count > 0 else 0
            if threshold_ratio < 0.8:  # 80% should meet threshold
                warnings.append(f"Low similarity ratio: {threshold_ratio:.1f}")
            
            # Average similarity check
            if avg_similarity < state.get("similarity_threshold", SIMILARITY_THRESHOLD):
                warnings.append(f"Low average similarity: {avg_similarity:.3f}")
            
            state["processing_metadata"]["validation_warnings"] = warnings
            state["processing_metadata"]["quality_metrics"] = {
                "coverage_percentage": coverage_pct,
                "bullets_count": bullets_count,
                "threshold_ratio": threshold_ratio,
                "average_similarity": avg_similarity,
            }
            
            # Determine if results are acceptable
            # Accept if we have minimum bullets and reasonable coverage/similarity
            if (
                bullets_count >= MIN_BULLETS
                and coverage_pct >= MIN_COVERAGE_PCT * 0.5  # Allow lower threshold
                and avg_similarity >= state.get("similarity_threshold", SIMILARITY_THRESHOLD) * 0.9  # Slight tolerance
            ):
                state["status"] = "results_validated"
                logger.info(
                    f"Results validation passed - {bullets_count} bullets, "
                    f"{coverage_pct:.1f}% coverage, {avg_similarity:.3f} similarity"
                )
            else:
                state["status"] = "results_validation_failed"
                state["error_message"] = (
                    f"Results validation failed - coverage: {coverage_pct:.1f}%, "
                    f"bullets: {bullets_count}, similarity: {avg_similarity:.3f}"
                )
                logger.warning(
                    f"Results validation failed - coverage: {coverage_pct:.1f}%, "
                    f"bullets: {bullets_count}, similarity: {avg_similarity:.3f}"
                )
            
        except Exception as e:
            state["error_message"] = str(e)
            state["status"] = "results_validation_failed"
            logger.error(f"Results validation error: {e}")
        
        return state
    
    def _handle_error(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Handle errors and determine retry strategy."""
        logger.info("Handling generation error")
        
        state["current_node"] = "handle_error"
        state["retry_count"] += 1
        
        error_msg = state.get("error_message", "Unknown error")
        logger.error(f"Handling error (attempt {state['retry_count']}): {error_msg}")
        
        # Determine if we should retry
        if state["retry_count"] <= self.max_retries:
            # Check if error is retryable
            retryable_statuses = [
                "keyword_extraction_failed",
                "bullet_generation_failed",
                "results_validation_failed",
            ]
            
            current_status = state.get("status", "")
            if any(status in current_status for status in retryable_statuses):
                state["status"] = "retrying"
                logger.info(f"Retrying generation (attempt {state['retry_count']})")
                
                # Adjust parameters for retry if validation failed
                if "results_validation_failed" in current_status:
                    # Lower similarity threshold slightly for retry
                    current_threshold = state.get("similarity_threshold", SIMILARITY_THRESHOLD)
                    state["similarity_threshold"] = max(0.7, current_threshold - 0.05)
                    logger.info(f"Adjusted similarity threshold to {state['similarity_threshold']}")
            else:
                state["status"] = "max_retries_exceeded"
                logger.error("Error not retryable")
        else:
            state["status"] = "max_retries_exceeded"
            logger.error("Max retries exceeded")
        
        return state
    
    def _finalize_generation(self, state: ResumeGenerationState) -> ResumeGenerationState:
        """Finalize generation workflow and prepare results."""
        logger.info("Finalizing resume generation workflow")
        
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
        if state.get("tailored_bullets") and current_status not in [
            "max_retries_exceeded",
            "input_validation_failed",
        ]:
            state["status"] = "completed"
            logger.info("Resume generation workflow completed successfully")
        else:
            state["status"] = "failed"
            logger.error("Resume generation workflow failed")
        
        return state
    
    def _route_after_validation(self, state: ResumeGenerationState) -> str:
        """Route workflow after input validation."""
        status = state.get("status", "")
        return "extract" if status == "input_validated" else "error"
    
    def _route_after_results(self, state: ResumeGenerationState) -> str:
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
    
    def _route_after_error(self, state: ResumeGenerationState) -> str:
        """Route workflow after error handling."""
        status = state.get("status", "")
        return "retry" if status == "retrying" else "finalize"
    
    def _format_result(self, state: ResumeGenerationState) -> Dict[str, Any]:
        """Format final result for return to caller."""
        return {
            "status": state.get("status"),
            "success": state.get("status") == "completed",
            "tailored_bullets": state.get("tailored_bullets"),
            "metrics": state.get("generation_metrics"),
            "diff_summaries": state.get("diff_summaries"),
            "metadata": state.get("processing_metadata", {}),
            "error": state.get("error_message"),
            "retry_count": state.get("retry_count", 0),
            "processing_time": state.get("processing_metadata", {}).get("processing_time_seconds"),
        }
    
    # Async interface methods for compatibility
    async def agenerate_tailored_bullets(
        self,
        job_posting: JobPosting,
        resume: Resume,
        max_bullets: int = 20,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """Async version of generate_tailored_bullets."""
        return self.generate_tailored_bullets(
            job_posting, resume, max_bullets, similarity_threshold
        )