"""
CompensationAnalystAgent for orchestrating compensation analysis within LangGraph workflow.

This agent handles the workflow orchestration, error handling, and state management
for compensation analysis operations. It uses the CompensationAnalysisTool internally 
for the actual SOC mapping and salary data retrieval while managing retries, 
validation, and LangGraph state updates.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from tools.compensation_analysis import CompensationAnalysisTool
from src.schemas.core import CompBand

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MIN_CONFIDENCE_THRESHOLD = 0.6
RETRY_CONFIDENCE_THRESHOLD = 0.4
MAX_RETRY_ATTEMPTS = 3


class CompensationAnalysisState(TypedDict):
    """
    State dictionary for compensation analysis workflow.

    This state is passed between nodes in the LangGraph and tracks
    the progress and results of compensation analysis operations.
    """

    # Input data
    job_title: str
    location: Optional[str]  # Geographic location for analysis
    
    # Processing status
    status: str
    error_message: Optional[str]
    retry_count: int
    
    # Intermediate results
    soc_mapping: Optional[Dict[str, Any]]  # SOC code and confidence
    validation_report: Optional[Dict[str, Any]]
    geographic_info: Optional[Dict[str, Any]]
    
    # Final results
    comp_band: Optional[Dict[str, Any]]  # Serialized CompBand
    analysis_metadata: Dict[str, Any]
    
    # Timestamps (as ISO strings for JSON serialization)
    started_at: Optional[str]
    completed_at: Optional[str]


class CompensationAnalystAgent:
    """
    Agent for orchestrating compensation analysis workflows.

    This agent manages the complete workflow for analyzing job compensation,
    including SOC mapping, salary data retrieval, error handling, retries,
    and LangGraph state management. It uses the CompensationAnalysisTool
    for the actual analysis work.
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        max_retries: int = MAX_RETRY_ATTEMPTS,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the CompensationAnalystAgent.

        Args:
            min_confidence: Minimum confidence threshold for SOC mappings
            max_retries: Maximum retry attempts for failed operations
            enable_checkpoints: Whether to enable LangGraph checkpointing
        """
        self.min_confidence = min_confidence
        self.max_retries = max_retries

        # Initialize the compensation analysis tool
        self.compensation_tool = CompensationAnalysisTool()

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
        Build the LangGraph workflow for compensation analysis.

        Returns:
            Configured StateGraph for the compensation analysis workflow
        """
        # Create the graph with our state model
        workflow = StateGraph(CompensationAnalysisState)

        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("analyze_compensation", self._analyze_compensation)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_results)

        # Define the workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "validate_input")

        # Conditional routing from input validation
        workflow.add_conditional_edges(
            "validate_input",
            self._input_validation_routing,
            {"proceed": "analyze_compensation", "retry": "handle_error", "failed": END},
        )

        # Conditional routing from analysis
        workflow.add_conditional_edges(
            "analyze_compensation",
            self._analysis_routing,
            {"validate": "validate_results", "retry": "handle_error", "failed": END},
        )

        # From results validation
        workflow.add_conditional_edges(
            "validate_results",
            self._validation_routing,
            {"success": "finalize", "retry": "handle_error"},
        )

        # Error handling flow
        workflow.add_conditional_edges(
            "handle_error",
            self._error_handling_routing,
            {"retry": "validate_input", "failed": END},
        )

        # Finalization
        workflow.add_edge("finalize", END)

        return workflow

    def _initialize_analysis(self, state: CompensationAnalysisState) -> CompensationAnalysisState:
        """
        Initialize the compensation analysis workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated state with initialization metadata
        """
        logger.info(
            f"Initializing compensation analysis for: {state.get('job_title', 'Unknown Job')} "
            f"in {state.get('location', 'national')}"
        )

        new_state = state.copy()
        new_state.update(
            {
                "status": "initializing",
                "retry_count": 0,
                "error_message": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "analysis_metadata": {
                    "agent_config": {
                        "min_confidence": self.min_confidence,
                        "max_retries": self.max_retries,
                    },
                    "workflow_version": "1.0",
                    "supported_geographies": self.compensation_tool.get_available_geographies(),
                },
            }
        )
        return new_state

    def _validate_input(self, state: CompensationAnalysisState) -> CompensationAnalysisState:
        """
        Validate input parameters and check SOC mapping availability.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation results
        """
        new_state = state.copy()
        new_state["status"] = "validating_input"

        job_title = state.get("job_title", "")
        location = state.get("location")

        try:
            # Use tool's validation method
            validation = self.compensation_tool.validate_inputs(job_title, location)
            
            new_state["validation_report"] = validation
            
            if not validation["valid_job_title"]:
                new_state["error_message"] = "Invalid or missing job title"
                new_state["status"] = "validation_failed"
                return new_state
                
            # Check SOC mapping confidence
            soc_confidence = validation.get("soc_mapping_confidence", 0.0)
            if soc_confidence < RETRY_CONFIDENCE_THRESHOLD:
                new_state["error_message"] = f"No SOC mapping found for job title: {job_title}"
                new_state["status"] = "validation_failed"
                return new_state
                
            # Store mapping information
            new_state["soc_mapping"] = {
                "soc_code": validation.get("soc_code"),
                "confidence": soc_confidence,
                "job_title": job_title,
            }
            
            new_state["geographic_info"] = {
                "requested_location": location,
                "geographic_code": validation["geographic_code"],
                "warnings": validation.get("warnings", []),
            }
            
            new_state["status"] = "input_validated"
            logger.info(f"Input validation successful for {job_title} (SOC confidence: {soc_confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            new_state["error_message"] = f"Input validation error: {str(e)}"
            new_state["status"] = "validation_failed"

        return new_state

    def _analyze_compensation(self, state: CompensationAnalysisState) -> CompensationAnalysisState:
        """
        Conduct compensation analysis using the CompensationAnalysisTool.

        Args:
            state: Current workflow state

        Returns:
            Updated state with analysis results
        """
        new_state = state.copy()
        new_state["status"] = "analyzing"

        job_title = state["job_title"]
        location = state.get("location")

        try:
            logger.info(f"Analyzing compensation for: {job_title} in {location or 'national'}")
            
            # Run the compensation analysis
            comp_band = self.compensation_tool.analyze_compensation(job_title, location)
            
            # Serialize CompBand for state storage
            comp_band_dict = comp_band.model_dump()
            # Convert datetime to string for JSON serialization
            comp_band_dict["as_of"] = comp_band.as_of.isoformat()
            # Convert HttpUrl to string
            comp_band_dict["sources"] = [str(url) for url in comp_band.sources]
            
            new_state["comp_band"] = comp_band_dict
            new_state["status"] = "analysis_completed"
            
            # Update metadata with analysis details
            new_state["analysis_metadata"].update({
                "soc_code": comp_band.occupation_code,
                "geography_used": comp_band.geography,
                "data_currency": comp_band_dict["as_of"],
                "percentiles_available": {
                    "p25": comp_band.p25 is not None,
                    "p50": comp_band.p50 is not None,
                    "p75": comp_band.p75 is not None,
                },
            })
            
            logger.info(f"Compensation analysis completed: ${comp_band.p50:,.0f} median in {comp_band.geography}")
            
        except Exception as e:
            logger.error(f"Compensation analysis failed: {str(e)}")
            new_state["error_message"] = f"Analysis failed: {str(e)}"
            new_state["status"] = "analysis_failed"

        return new_state

    def _validate_results(self, state: CompensationAnalysisState) -> CompensationAnalysisState:
        """
        Validate analysis results for completeness and quality.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation status
        """
        new_state = state.copy()
        new_state["status"] = "validating_results"

        comp_band = state.get("comp_band")
        if not comp_band:
            new_state["error_message"] = "No compensation data available for validation"
            new_state["status"] = "results_validation_failed"
            return new_state

        try:
            # Validate data completeness
            validation_issues = []
            
            # Check for salary data
            if not any([comp_band.get("p25"), comp_band.get("p50"), comp_band.get("p75")]):
                validation_issues.append("No salary percentiles available")
                
            # Check data recency (should be from 2024)
            as_of_str = comp_band.get("as_of", "")
            if "2024" not in as_of_str:
                validation_issues.append("Salary data may be outdated")
                
            # Check SOC mapping confidence if available
            soc_mapping = state.get("soc_mapping", {})
            confidence = soc_mapping.get("confidence", 0.0)
            if confidence < self.min_confidence:
                validation_issues.append(f"Low SOC mapping confidence: {confidence:.2f}")

            # Update validation report
            validation_report = state.get("validation_report", {})
            validation_report.update({
                "results_validation": {
                    "issues": validation_issues,
                    "data_completeness_score": self._calculate_completeness_score(comp_band),
                    "overall_quality": "high" if len(validation_issues) == 0 else "medium" if len(validation_issues) <= 2 else "low",
                }
            })
            new_state["validation_report"] = validation_report

            if validation_issues:
                logger.warning(f"Validation issues found: {', '.join(validation_issues)}")
                # Don't fail on validation issues, just log them
                
            new_state["status"] = "results_validated"
            logger.info("Results validation completed successfully")

        except Exception as e:
            logger.error(f"Results validation failed: {str(e)}")
            new_state["error_message"] = f"Results validation error: {str(e)}"
            new_state["status"] = "results_validation_failed"

        return new_state

    def _handle_error(self, state: CompensationAnalysisState) -> CompensationAnalysisState:
        """
        Handle errors and determine retry strategy.

        Args:
            state: Current workflow state

        Returns:
            Updated state with error handling decisions
        """
        new_state = state.copy()
        new_state["retry_count"] = state.get("retry_count", 0) + 1
        
        error_msg = state.get("error_message", "Unknown error")
        retry_count = new_state["retry_count"]
        
        logger.warning(f"Handling error (attempt {retry_count}): {error_msg}")
        
        # Determine if we should retry based on error type and retry count
        if retry_count >= self.max_retries:
            new_state["status"] = "max_retries_exceeded"
            logger.error(f"Max retries ({self.max_retries}) exceeded for compensation analysis")
        else:
            new_state["status"] = "retrying"
            logger.info(f"Retrying compensation analysis (attempt {retry_count + 1})")
            
        return new_state

    def _finalize_results(self, state: CompensationAnalysisState) -> CompensationAnalysisState:
        """
        Finalize the compensation analysis results.

        Args:
            state: Current workflow state

        Returns:
            Final state with completion metadata
        """
        new_state = state.copy()
        new_state["status"] = "completed"
        new_state["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Add final summary to metadata
        comp_band = state.get("comp_band", {})
        new_state["analysis_metadata"].update({
            "completion_summary": {
                "job_title": state["job_title"],
                "location": state.get("location", "national"),
                "soc_code": comp_band.get("occupation_code"),
                "median_salary": comp_band.get("p50"),
                "currency": comp_band.get("currency", "USD"),
                "data_source": "BLS OEWS",
                "total_retries": state.get("retry_count", 0),
            }
        })

        logger.info(f"Compensation analysis finalized for {state['job_title']}")
        return new_state

    def _input_validation_routing(self, state: CompensationAnalysisState) -> str:
        """Route based on input validation results."""
        status = state.get("status", "")
        if status == "input_validated":
            return "proceed"
        elif status == "validation_failed" and state.get("retry_count", 0) < self.max_retries:
            return "retry"
        else:
            return "failed"

    def _analysis_routing(self, state: CompensationAnalysisState) -> str:
        """Route based on analysis results."""
        status = state.get("status", "")
        if status == "analysis_completed":
            return "validate"
        elif status == "analysis_failed" and state.get("retry_count", 0) < self.max_retries:
            return "retry"
        else:
            return "failed"

    def _validation_routing(self, state: CompensationAnalysisState) -> str:
        """Route based on validation results."""
        status = state.get("status", "")
        if status == "results_validated":
            return "success"
        elif state.get("retry_count", 0) < self.max_retries:
            return "retry"
        else:
            return "success"  # Accept results even with validation issues

    def _error_handling_routing(self, state: CompensationAnalysisState) -> str:
        """Route based on error handling decisions."""
        status = state.get("status", "")
        if status == "retrying":
            return "retry"
        else:
            return "failed"

    def _calculate_completeness_score(self, comp_band: Dict[str, Any]) -> float:
        """Calculate data completeness score (0-1)."""
        score = 0.0
        
        # Check for salary percentiles
        if comp_band.get("p25") is not None:
            score += 0.2
        if comp_band.get("p50") is not None:
            score += 0.2
        if comp_band.get("p75") is not None:
            score += 0.2
            
        # Check for metadata
        if comp_band.get("occupation_code"):
            score += 0.2
        if comp_band.get("sources"):
            score += 0.2
            
        return score

    def analyze_compensation(
        self, 
        job_title: str, 
        location: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous interface for compensation analysis.

        Args:
            job_title: Job title to analyze
            location: Geographic location (optional)
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Dict containing analysis results and metadata
        """
        initial_state = CompensationAnalysisState(
            job_title=job_title,
            location=location,
            status="initialized",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )

        config = {"configurable": {"thread_id": thread_id or "default"}} if thread_id else {}
        
        try:
            # Execute the workflow
            final_state = self.compiled_graph.invoke(initial_state, config)
            
            # Convert back to CompBand object if available
            if final_state.get("comp_band"):
                comp_band_data = final_state["comp_band"].copy()
                # Convert string back to datetime
                if "as_of" in comp_band_data:
                    comp_band_data["as_of"] = datetime.fromisoformat(comp_band_data["as_of"])
                final_state["comp_band_object"] = CompBand(**comp_band_data)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "status": "workflow_failed",
                "error_message": str(e),
                "job_title": job_title,
                "location": location,
            }

    async def analyze_compensation_async(
        self,
        job_title: str,
        location: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous interface for compensation analysis.

        Args:
            job_title: Job title to analyze
            location: Geographic location (optional)
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Dict containing analysis results and metadata
        """
        initial_state = CompensationAnalysisState(
            job_title=job_title,
            location=location,
            status="initialized",
            error_message=None,
            retry_count=0,
            soc_mapping=None,
            validation_report=None,
            geographic_info=None,
            comp_band=None,
            analysis_metadata={},
            started_at=None,
            completed_at=None,
        )

        config = {"configurable": {"thread_id": thread_id or "default"}} if thread_id else {}

        try:
            # Execute the workflow asynchronously
            final_state = await self.compiled_graph.ainvoke(initial_state, config)
            
            # Convert back to CompBand object if available
            if final_state.get("comp_band"):
                comp_band_data = final_state["comp_band"].copy()
                # Convert string back to datetime
                if "as_of" in comp_band_data:
                    comp_band_data["as_of"] = datetime.fromisoformat(comp_band_data["as_of"])
                final_state["comp_band_object"] = CompBand(**comp_band_data)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Async workflow execution failed: {str(e)}")
            return {
                "status": "workflow_failed",
                "error_message": str(e),
                "job_title": job_title,
                "location": location,
            }