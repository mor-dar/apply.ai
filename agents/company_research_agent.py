"""
CompanyResearchAgent for orchestrating company research within LangGraph workflow.

This agent handles the workflow orchestration, error handling, and state management
for company research operations. It uses the CompanyResearchTool internally for the
actual web scraping and fact extraction while managing retries, validation, and 
LangGraph state updates.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from tools.company_research import CompanyResearchTool
from src.schemas.core import FactSheet


# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MIN_FACTS_REQUIRED = 3
MAX_FACTS_TARGET = 10
CONFIDENCE_MIN = 0.4


class CompanyResearchState(TypedDict):
    """
    State dictionary for company research workflow.

    This state is passed between nodes in the LangGraph and tracks
    the progress and results of company research operations.
    """

    # Input data
    company_name: str
    research_focus: Optional[
        str
    ]  # Optional focus area (e.g., "recent news", "funding")

    # Processing status
    status: str
    error_message: Optional[str]
    retry_count: int

    # Results
    fact_sheet: Optional[Dict[str, Any]]  # Serialized FactSheet
    research_metadata: Dict[str, Any]
    validation_report: Optional[Dict[str, Any]]

    # Timestamps (as ISO strings for JSON serialization)
    started_at: Optional[str]
    completed_at: Optional[str]


class CompanyResearchAgent:
    """
    Agent for orchestrating company research workflows.

    This agent manages the complete workflow for researching companies,
    including error handling, retries, and LangGraph state management.
    It uses the CompanyResearchTool for the actual research work.
    """

    def __init__(
        self,
        max_facts: int = MAX_FACTS_TARGET,
        request_timeout: int = 10,
        request_delay: float = 1.0,
        max_retries: int = 3,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the CompanyResearchAgent.

        Args:
            max_facts: Maximum number of facts to include in research
            request_timeout: HTTP request timeout in seconds
            request_delay: Delay between requests for rate limiting
            max_retries: Maximum retry attempts for failed operations
            enable_checkpoints: Whether to enable LangGraph checkpointing
        """
        self.max_retries = max_retries

        # Initialize the research tool
        self.research_tool = CompanyResearchTool(
            max_facts=max_facts,
            request_timeout=request_timeout,
            request_delay=request_delay,
            max_retries=max_retries,
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
        Build the LangGraph workflow for company research.

        Returns:
            Configured StateGraph for the research workflow
        """
        # Create the graph with our state model
        workflow = StateGraph(CompanyResearchState)

        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_research)
        workflow.add_node("conduct_research", self._conduct_research)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_results)

        # Define the workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "conduct_research")

        # Conditional routing from research
        workflow.add_conditional_edges(
            "conduct_research",
            self._should_retry_research,
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
            {"retry": "conduct_research", "failed": END},
        )

        # Finalization
        workflow.add_edge("finalize", END)

        return workflow

    def _initialize_research(self, state: CompanyResearchState) -> CompanyResearchState:
        """
        Initialize the research workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated state with initialization metadata
        """
        logger.info(
            f"Initializing company research for: {state.get('company_name', 'Unknown Company')}"
        )

        new_state = state.copy()
        new_state.update(
            {
                "status": "initializing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "research_metadata": {
                    "tool_config": {
                        "max_facts": self.research_tool.max_facts,
                        "request_timeout": self.research_tool.request_timeout,
                        "request_delay": self.research_tool.request_delay,
                    },
                    "workflow_version": "1.0",
                },
            }
        )
        return new_state

    def _conduct_research(self, state: CompanyResearchState) -> CompanyResearchState:
        """
        Conduct company research using the CompanyResearchTool.

        This method implements the agent policy:
        1. Call the research tool with company name
        2. Check fact quality and quantity
        3. Apply validation rules
        4. Update state with results

        Args:
            state: Current workflow state

        Returns:
            Updated state with research results
        """
        try:
            logger.info(
                f"Conducting research (attempt {state.get('retry_count', 0) + 1}/{self.max_retries})"
            )

            company_name = state["company_name"]

            # Use the tool to research the company
            fact_sheet = self.research_tool.research_company(company_name)

            logger.info(
                f"Research completed: {len(fact_sheet.facts)} facts extracted "
                f"from {len(set(f.source_url for f in fact_sheet.facts))} sources"
            )

            # Log fact breakdown by source type
            official_facts = sum(
                1 for f in fact_sheet.facts if f.source_domain_class.value == "official"
            )
            news_facts = sum(
                1
                for f in fact_sheet.facts
                if f.source_domain_class.value == "reputable_news"
            )
            other_facts = sum(
                1 for f in fact_sheet.facts if f.source_domain_class.value == "other"
            )

            logger.info(
                f"Fact sources: {official_facts} official, {news_facts} news, {other_facts} other"
            )

            new_state = state.copy()
            new_state.update(
                {
                    "status": "research_completed",
                    "fact_sheet": fact_sheet.model_dump(
                        mode="json"
                    ),  # Use JSON mode for serialization
                    "error_message": None,
                }
            )
            return new_state

        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            logger.error(error_msg)

            new_state = state.copy()
            new_state.update({"status": "error", "error_message": error_msg})
            return new_state

    def _validate_results(self, state: CompanyResearchState) -> CompanyResearchState:
        """
        Validate the research results using quality gates.

        This method validates the FactSheet quality based on:
        - Minimum number of facts
        - Fact confidence scores
        - Source diversity
        - Required field presence

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation results
        """
        fact_sheet_data = state.get("fact_sheet")

        if not fact_sheet_data:
            new_state = state.copy()
            new_state.update(
                {
                    "status": "validation_failed",
                    "error_message": "No fact sheet data to validate",
                }
            )
            return new_state

        try:
            # Recreate FactSheet object for validation
            fact_sheet = FactSheet(**fact_sheet_data)
            validation_issues = []

            # Quality gate 1: Minimum number of facts
            if len(fact_sheet.facts) < MIN_FACTS_REQUIRED:
                validation_issues.append(
                    f"Insufficient facts extracted ({len(fact_sheet.facts)} < {MIN_FACTS_REQUIRED})"
                )

            # Quality gate 2: Average confidence threshold
            if fact_sheet.facts:
                avg_confidence = sum(f.confidence for f in fact_sheet.facts) / len(
                    fact_sheet.facts
                )
                if avg_confidence < CONFIDENCE_MIN:
                    validation_issues.append(
                        f"Low average confidence ({avg_confidence:.3f} < {CONFIDENCE_MIN})"
                    )

            # Quality gate 3: Source diversity (relaxed for small fact sets)
            source_domains = set(str(f.source_url) for f in fact_sheet.facts)
            if len(source_domains) < 2 and len(fact_sheet.facts) >= 5:
                validation_issues.append("Insufficient source diversity")

            # Quality gate 4: Check for required fact categories
            fact_statements = " ".join(f.statement for f in fact_sheet.facts).lower()
            has_company_info = any(
                keyword in fact_statements
                for keyword in ["company", "founded", "business", "organization"]
            )
            if not has_company_info:
                validation_issues.append("No basic company information found")

            # Quality gate 5: Recent news validation (if any news facts exist)
            news_facts = [
                f
                for f in fact_sheet.facts
                if "news" in f.statement.lower()
                or "announced" in f.statement.lower()
                or "launched" in f.statement.lower()
            ]
            outdated_news = 0
            if news_facts:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)
                outdated_news = sum(1 for f in news_facts if f.as_of_date < cutoff_date)

            # Generate validation report
            validation_report = {
                "total_facts": len(fact_sheet.facts),
                "avg_confidence": avg_confidence if fact_sheet.facts else 0.0,
                "source_count": len(source_domains),
                "validation_issues": validation_issues,
                "fact_categories": self._categorize_facts(fact_sheet.facts),
                "outdated_news_count": outdated_news,
            }

            if validation_issues:
                error_msg = f"Validation failed: {'; '.join(validation_issues)}"
                logger.warning(error_msg)

                new_state = state.copy()
                new_state.update(
                    {
                        "status": "validation_failed",
                        "error_message": error_msg,
                        "validation_report": validation_report,
                    }
                )
                return new_state

            logger.info("Company research validation passed")
            new_state = state.copy()
            new_state.update(
                {
                    "status": "validated",
                    "validation_report": validation_report,
                }
            )
            return new_state

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)

            new_state = state.copy()
            new_state.update(
                {
                    "status": "validation_failed",
                    "error_message": error_msg,
                }
            )
            return new_state

    def _categorize_facts(self, facts: List[Any]) -> Dict[str, int]:
        """Categorize facts by type for validation reporting."""
        categories = {
            "company_info": 0,
            "products": 0,
            "funding": 0,
            "news": 0,
            "mission": 0,
            "other": 0,
        }

        for fact in facts:
            statement = fact.statement.lower()

            if any(
                keyword in statement
                for keyword in ["founded", "headquartered", "based", "company"]
            ):
                categories["company_info"] += 1
            elif any(
                keyword in statement
                for keyword in ["product", "service", "platform", "offers"]
            ):
                categories["products"] += 1
            elif any(
                keyword in statement
                for keyword in ["funding", "raised", "investment", "valuation"]
            ):
                categories["funding"] += 1
            elif any(
                keyword in statement
                for keyword in ["announced", "launched", "partnership", "acquisition"]
            ):
                categories["news"] += 1
            elif any(
                keyword in statement
                for keyword in ["mission", "vision", "believes", "committed"]
            ):
                categories["mission"] += 1
            else:
                categories["other"] += 1

        return categories

    def _handle_error(self, state: CompanyResearchState) -> CompanyResearchState:
        """
        Handle errors and prepare for potential retry.

        Args:
            state: Current workflow state

        Returns:
            Updated state with incremented retry count
        """
        new_retry_count = state.get("retry_count", 0) + 1
        logger.warning(
            f"Handling error (retry {new_retry_count}/{self.max_retries}): "
            f"{state.get('error_message')}"
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

    def _finalize_results(self, state: CompanyResearchState) -> CompanyResearchState:
        """
        Finalize the research workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated state with completion metadata
        """
        logger.info("Finalizing company research results")

        # Update research metadata with final statistics
        metadata = state.get("research_metadata", {}).copy()
        fact_sheet_data = state.get("fact_sheet", {})
        validation_report = state.get("validation_report", {})

        metadata.update(
            {
                "final_status": "completed",
                "total_facts": len(fact_sheet_data.get("facts", [])),
                "retry_attempts": state.get("retry_count", 0),
                "validation_passed": True,
                "fact_categories": validation_report.get("fact_categories", {}),
                "avg_confidence": validation_report.get("avg_confidence", 0.0),
            }
        )

        new_state = state.copy()
        new_state.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "research_metadata": metadata,
            }
        )
        return new_state

    def _should_retry_research(self, state: CompanyResearchState) -> str:
        """
        Determine if research should be retried based on current state.

        Args:
            state: Current workflow state

        Returns:
            Next workflow step: "validate", "retry", or "failed"
        """
        status = state.get("status")
        retry_count = state.get("retry_count", 0)

        if status == "research_completed":
            return "validate"
        elif status == "error" and retry_count < self.max_retries:
            return "retry"
        else:
            return "failed"

    def _validation_complete(self, state: CompanyResearchState) -> str:
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

    def _should_continue_after_error(self, state: CompanyResearchState) -> str:
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

    async def research_company(
        self,
        company_name: str,
        research_focus: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Research a company using the LangGraph workflow.

        Args:
            company_name: Name of the company to research
            research_focus: Optional focus area for research
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Dictionary containing the final state and results
        """
        # Create initial state
        initial_state: CompanyResearchState = {
            "company_name": company_name,
            "research_focus": research_focus,
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
            "fact_sheet": None,
            "research_metadata": {},
            "validation_report": None,
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
                "fact_sheet": final_state.get("fact_sheet"),
                "metadata": final_state.get("research_metadata", {}),
                "validation_report": final_state.get("validation_report", {}),
                "error": final_state.get("error_message"),
                "retry_count": final_state.get("retry_count", 0),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "fact_sheet": None,
                "metadata": {"error": "workflow_execution_failed"},
                "validation_report": {},
                "error": str(e),
                "retry_count": 0,
            }

    def research_company_sync(
        self,
        company_name: str,
        research_focus: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous version of research_company for simpler usage.

        Args:
            company_name: Name of the company to research
            research_focus: Optional focus area for research
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Dictionary containing the final state and results
        """
        # Create initial state
        initial_state: CompanyResearchState = {
            "company_name": company_name,
            "research_focus": research_focus,
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
            "fact_sheet": None,
            "research_metadata": {},
            "validation_report": None,
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
                "fact_sheet": final_state.get("fact_sheet"),
                "metadata": final_state.get("research_metadata", {}),
                "validation_report": final_state.get("validation_report", {}),
                "error": final_state.get("error_message"),
                "retry_count": final_state.get("retry_count", 0),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "fact_sheet": None,
                "metadata": {"error": "workflow_execution_failed"},
                "validation_report": {},
                "error": str(e),
                "retry_count": 0,
            }
