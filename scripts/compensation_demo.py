#!/usr/bin/env python3
"""
Demonstration script for the CompensationAnalystAgent functionality.

This script shows how to use both the CompensationAnalysisTool directly
and the CompensationAnalystAgent for LangGraph workflow orchestration.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.compensation_analysis import CompensationAnalysisTool
from agents.compensation_analyst_agent import CompensationAnalystAgent


def demo_compensation_tool():
    """Demonstrate the CompensationAnalysisTool functionality."""
    print("=" * 60)
    print("CompensationAnalysisTool Demo")
    print("=" * 60)
    
    tool = CompensationAnalysisTool()
    
    # Test cases
    test_cases = [
        ("Software Engineer", "California"),
        ("Data Scientist", "New York"),
        ("AI Algorithms Team Lead", "Washington"),
        ("Head of AI", "Texas"),
        ("Senior Machine Learning Engineer", None),  # National data
        ("Product Manager", "Colorado"),
    ]
    
    for job_title, location in test_cases:
        print(f"\nAnalyzing: {job_title} in {location or 'National'}")
        print("-" * 50)
        
        try:
            # Validate inputs first
            validation = tool.validate_inputs(job_title, location)
            print(f"SOC Mapping Confidence: {validation['soc_mapping_confidence']:.2f}")
            
            if validation['warnings']:
                print(f"Warnings: {', '.join(validation['warnings'])}")
            
            # Run full analysis
            comp_band = tool.analyze_compensation(job_title, location)
            
            print(f"SOC Code: {comp_band.occupation_code}")
            print(f"Geography: {comp_band.geography}")
            print(f"Salary Range (USD):")
            if comp_band.p25:
                print(f"  25th Percentile: ${comp_band.p25:,.0f}")
            if comp_band.p50:
                print(f"  Median (50th):   ${comp_band.p50:,.0f}")
            if comp_band.p75:
                print(f"  75th Percentile: ${comp_band.p75:,.0f}")
            print(f"Data as of: {comp_band.as_of.strftime('%B %Y')}")
            
        except Exception as e:
            print(f"Error: {e}")


def demo_compensation_agent():
    """Demonstrate the CompensationAnalystAgent workflow."""
    print("\n" + "=" * 60)
    print("CompensationAnalystAgent Demo")
    print("=" * 60)
    
    agent = CompensationAnalystAgent(
        min_confidence=0.6,
        max_retries=3,
        enable_checkpoints=False,  # Disable for demo
    )
    
    # Test successful analysis
    print(f"\nRunning workflow analysis for: Software Engineer in California")
    print("-" * 50)
    
    result = agent.analyze_compensation("Software Engineer", "California")
    
    print(f"Status: {result['status']}")
    print(f"Total Retries: {result.get('analysis_metadata', {}).get('completion_summary', {}).get('total_retries', 0)}")
    
    if result.get('comp_band_object'):
        comp_band = result['comp_band_object']
        print(f"Result: ${comp_band.p50:,.0f} median salary for {comp_band.occupation_code} in {comp_band.geography}")
    
    # Show metadata
    if result.get('validation_report'):
        validation = result['validation_report']
        print(f"Data Quality: {validation.get('results_validation', {}).get('overall_quality', 'unknown')}")
        
    # Test failure case
    print(f"\nTesting workflow with unmappable job title...")
    print("-" * 50)
    
    failure_result = agent.analyze_compensation("Professional Unicorn Wrangler", "California")
    print(f"Status: {failure_result['status']}")
    if failure_result.get('error_message'):
        print(f"Error: {failure_result['error_message']}")


async def demo_async_agent():
    """Demonstrate the asynchronous CompensationAnalystAgent workflow."""
    print("\n" + "=" * 60)
    print("CompensationAnalystAgent Async Demo")
    print("=" * 60)
    
    agent = CompensationAnalystAgent(enable_checkpoints=False)
    
    # Run multiple analyses concurrently
    job_titles = ["Data Scientist", "Product Manager", "DevOps Engineer"]
    locations = ["New York", "California", "Washington"]
    
    tasks = []
    for title, location in zip(job_titles, locations):
        task = agent.analyze_compensation_async(title, location)
        tasks.append((title, location, task))
    
    print("Running 3 concurrent compensation analyses...")
    
    for title, location, task in tasks:
        result = await task
        print(f"\n{title} in {location}:")
        
        if result.get('comp_band_object'):
            comp_band = result['comp_band_object']
            print(f"  Median Salary: ${comp_band.p50:,.0f}")
        else:
            print(f"  Status: {result['status']}")


def show_available_data():
    """Show what data is available in the system."""
    print("\n" + "=" * 60)
    print("Available Data")
    print("=" * 60)
    
    tool = CompensationAnalysisTool()
    
    print("\nSupported Job Roles (sample):")
    roles = tool.get_supported_roles()[:10]  # Show first 10
    for role in roles:
        print(f"  - {role.title()}")
    print(f"  ... and {len(tool.get_supported_roles()) - 10} more")
    
    print("\nSupported Geographies:")
    geographies = tool.get_available_geographies()
    for geo in geographies[:10]:  # Show first 10
        print(f"  - {geo}")
    if len(geographies) > 10:
        print(f"  ... and {len(geographies) - 10} more")


def main():
    """Run all demonstration functions."""
    print("CompensationAnalyst System Demo")
    print("Generated on:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Show available data
    show_available_data()
    
    # Demonstrate tool functionality
    demo_compensation_tool()
    
    # Demonstrate agent workflow
    demo_compensation_agent()
    
    # Demonstrate async functionality
    print("\nRunning async demo...")
    asyncio.run(demo_async_agent())
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    print(f"\nNext Steps:")
    print("- Integrate with real BLS OEWS API (replace mock data)")
    print("- Add more SOC code mappings via O*NET Web Services")
    print("- Implement geographic cost-of-living adjustments")
    print("- Add currency conversion for international markets")


if __name__ == "__main__":
    main()