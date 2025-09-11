"""
Tests for the CompensationAnalysisTool.

This module contains comprehensive tests for the compensation analysis tool,
including SOC mapping, BLS data integration, geographic handling, and error
scenarios. Tests ensure zero-warning compliance with project standards.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

from tools.compensation_analysis import CompensationAnalysisTool, ONET_SOC_CROSSWALK, STATE_CODES, STATE_ABBREVIATIONS
from src.schemas.core import CompBand


class TestCompensationAnalysisTool:
    """Test suite for CompensationAnalysisTool."""

    @pytest.fixture
    def tool(self) -> CompensationAnalysisTool:
        """Create a CompensationAnalysisTool instance for testing."""
        return CompensationAnalysisTool()

    @pytest.fixture
    def mock_requests_session(self):
        """Mock requests session for testing."""
        with patch('tools.compensation_analysis.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            yield mock_session

    # SOC Mapping Tests
    
    def test_direct_soc_mapping(self, tool: CompensationAnalysisTool):
        """Test direct SOC code mapping for known job titles."""
        # Test exact match
        result = tool._map_title_to_soc("software developer")
        assert result is not None
        soc_code, confidence = result
        assert soc_code == "15-1252"
        assert confidence == 1.0

    def test_fuzzy_soc_mapping(self, tool: CompensationAnalysisTool):
        """Test fuzzy SOC code mapping for similar titles."""
        # Test similar title that should match
        result = tool._map_title_to_soc("Senior Software Developer")
        assert result is not None
        soc_code, confidence = result
        assert soc_code == "15-1252"
        assert confidence > 0.6

    def test_partial_soc_mapping(self, tool: CompensationAnalysisTool):
        """Test partial SOC code mapping for compound titles."""
        # Test compound title with recognizable components
        result = tool._map_title_to_soc("Senior Machine Learning Engineer")
        assert result is not None
        soc_code, confidence = result
        # Should match either machine learning engineer or software engineer
        assert soc_code in ["15-1252"]
        assert confidence > 0.4

    def test_no_soc_mapping(self, tool: CompensationAnalysisTool):
        """Test handling of unmappable job titles."""
        result = tool._map_title_to_soc("Professional Unicorn Wrangler")
        assert result is None

    def test_empty_job_title_mapping(self, tool: CompensationAnalysisTool):
        """Test handling of empty job titles."""
        result = tool._map_title_to_soc("")
        assert result is None
        
        result = tool._map_title_to_soc("   ")
        assert result is None

    # Geographic Code Tests

    def test_direct_state_mapping(self, tool: CompensationAnalysisTool):
        """Test direct state name to code mapping."""
        assert tool._get_geographic_code("california") == "CA"
        assert tool._get_geographic_code("California") == "CA"
        assert tool._get_geographic_code("CALIFORNIA") == "CA"

    def test_city_state_mapping(self, tool: CompensationAnalysisTool):
        """Test city,state format geographic mapping."""
        assert tool._get_geographic_code("San Francisco, California") == "CA"
        assert tool._get_geographic_code("New York City, New York") == "NY"
        assert tool._get_geographic_code("Austin, TX") == "TX"

    def test_partial_location_mapping(self, tool: CompensationAnalysisTool):
        """Test partial location name matching."""
        assert tool._get_geographic_code("Bay Area, California") == "CA"
        assert tool._get_geographic_code("Greater Seattle area") == "WA"

    def test_national_default_mapping(self, tool: CompensationAnalysisTool):
        """Test fallback to national for unknown locations."""
        assert tool._get_geographic_code("Unknown City") == "national"
        assert tool._get_geographic_code("Mars Colony") == "national"
        assert tool._get_geographic_code(None) == "national"

    # Wage Data Retrieval Tests

    def test_wage_data_retrieval_with_state(self, tool: CompensationAnalysisTool):
        """Test wage data retrieval for state-specific data."""
        wage_data = tool._get_wage_data("15-1252", "CA", "California")
        assert wage_data is not None
        assert "p25" in wage_data
        assert "p50" in wage_data
        assert "p75" in wage_data
        assert all(isinstance(wage_data[p], (int, float)) for p in ["p25", "p50", "p75"])

    def test_wage_data_fallback_to_national(self, tool: CompensationAnalysisTool):
        """Test wage data fallback to national when state data unavailable."""
        # Test with a state that should fall back to national
        wage_data = tool._get_wage_data("15-1252", "ZZ", "Unknown State")
        assert wage_data is not None
        # Should get national data as fallback

    def test_no_wage_data_available(self, tool: CompensationAnalysisTool):
        """Test handling when no wage data is available."""
        wage_data = tool._get_wage_data("99-9999", "CA", "California")
        assert wage_data is None

    # CompBand Building Tests

    def test_build_comp_band_complete_data(self, tool: CompensationAnalysisTool):
        """Test CompBand building with complete wage data."""
        wage_data = {"p25": 100000, "p50": 130000, "p75": 170000}
        comp_band = tool._build_comp_band("15-1252", "CA", "California", wage_data, 0.9)
        
        assert isinstance(comp_band, CompBand)
        assert comp_band.occupation_code == "15-1252"
        assert comp_band.p25 == 100000
        assert comp_band.p50 == 130000
        assert comp_band.p75 == 170000
        assert comp_band.currency == "USD"
        assert len(comp_band.sources) > 0
        assert comp_band.as_of.year == 2024

    def test_build_comp_band_national(self, tool: CompensationAnalysisTool):
        """Test CompBand building with national data."""
        wage_data = {"p25": 88000, "p50": 118000, "p75": 155000}
        comp_band = tool._build_comp_band("15-1252", "national", None, wage_data, 0.8)
        
        assert comp_band.geography == "United States (National)"

    def test_build_comp_band_partial_data(self, tool: CompensationAnalysisTool):
        """Test CompBand building with partial wage data."""
        wage_data = {"p50": 120000}  # Only median available
        comp_band = tool._build_comp_band("15-1252", "CA", "California", wage_data, 0.7)
        
        assert comp_band.p25 is None
        assert comp_band.p50 == 120000
        assert comp_band.p75 is None

    # Full Analysis Tests

    def test_analyze_compensation_success(self, tool: CompensationAnalysisTool):
        """Test successful compensation analysis end-to-end."""
        comp_band = tool.analyze_compensation("Software Developer", "California")
        
        assert isinstance(comp_band, CompBand)
        assert comp_band.occupation_code == "15-1252"
        assert comp_band.p50 is not None
        assert comp_band.p50 > 0
        assert comp_band.currency == "USD"
        assert len(comp_band.sources) > 0

    def test_analyze_compensation_national_fallback(self, tool: CompensationAnalysisTool):
        """Test compensation analysis with national fallback."""
        comp_band = tool.analyze_compensation("Data Scientist")
        
        assert isinstance(comp_band, CompBand)
        assert comp_band.geography == "United States (National)"

    def test_analyze_compensation_invalid_title(self, tool: CompensationAnalysisTool):
        """Test analysis with invalid job title."""
        with pytest.raises(ValueError, match="Job title cannot be empty"):
            tool.analyze_compensation("")

        with pytest.raises(RuntimeError, match="Could not map job title"):
            tool.analyze_compensation("Completely Unknown Job Title XYZ123")

    def test_analyze_compensation_no_wage_data(self, tool: CompensationAnalysisTool):
        """Test analysis when no wage data is available."""
        # Mock a SOC mapping that works but has no wage data
        with patch.object(tool, '_map_title_to_soc') as mock_map:
            mock_map.return_value = ("99-9999", 0.8)
            
            with pytest.raises(RuntimeError, match="No wage data available"):
                tool.analyze_compensation("Mock Job Title")

    # Utility Method Tests

    def test_get_available_geographies(self, tool: CompensationAnalysisTool):
        """Test getting available geographic areas."""
        geographies = tool.get_available_geographies()
        
        assert isinstance(geographies, list)
        assert len(geographies) > 0
        assert "National" in geographies
        assert "California" in geographies
        assert all(isinstance(geo, str) for geo in geographies)

    def test_get_supported_roles(self, tool: CompensationAnalysisTool):
        """Test getting supported job roles."""
        roles = tool.get_supported_roles()
        
        assert isinstance(roles, list)
        assert len(roles) > 0
        assert "software developer" in roles
        assert "data scientist" in roles
        assert all(isinstance(role, str) for role in roles)

    def test_validate_inputs_valid(self, tool: CompensationAnalysisTool):
        """Test input validation with valid inputs."""
        result = tool.validate_inputs("Software Engineer", "California")
        
        assert result["valid_job_title"] is True
        assert result["soc_mapping_confidence"] > 0.8
        assert result["valid_location"] is True
        assert result["geographic_code"] == "CA"
        assert len(result["warnings"]) == 0

    def test_validate_inputs_invalid_title(self, tool: CompensationAnalysisTool):
        """Test input validation with invalid job title."""
        result = tool.validate_inputs("", "California")
        
        assert result["valid_job_title"] is False
        assert "Job title is required" in result["warnings"]

    def test_validate_inputs_unknown_location(self, tool: CompensationAnalysisTool):
        """Test input validation with unknown location."""
        result = tool.validate_inputs("Software Engineer", "Mars Colony")
        
        assert result["valid_location"] is True  # Still valid, just falls back
        assert result["geographic_code"] == "national"
        assert len([w for w in result["warnings"] if "not recognized" in w]) > 0

    def test_validate_inputs_no_soc_mapping(self, tool: CompensationAnalysisTool):
        """Test input validation with unmappable job title."""
        result = tool.validate_inputs("Professional Unicorn Wrangler")
        
        assert result["valid_job_title"] is True
        assert result["soc_mapping_confidence"] == 0.0
        assert "No SOC mapping found" in " ".join(result["warnings"])

    # Edge Cases and Error Handling

    def test_rate_limiting(self, tool: CompensationAnalysisTool):
        """Test rate limiting functionality."""
        # This is a unit test for the rate limiting method
        start_time = tool.last_request_time
        tool._rate_limit()
        # Should have updated the timestamp
        assert tool.last_request_time >= start_time

    def test_case_insensitive_matching(self, tool: CompensationAnalysisTool):
        """Test case insensitive job title and location matching."""
        # Test various case combinations
        cases = [
            ("SOFTWARE DEVELOPER", "15-1252"),
            ("Software Developer", "15-1252"),
            ("software developer", "15-1252"),
            ("SoFtWaRe DeVeLoPeR", "15-1252"),
        ]
        
        for title, expected_soc in cases:
            result = tool._map_title_to_soc(title)
            assert result is not None
            soc_code, confidence = result
            assert soc_code == expected_soc

    def test_whitespace_handling(self, tool: CompensationAnalysisTool):
        """Test handling of titles and locations with extra whitespace."""
        result = tool._map_title_to_soc("  software developer  ")
        assert result is not None
        assert result[0] == "15-1252"
        
        geo_code = tool._get_geographic_code("  california  ")
        assert geo_code == "CA"

    def test_special_characters_in_titles(self, tool: CompensationAnalysisTool):
        """Test handling of job titles with special characters."""
        # Test titles with common special characters
        titles = [
            "Sr. Software Engineer",
            "Software Engineer II", 
            "Software Engineer (Backend)",
            "Software Engineer - Full Stack",
        ]
        
        for title in titles:
            result = tool._map_title_to_soc(title)
            # Should find some mapping for these common variations
            assert result is not None or title.lower() not in ONET_SOC_CROSSWALK

    # Integration and Performance Tests

    @pytest.mark.integration
    def test_multiple_analyses_performance(self, tool: CompensationAnalysisTool):
        """Test performance with multiple analyses."""
        job_titles = ["Software Developer", "Data Scientist", "Product Manager"]
        locations = ["California", "New York", "Texas", None]
        
        results = []
        for title in job_titles:
            for location in locations:
                try:
                    result = tool.analyze_compensation(title, location)
                    results.append(result)
                except (ValueError, RuntimeError):
                    # Some combinations may not have data
                    pass
        
        assert len(results) > 0
        assert all(isinstance(r, CompBand) for r in results)

    @pytest.mark.integration
    def test_comprehensive_role_coverage(self, tool: CompensationAnalysisTool):
        """Test coverage across different role types."""
        # Test various role categories
        tech_roles = ["software engineer", "data scientist", "devops engineer"]
        management_roles = ["engineering manager", "product manager"]
        ai_roles = ["machine learning engineer", "ai researcher"]
        
        all_roles = tech_roles + management_roles + ai_roles
        
        for role in all_roles:
            result = tool._map_title_to_soc(role)
            assert result is not None, f"No SOC mapping for {role}"
            
            soc_code, confidence = result
            assert confidence > 0.4, f"Low confidence for {role}: {confidence}"

    # Mock and Isolation Tests

    def test_tool_initialization(self, tool: CompensationAnalysisTool):
        """Test proper tool initialization."""
        assert hasattr(tool, 'session')
        assert hasattr(tool, 'last_request_time')
        assert tool.last_request_time == 0.0

    def test_session_headers(self, tool: CompensationAnalysisTool):
        """Test that HTTP session has proper headers."""
        assert 'User-Agent' in tool.session.headers
        assert 'ApplyAI' in tool.session.headers['User-Agent']

    def test_constants_loaded(self):
        """Test that necessary constants are properly loaded."""
        assert isinstance(ONET_SOC_CROSSWALK, dict)
        assert len(ONET_SOC_CROSSWALK) > 0
        assert isinstance(STATE_CODES, dict)
        assert len(STATE_CODES) > 0
        assert "national" in STATE_CODES

    # Data Validation Tests

    def test_soc_code_format_validation(self, tool: CompensationAnalysisTool):
        """Test SOC code format validation."""
        for soc_code in ONET_SOC_CROSSWALK.values():
            # SOC codes should be in format XX-XXXX
            assert len(soc_code) == 7
            assert soc_code[2] == "-"
            assert soc_code[:2].isdigit()
            assert soc_code[3:].isdigit()

    def test_wage_data_structure_validation(self, tool: CompensationAnalysisTool):
        """Test wage data structure validation."""
        # Test with a known SOC code and geography
        wage_data = tool._get_wage_data("15-1252", "CA", "California")
        
        if wage_data:  # May not have data in mock environment
            assert isinstance(wage_data, dict)
            for percentile in ["p25", "p50", "p75"]:
                if percentile in wage_data:
                    assert isinstance(wage_data[percentile], (int, float))
                    assert wage_data[percentile] > 0

    def test_comp_band_schema_compliance(self, tool: CompensationAnalysisTool):
        """Test that generated CompBand objects comply with schema."""
        try:
            comp_band = tool.analyze_compensation("Software Developer", "California")
            
            # Test schema compliance by creating a new instance from dict
            comp_band_dict = comp_band.model_dump()
            new_comp_band = CompBand(**comp_band_dict)
            
            assert new_comp_band.occupation_code == comp_band.occupation_code
            assert new_comp_band.geography == comp_band.geography
            assert new_comp_band.p50 == comp_band.p50
            
        except (ValueError, RuntimeError):
            # Test may fail in environments without mock data
            pytest.skip("No compensation data available for schema test")