"""
Test script for Global Risk Knowledge Agent

Tests core functionality:
1. Agent initialization
2. Internal search
3. External analysis (if API key available)
4. Report generation

Usage:
    python test_agent.py          # Run locally (requires dependencies)
    docker exec <container> python /app/test_agent.py  # Run in Docker
"""

import os
import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / "app"
if app_dir.exists():
    sys.path.insert(0, str(app_dir))
else:
    # Running in Docker
    sys.path.insert(0, "/app")

try:
    from agent import RiskIntelligenceAgent, get_perplexity_api_key
except ImportError:
    # Try app.agent if running from root
    try:
        from app.agent import RiskIntelligenceAgent, get_perplexity_api_key
    except ImportError as e:
        print(f"Error importing agent: {e}")
        print("Make sure you're running from the project root or inside Docker container")
        sys.exit(1)

def test_agent_initialization():
    """Test that agent can be initialized."""
    print("=" * 60)
    print("TEST 1: Agent Initialization")
    print("=" * 60)
    try:
        agent = RiskIntelligenceAgent()
        print("✓ Agent initialized successfully")
        return agent
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return None

def test_api_key():
    """Test API key availability."""
    print("\n" + "=" * 60)
    print("TEST 2: Perplexity API Key Check")
    print("=" * 60)
    try:
        key = get_perplexity_api_key()
        if key:
            print(f"✓ API key found (length: {len(key)})")
            return True
    except RuntimeError as e:
        print(f"⚠ API key not set: {e}")
        print("  Continuing with limited tests (no external analysis)")
        return False

def test_internal_search(agent):
    """Test internal document search."""
    print("\n" + "=" * 60)
    print("TEST 3: Internal Document Search")
    print("=" * 60)
    if not agent:
        print("✗ Skipping - agent not initialized")
        return None
    
    test_query = "tunnel construction risks"
    print(f"Query: '{test_query}'")
    
    try:
        results = agent.internal_search(test_query)
        print(f"✓ Internal search completed")
        print(f"  Found {len(results)} results")
        if results:
            print(f"  First result: {results[0].get('source', 'N/A')[:50]}...")
        return results
    except Exception as e:
        print(f"✗ Internal search failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_internal_processing(agent, results):
    """Test internal lessons processing."""
    print("\n" + "=" * 60)
    print("TEST 4: Internal Lessons Processing")
    print("=" * 60)
    if not agent or not results:
        print("✗ Skipping - no results to process")
        return None
    
    try:
        lessons = agent.process_internal_lessons(results[:3], "tunnel construction")
        print(f"✓ Processed {len(lessons)} internal lessons")
        if lessons:
            print(f"  First lesson project: {lessons[0].get('project_name', 'N/A')}")
        return lessons
    except Exception as e:
        print(f"✗ Internal processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_external_analysis(agent, has_api_key):
    """Test external analysis (requires API key)."""
    print("\n" + "=" * 60)
    print("TEST 5: External Analysis (Perplexity API)")
    print("=" * 60)
    if not agent:
        print("✗ Skipping - agent not initialized")
        return None
    
    if not has_api_key:
        print("⚠ Skipping - API key not available")
        return None
    
    test_query = "tunnel construction risks"
    print(f"Query: '{test_query}'")
    print("  Note: This may take 30-60 seconds...")
    
    try:
        internal_context = "Test internal context from tunnel projects"
        analysis = agent.generate_external_analysis(test_query, internal_context)
        print(f"✓ External analysis completed")
        print(f"  Response length: {len(analysis)} characters")
        if analysis:
            print(f"  Preview: {analysis[:200]}...")
        return analysis
    except Exception as e:
        print(f"✗ External analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_answer(agent, has_api_key):
    """Test full answer generation."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Answer Generation")
    print("=" * 60)
    if not agent:
        print("✗ Skipping - agent not initialized")
        return None
    
    test_query = "What are the main risks in tunnel construction projects?"
    print(f"Query: '{test_query}'")
    print("  Note: This may take 1-2 minutes...")
    
    try:
        if not has_api_key:
            print("⚠ Skipping - API key required for full answer")
            return None
        
        answer = agent.answer(test_query)
        print(f"✓ Full answer generated")
        print(f"  Response length: {len(answer)} characters")
        
        # Check for required sections
        required_sections = [
            "EXECUTIVE SUMMARY",
            "KEY EXTERNAL RISKS",
            "SIMILAR GLOBAL PROJECTS",
            "INTERNAL LESSONS LEARNED",
            "BENCHMARKING",
            "RECOMMENDED ACTIONS",
            "FULL REFERENCE LIST"
        ]
        
        found_sections = []
        for section in required_sections:
            if section in answer.upper():
                found_sections.append(section)
        
        print(f"  Found {len(found_sections)}/{len(required_sections)} required sections:")
        for section in found_sections:
            print(f"    ✓ {section}")
        
        missing = set(required_sections) - set(found_sections)
        if missing:
            print(f"  Missing sections:")
            for section in missing:
                print(f"    ✗ {section}")
        
        return answer
    except Exception as e:
        print(f"✗ Full answer generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GLOBAL RISK KNOWLEDGE AGENT - TEST SUITE")
    print("=" * 60 + "\n")
    
    # Test 1: Initialization
    agent = test_agent_initialization()
    
    # Test 2: API Key
    has_api_key = test_api_key()
    
    # Test 3: Internal Search
    results = test_internal_search(agent)
    
    # Test 4: Internal Processing
    lessons = test_internal_processing(agent, results)
    
    # Test 5: External Analysis (if API key available)
    external_analysis = test_external_analysis(agent, has_api_key)
    
    # Test 6: Full Answer (if API key available)
    full_answer = test_full_answer(agent, has_api_key)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Agent initialized: {'✓' if agent else '✗'}")
    print(f"API key available: {'✓' if has_api_key else '✗'}")
    print(f"Internal search: {'✓' if results else '✗'}")
    print(f"Internal processing: {'✓' if lessons else '✗'}")
    print(f"External analysis: {'✓' if external_analysis else '⚠' if has_api_key else 'N/A'}")
    print(f"Full answer: {'✓' if full_answer else '⚠' if has_api_key else 'N/A'}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

