#!/bin/bash
# Test script to run inside Docker container

echo "=========================================="
echo "GLOBAL RISK KNOWLEDGE AGENT - TEST SUITE"
echo "=========================================="
echo ""

# Test 1: Check if agent module can be imported
echo "TEST 1: Module Import"
echo "-------------------"
python -c "from app.agent import RiskIntelligenceAgent; print('✓ Agent module imported successfully')" || echo "✗ Failed to import agent module"
echo ""

# Test 2: Check API key
echo "TEST 2: API Key Check"
echo "-------------------"
python -c "
import os
from app.agent import get_perplexity_api_key
try:
    key = get_perplexity_api_key()
    print(f'✓ API key found (length: {len(key)})')
except RuntimeError as e:
    print(f'⚠ API key not set: {e}')
" || echo "✗ API key check failed"
echo ""

# Test 3: Initialize agent
echo "TEST 3: Agent Initialization"
echo "-------------------"
python -c "
from app.agent import RiskIntelligenceAgent
try:
    agent = RiskIntelligenceAgent()
    print('✓ Agent initialized successfully')
except Exception as e:
    print(f'✗ Agent initialization failed: {e}')
" || echo "✗ Agent initialization failed"
echo ""

# Test 4: Internal search
echo "TEST 4: Internal Document Search"
echo "-------------------"
python -c "
import sys
sys.path.insert(0, '/app')
from app.agent import RiskIntelligenceAgent
try:
    agent = RiskIntelligenceAgent()
    results = agent.internal_search('tunnel construction risks')
    print(f'✓ Internal search completed: {len(results)} results')
    if results:
        print(f'  First result source: {results[0].get(\"source\", \"N/A\")[:50]}...')
except Exception as e:
    print(f'✗ Internal search failed: {e}')
    import traceback
    traceback.print_exc()
" || echo "✗ Internal search test failed"
echo ""

# Test 5: Check if Streamlit app can start
echo "TEST 5: Streamlit App Check"
echo "-------------------"
python -c "
import sys
sys.path.insert(0, '/app')
try:
    import streamlit
    print('✓ Streamlit available')
except ImportError:
    print('✗ Streamlit not available')
" || echo "✗ Streamlit check failed"
echo ""

echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Run 'docker exec <container-id> python -c \"from app.agent import RiskIntelligenceAgent; agent = RiskIntelligenceAgent(); print(agent.answer('test query'))\"' for full test"
echo ""

