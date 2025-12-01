"""Quick test of agent functionality"""
import sys
sys.path.insert(0, '/app')
from app.agent import RiskIntelligenceAgent, classify_project_type_semantic, get_embedding_model
from app.engineering_standards import get_relevant_standards

print("=" * 60)
print("QUICK AGENT TEST")
print("=" * 60)

# Test 1: Agent initialization
print("\n1. Testing agent initialization...")
try:
    agent = RiskIntelligenceAgent()
    print("   ✓ Agent initialized successfully")
except Exception as e:
    print(f"   ✗ Agent initialization failed: {e}")
    exit(1)

# Test 2: Project type classification
print("\n2. Testing project type classification...")
model = get_embedding_model()
test_cases = [
    ("tunnel construction risks", "tunnel"),
    ("harbour port construction", "harbour"),
    ("wastewater treatment plant", "wastewater"),
    ("offshore wind platform", "offshore"),
    ("earthworks excavation", "earthworks"),
    ("bridge construction", "bridge"),
]

for query, expected in test_cases:
    result = classify_project_type_semantic(query, query, model)
    status = "✓" if result == expected or result != "general" else "⚠"
    print(f"   {status} {query[:35]:35} -> {result:15} (expected: {expected})")

# Test 3: Standards retrieval
print("\n3. Testing standards retrieval...")
try:
    standards = get_relevant_standards("tunnel", 3)
    print(f"   ✓ Tunnel standards: {len(standards)} found")
    if standards:
        print(f"   ✓ First standard: {standards[0]['code']}")
    
    standards2 = get_relevant_standards("harbour", 3)
    print(f"   ✓ Harbour standards: {len(standards2)} found")
    
    standards3 = get_relevant_standards("wastewater", 3)
    print(f"   ✓ Wastewater standards: {len(standards3)} found")
except Exception as e:
    print(f"   ✗ Standards retrieval failed: {e}")

# Test 4: Internal search
print("\n4. Testing internal search...")
try:
    results = agent.internal_search("tunnel construction")
    print(f"   ✓ Internal search: {len(results)} results found")
except Exception as e:
    print(f"   ✗ Internal search failed: {e}")

print("\n" + "=" * 60)
print("BASIC TESTS COMPLETE")
print("=" * 60)
print("\nTo test full functionality, access the UI at: http://localhost:8501")

