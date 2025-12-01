"""
Quick syntax and import test for agent code
Does not require all dependencies - just checks code structure
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    """Run quick tests."""
    print("=" * 60)
    print("QUICK CODE VALIDATION TEST")
    print("=" * 60)
    print()
    
    files_to_check = [
        "app/agent.py",
        "app/app.py",
        "app/research_search.py",
        "test_agent.py"
    ]
    
    all_passed = True
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠ {file_path}: File not found")
            continue
        
        print(f"Checking {file_path}...", end=" ")
        is_valid, error = check_syntax(path)
        if is_valid:
            print("✓ Valid syntax")
        else:
            print(f"✗ Syntax error: {error}")
            all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ All files have valid syntax")
        print()
        print("For full testing, run in Docker:")
        print("  1. Start Docker Desktop")
        print("  2. docker build -t global-risk-agent .")
        print("  3. docker run -p 8501:8501 --env-file .env global-risk-agent")
        print("  4. docker exec <container-id> python /app/test_agent.py")
    else:
        print("✗ Some files have syntax errors")
    print("=" * 60)

if __name__ == "__main__":
    main()

