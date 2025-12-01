"""
Citation Validation Module

Validates that citations in project tables reference actual sources
from the FULL REFERENCE LIST.
"""

import re
from typing import List, Dict, Any, Tuple


def extract_citation_numbers(text: str) -> List[int]:
    """
    Extract citation numbers from text (e.g., [1], [2], [3]).
    
    Args:
        text: Text containing citations
    
    Returns:
        List of citation numbers found
    """
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


def extract_reference_numbers(reference_list: List[str]) -> List[int]:
    """
    Extract reference numbers from reference list.
    
    Args:
        reference_list: List of reference strings like "[1] Title — URL"
    
    Returns:
        List of reference numbers found
    """
    ref_numbers = []
    for ref in reference_list:
        match = re.match(r'\[(\d+)\]', ref.strip())
        if match:
            ref_numbers.append(int(match.group(1)))
    return ref_numbers


def validate_project_citations(project_row: str, reference_list: List[str]) -> Tuple[bool, str, List[int]]:
    """
    Validate that citations in a project table row reference actual sources.
    
    Args:
        project_row: Table row string (markdown format)
        reference_list: List of reference strings from FULL REFERENCE LIST
    
    Returns:
        (is_valid, reason, citation_numbers) tuple
    """
    # Extract citation numbers from project row
    citation_numbers = extract_citation_numbers(project_row)
    
    if not citation_numbers:
        return False, "No citations found", []
    
    if len(citation_numbers) < 1 or len(citation_numbers) > 3:
        return False, f"Invalid citation count: {len(citation_numbers)} (must be 1-3)", citation_numbers
    
    # Extract reference numbers from reference list
    ref_numbers = extract_reference_numbers(reference_list)
    
    if not ref_numbers:
        return False, "No references found in reference list", citation_numbers
    
    # Check if all citations exist in reference list
    missing_citations = [c for c in citation_numbers if c not in ref_numbers]
    
    if missing_citations:
        return False, f"Citations not found in reference list: {missing_citations}", citation_numbers
    
    return True, "Valid", citation_numbers


def validate_all_project_citations(project_table_text: str, reference_list: List[str]) -> Dict[str, Any]:
    """
    Validate citations for all projects in the table.
    
    Args:
        project_table_text: Full text containing project table
        reference_list: List of reference strings from FULL REFERENCE LIST
    
    Returns:
        Dictionary with validation results
    """
    lines = project_table_text.split("\n")
    in_table = False
    results = {
        "valid_projects": 0,
        "invalid_projects": 0,
        "projects_without_citations": 0,
        "details": []
    }
    
    for line in lines:
        if "SIMILAR GLOBAL PROJECTS" in line.upper():
            in_table = True
            continue
        if in_table and line.strip().startswith("|"):
            if "---" not in line and "Region" not in line and "Project Name" not in line.upper():
                is_valid, reason, citations = validate_project_citations(line, reference_list)
                if is_valid:
                    results["valid_projects"] += 1
                else:
                    results["invalid_projects"] += 1
                    if "No citations found" in reason:
                        results["projects_without_citations"] += 1
                results["details"].append({
                    "row": line[:100],  # First 100 chars
                    "valid": is_valid,
                    "reason": reason,
                    "citations": citations
                })
        if in_table and line.strip().startswith("##"):
            break
    
    return results

