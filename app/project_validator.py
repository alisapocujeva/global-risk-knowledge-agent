"""
Project Validation Module

Validates that global project examples are real engineering projects,
not fictional or conceptual designs.
STRICT: Requires projects to come from engineering-authoritative sources.
"""

import re
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse


# Real project name patterns (must contain location or infrastructure type)
PROJECT_VALIDATION_PATTERNS = {
    "location_indicators": [
        r"\b(singapore|singapore|rotterdam|hamburg|copenhagen|london|paris|new york|los angeles|tokyo|hong kong|shanghai|sydney|melbourne|oslo|stockholm|helsinki|copenhagen|amsterdam|brussels|berlin|munich|vienna|zurich|milan|rome|madrid|barcelona|lisbon|dublin|edinburgh|glasgow|manchester|birmingham|boston|chicago|san francisco|seattle|vancouver|toronto|montreal|dubai|doha|riyadh|jeddah|istanbul|ankara|moscow|saint petersburg|beijing|guangzhou|shenzhen|seoul|taipei|bangkok|kuala lumpur|jakarta|manila|ho chi minh|hanoi)",
        r"\b(norway|sweden|denmark|finland|netherlands|germany|france|uk|united kingdom|spain|italy|switzerland|austria|belgium|poland|czech|portugal|greece|ireland|iceland)",
        r"\b(usa|united states|canada|australia|new zealand|japan|china|south korea|taiwan|thailand|malaysia|indonesia|philippines|vietnam|india|pakistan|bangladesh|saudi arabia|uae|qatar|kuwait|oman|bahrain)",
        r"\b(bay|channel|strait|river|harbor|harbour|port|airport|metro|subway|tunnel|bridge|highway|motorway|expressway)",
    ],
    "infrastructure_types": [
        r"\b(port|harbor|harbour|terminal|quay|breakwater|pier|wharf)",
        r"\b(tunnel|metro|subway|underground|tbm)",
        r"\b(bridge|viaduct|overpass|crossing)",
        r"\b(airport|runway|terminal)",
        r"\b(highway|motorway|expressway|road|pavement)",
        r"\b(railway|rail|metro|subway|light rail)",
        r"\b(treatment plant|water|wastewater|sewage)",
        r"\b(power plant|energy|wind farm|solar)",
        r"\b(offshore|platform|wind|oil|gas)",
    ],
    "project_indicators": [
        r"\b(project|construction|development|expansion|upgrade|extension|phase)",
        r"\b(mega|major|large-scale|infrastructure)",
    ]
}

# Patterns that indicate conceptual/non-real projects (reject)
REJECTION_PATTERNS = [
    r"\b(conceptual|proposed|planned|future|potential|hypothetical|theoretical)",
    r"\b(example|sample|case study|illustration)\s+(project|design)",
    r"\b(small|private|residential|house|apartment)\s+(construction|building)",
    r"\b(not\s+)?(real|actual|existing)",
]

# Engineering-authoritative source domains (REQUIRED for project sources)
ENGINEERING_AUTHORITATIVE_DOMAINS = {
    "asce.org",  # ASCE Library
    "ice.org.uk",  # ICE Virtual Library
    "sciencedirect.com",  # ScienceDirect
    "springer.com",  # Springer Engineering
    "springerlink.com",  # SpringerLink
    "arxiv.org",  # ArXiv (engineering)
    "usace.army.mil",  # USACE technical reports
    "pianc.org",  # PIANC
    "ita-aites.org",  # ITA-AITES
    "tunnellingonline.org",  # ITA-AITES
    "dnv.com",  # DNV
    "doi.org",  # DOI links (engineering papers)
    "scholar.google.com",  # Google Scholar
    "ieeexplore.ieee.org",  # IEEE
    "researchgate.net",  # ResearchGate (engineering papers)
    "gov.uk",  # UK government
    "gov.au",  # Australian government
    "gov.ca",  # Canadian government
    "dot.gov",  # US DOT
    "fhwa.dot.gov",  # FHWA
    "fema.gov",  # FEMA
    "europa.eu",  # EU
    "worldbank.org",  # World Bank
    "undp.org",  # UNDP
}

# Low-authority domains (REJECT projects from these)
LOW_AUTHORITY_DOMAINS = {
    "wikipedia.org",
    "wikimedia.org",
    "blogspot.com",
    "wordpress.com",
    "medium.com",
    "tumblr.com",
    "reddit.com",
    "quora.com",
    "facebook.com",
    "twitter.com",
    "linkedin.com",
    "youtube.com",
}


def is_engineering_authoritative_source(source_url: str) -> Tuple[bool, str]:
    """
    Check if a project source URL is from an engineering-authoritative source.
    
    Args:
        source_url: URL of the source
    
    Returns:
        (is_authoritative, reason) tuple
    """
    if not source_url or not source_url.startswith("http"):
        return False, "Invalid or missing URL"
    
    try:
        parsed = urlparse(source_url.lower())
        domain = parsed.netloc.lower()
        
        # REJECT low-authority domains (blogs, social media, Wikipedia)
        if any(low_domain in domain for low_domain in LOW_AUTHORITY_DOMAINS):
            return False, f"Low-authority source: {domain}"
        
        # ACCEPT engineering-authoritative domains
        if any(auth_domain in domain for auth_domain in ENGINEERING_AUTHORITATIVE_DOMAINS):
            return True, f"Engineering-authoritative source: {domain}"
        
        # ACCEPT government/educational domains (.gov, .edu, .ac.)
        if ".gov" in domain or ".edu" in domain or ".ac." in domain:
            return True, f"Government/educational source: {domain}"
        
        # ACCEPT if URL contains engineering keywords (for conference proceedings, technical reports)
        url_lower = source_url.lower()
        engineering_indicators = [
            "conference", "proceedings", "technical report", "case study",
            "engineering", "construction", "infrastructure", "project report"
        ]
        if any(indicator in url_lower for indicator in engineering_indicators):
            return True, f"Engineering-related source: {domain}"
        
        # REJECT if no clear engineering authority
        return False, f"Not from engineering-authoritative source: {domain}"
    
    except Exception:
        return False, "Invalid URL format"


def validate_project_name(project_name: str, project_type: str = "") -> Tuple[bool, str]:
    """
    Validate that a project name represents a real engineering project.
    RELAXED: Accept any result that includes:
    - a project name OR
    - a structure type (harbor, tunnel, bridge, foundation, etc.) OR
    - a major contractor/authority OR
    - a location AND a risk event
    
    Do NOT require cost numbers to classify as a project.
    Allow engineering case studies, conference papers, and official reports.
    
    Args:
        project_name: Name of the project
        project_type: Optional project type for context
    
    Returns:
        (is_valid, reason) tuple
    """
    if not project_name or len(project_name.strip()) < 3:
        return False, "Project name too short or empty"
    
    name_lower = project_name.lower()
    
    # Reject if matches rejection patterns (conceptual, proposed, etc.)
    for pattern in REJECTION_PATTERNS:
        if re.search(pattern, name_lower):
            return False, f"Matches rejection pattern: {pattern}"
    
    # RELAXED: Accept if has ANY of these indicators
    has_location = False
    has_infrastructure = False
    has_contractor = False
    has_risk_event = False
    
    # Check location indicators
    for pattern in PROJECT_VALIDATION_PATTERNS["location_indicators"]:
        if re.search(pattern, name_lower, re.IGNORECASE):
            has_location = True
            break
    
    # Check infrastructure type indicators
    for pattern in PROJECT_VALIDATION_PATTERNS["infrastructure_types"]:
        if re.search(pattern, name_lower, re.IGNORECASE):
            has_infrastructure = True
            break
    
    # Check for major contractors/authorities
    contractor_patterns = [
        r"\b(boskalis|van oord|jan de nul|royal haskoning|witteveen\+bos|arcadis|fugro|aecom|jacobs|bechtel|fluor|kbr|mott macdonald|atkins|wsp|arup|buro happold|ramboll|sweco|afry|niras|coast|port authority|harbor authority|metro authority)",
        r"\b(ministry|department|agency|authority|commission|corporation)",
    ]
    for pattern in contractor_patterns:
        if re.search(pattern, name_lower, re.IGNORECASE):
            has_contractor = True
            break
    
    # Check for risk events (indicating real project with issues)
    risk_event_patterns = [
        r"\b(delay|overrun|failure|collapse|accident|incident|problem|issue|risk|hazard)",
        r"\b(cost|schedule|time|budget|safety|quality)",
    ]
    for pattern in risk_event_patterns:
        if re.search(pattern, name_lower, re.IGNORECASE):
            has_risk_event = True
            break
    
    # RELAXED: Accept if has project name OR structure type OR contractor OR (location AND risk event)
    if has_infrastructure or has_contractor or (has_location and has_risk_event):
        return True, "Valid real project"
    
    # Also check project indicators
    has_project_indicator = any(
        re.search(pattern, name_lower, re.IGNORECASE)
        for pattern in PROJECT_VALIDATION_PATTERNS["project_indicators"]
    )
    
    if has_project_indicator:
        return True, "Valid project with indicators"
    
    return False, "Does not match real project pattern"


def extract_and_validate_projects_from_text(text: str, project_type: str = "") -> List[Dict[str, Any]]:
    """
    Extract project names from text and validate them.
    STRICT: Requires projects to come from engineering-authoritative sources.
    Rejects projects from blogs or low-authority news (unless describing actual engineering failures/project performance).
    
    Args:
        text: Text containing project references
        project_type: Project type for context
    
    Returns:
        List of validated project dictionaries
    """
    validated_projects = []
    
    # Look for project table rows (markdown format)
    lines = text.split("\n")
    in_table = False
    
    for line in lines:
        if "SIMILAR GLOBAL PROJECTS" in line.upper():
            in_table = True
            continue
        
        if in_table and line.strip().startswith("|"):
            if "---" in line or "Region" in line:
                continue
            
            # Parse table row
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) >= 2:
                project_name = cells[1] if len(cells) > 1 else ""
                source_url = ""
                
                # Extract source URL from last column (Source column)
                if len(cells) >= 7:  # Project | Region | Risk Type | Impact | Mitigation | Year | Source
                    source_url = cells[-1]  # Last column is Source
                elif len(cells) >= 6:  # Alternative format
                    source_url = cells[-1]
                
                if project_name:
                    # Validate project name
                    is_valid, reason = validate_project_name(project_name, project_type)
                    if not is_valid:
                        continue
                    
                    # STRICT: Validate source is from engineering-authoritative source
                    if source_url:
                        is_authoritative, source_reason = is_engineering_authoritative_source(source_url)
                        if not is_authoritative:
                            # REJECT if not from engineering-authoritative source
                            continue
                    else:
                        # REJECT if no source URL provided
                        continue
                    
                    validated_projects.append({
                        "name": project_name,
                        "region": cells[0] if len(cells) > 0 else "",
                        "type": cells[2] if len(cells) > 2 else project_type,
                        "source_url": source_url,
                        "validated": True
                    })
    
    return validated_projects


def count_validated_projects(text: str, project_type: str = "") -> int:
    """
    Count number of validated real projects in text.
    
    Args:
        text: Text containing project references
        project_type: Project type for context
    
    Returns:
        Number of validated projects
    """
    validated = extract_and_validate_projects_from_text(text, project_type)
    return len(validated)

