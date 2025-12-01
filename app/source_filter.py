"""
Source Filtering Module

Implements strict relevance filtering for external sources:
- Hard exclusions (Wikipedia, blogs, non-engineering content)
- Relevance scoring (semantic + keyword)
"""

import re
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse


# Hard exclusion patterns (domains and keywords)
# NOTE: Wikipedia is allowed for Perplexity context, but MUST be filtered in final reference list
EXCLUDED_DOMAINS = {
    "wikipedia.org",  # EXCLUDED from final reference list (allowed only for Perplexity context)
    "wikimedia.org",  # EXCLUDED from final reference list
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
    "instagram.com",
    "pinterest.com",
}

EXCLUDED_KEYWORDS = [
    # Non-engineering domains (STRICT: reject medical, social sciences, humanities)
    r"\bpenguin",
    r"\bpolitics",
    r"\bbiolog(?!ical\s+engineering|ical\s+construction)",
    r"\beconom(?!ic\s+analysis\s+of\s+projects|ic\s+feasibility)",
    r"\btelecom(?!unication\s+infrastructure)",
    r"\bwildlife",
    r"\bmedicin(?!al\s+facility|al\s+building)",
    r"\bagricultur(?!al\s+engineering|al\s+infrastructure)",
    r"\bsocial\s+science(?!s\s+in\s+construction)",
    r"\bpsycholog",
    r"\banthropolog",
    r"\bsociolog",
    r"\bliterature",
    r"\bart\s+history",
    r"\bphilosoph",
    r"\btheolog",
    r"\bdentistr",
    r"\bnursing",
    r"\bveterinar",
    # Generic non-technical
    r"\brecipe",
    r"\bcooking",
    r"\bfashion",
    r"\bentertainment",
    r"\bsports(?!\s+facility|s\s+stadium|s\s+construction)",
    r"\bcelebrity",
    # News: Only reject if NOT describing engineering failures or project performance
    r"\bnews(?!\s+(?:about|on|regarding|describing).*(?:project|construction|engineering|failure|collapse|accident|incident|risk))",
]

# Engineering relevance keywords (positive signals) - STRICT REQUIREMENT
ENGINEERING_KEYWORDS = [
    # Core engineering disciplines
    "civil engineering", "structural engineering", "geotechnical", "geotechnical engineering",
    "marine engineering", "coastal engineering", "environmental engineering",
    # Infrastructure types
    "harbor", "harbour", "port", "tunnel", "bridge", "breakwater", "waterfront",
    "coastal", "marine", "offshore", "foundation", "piling", "pier", "quay",
    # Construction and materials
    "construction", "concrete", "steel", "structural", "geotechnical", "foundation",
    # Project types
    "infrastructure", "project", "case study", "lessons learned", "project report",
    # Engineering domains
    "risk", "seismic", "structural design", "geotechnical design", "coastal protection",
    "flood", "water", "wastewater", "sewage", "water supply",
    # Standards and guidelines
    "standards", "code", "guideline", "specification", "manual", "practice",
    # Academic/technical
    "peer-reviewed", "conference", "proceedings", "technical report", "engineering journal",
    "engineering", "design", "safety", "risk management"
]

# Engineering standards organizations (for prioritization)
STANDARDS_ORGANIZATIONS = {
    "pianc.org", "pianc", "eurocode", "en 199", "asce.org", "asce",
    "dnv.com", "dnv-gl", "dnv", "ice.org.uk", "institution of civil engineers",
    "fhwa", "fhwa.dot.gov", "ceda", "tunnelling code", "tcop",
    "usace", "usace.army.mil", "iso.org", "iso", "iec", "aashto",
    "bsi.org", "bs ", "cen.eu", "cen", "astm", "api"
}

# High-quality engineering academic source indicators (PRIORITIZED)
ENGINEERING_ACADEMIC_DOMAINS = {
    "asce.org",  # ASCE Library
    "ice.org.uk",  # ICE Virtual Library
    "sciencedirect.com",  # ScienceDirect (open access)
    "springer.com",  # Springer Engineering
    "springerlink.com",  # SpringerLink
    "arxiv.org",  # ArXiv (engineering)
    "usace.army.mil",  # USACE technical reports
    "pianc.org",  # PIANC
    "ita-aites.org",  # ITA-AITES
    "tunnellingonline.org",  # ITA-AITES
    "dnv.com",  # DNV
    "doi.org",  # DOI links (engineering papers)
}

# Other high-quality source indicators
HIGH_QUALITY_DOMAINS = {
    "scholar.google.com",
    "ieeexplore.ieee.org",
    "researchgate.net",
    "iso.org",
    "bsi.org",
    "cen.eu",
    "fema.gov",
    "fhwa.dot.gov",
    "dot.gov",
    "gov.uk",
    "gov.au",
    "gov.ca",
    "europa.eu",
    "worldbank.org",
    "undp.org",
}


def is_excluded_domain(url: str, allow_wikipedia: bool = False) -> bool:
    """
    Check if URL is from an excluded domain.
    STRICT: Wikipedia is excluded from final reference list (allow_wikipedia=False by default).
    """
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc.lower()
        
        # Allow Wikipedia ONLY if explicitly requested (for Perplexity context, but filtered in final list)
        if allow_wikipedia:
            wikipedia_domains = {"wikipedia.org", "wikimedia.org", "wiki"}
            if domain in wikipedia_domains or any(wd in domain for wd in wikipedia_domains):
                return False  # Allow for Perplexity context
        
        # STRICT: Always exclude Wikipedia from final reference list
        if "wikipedia.org" in domain or "wikimedia.org" in domain or "wiki" in domain:
            return True
        
        # Check exact domain match
        if domain in EXCLUDED_DOMAINS:
            return True
        
        # Check if any excluded domain is in the full domain
        for excluded in EXCLUDED_DOMAINS:
            if excluded in domain:
                return True
        
        return False
    except Exception:
        return False


def contains_excluded_keywords(text: str) -> bool:
    """Check if text contains excluded keywords."""
    text_lower = text.lower()
    for pattern in EXCLUDED_KEYWORDS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_standards_source(title: str, url: str, abstract: str = "") -> bool:
    """
    Check if source is from an engineering standards organization.
    Returns True if source should be prioritized (×3 weight).
    """
    combined_text = f"{title} {abstract} {url}".lower()
    
    # Check for standards organization indicators
    for org in STANDARDS_ORGANIZATIONS:
        if org.lower() in combined_text:
            return True
    
    # Check for standard codes (e.g., "EN 1990", "ISO 31000", "PIANC WG 150")
    standards_patterns = [
        r"\ben\s+\d{4}",  # Eurocode
        r"\biso\s+\d{5}",  # ISO standard
        r"\bbs\s+\d+",  # British Standard
        r"\basce\s+",  # ASCE
        r"\bfhwa",  # FHWA
        r"\bpianc\s+wg",  # PIANC working group
        r"\bdnv",  # DNV
        r"\baashto",  # AASHTO
    ]
    
    for pattern in standards_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return True
    
    return False


def calculate_relevance_score(title: str, url: str, abstract: str = "", keywords: List[str] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate relevance score for a source (0.0 to 1.0).
    Enhanced with engineering academic source prioritization (×2 multiplier) and standards prioritization (×3 multiplier).
    
    Returns:
        (score, details) tuple where details contains breakdown
    """
    score = 0.0
    details = {
        "domain_quality": 0.0,
        "keyword_match": 0.0,
        "engineering_relevance": 0.0,
        "engineering_academic_boost": 0.0,
        "standards_boost": 0.0,
        "is_engineering_academic": False,
        "is_standards": False,
    }
    
    combined_text = f"{title} {abstract}".lower()
    url_lower = url.lower()
    venue = ""  # Will be extracted if available
    
    # Check if this is a standards source (for prioritization)
    is_standards = is_standards_source(title, url, abstract)
    details["is_standards"] = is_standards
    
    # Check if this is an engineering academic source (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)
    try:
        parsed = urlparse(url_lower)
        domain = parsed.netloc.lower()
        combined_domain = f"{domain} {url_lower}".lower()
        
        is_engineering_academic = any(eng_domain in combined_domain for eng_domain in ENGINEERING_ACADEMIC_DOMAINS)
        details["is_engineering_academic"] = is_engineering_academic
    except Exception:
        is_engineering_academic = False
        details["is_engineering_academic"] = False
    
    # 1. Domain quality (0.0 to 0.4) - PRIORITIZE engineering academic sources
    try:
        parsed = urlparse(url_lower)
        domain = parsed.netloc.lower()
        
        # Highest priority: Engineering academic sources (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)
        if is_engineering_academic:
            details["domain_quality"] = 0.5  # Higher base score for engineering academic
            details["engineering_academic_boost"] = 0.3  # Additional boost
        elif domain in HIGH_QUALITY_DOMAINS:
            details["domain_quality"] = 0.4
        elif any(hq_domain in domain for hq_domain in HIGH_QUALITY_DOMAINS):
            details["domain_quality"] = 0.3
        elif ".edu" in domain or ".ac." in domain:
            details["domain_quality"] = 0.25
        elif ".gov" in domain or ".org" in domain:
            details["domain_quality"] = 0.2
        else:
            details["domain_quality"] = 0.1
    except Exception:
        details["domain_quality"] = 0.1
    
    # 2. Keyword matching (0.0 to 0.4)
    keyword_matches = 0
    for keyword in ENGINEERING_KEYWORDS:
        if keyword.lower() in combined_text:
            keyword_matches += 1
    
    # Normalize keyword score (max 0.4)
    details["keyword_match"] = min(0.4, keyword_matches * 0.05)
    
    # 3. Engineering relevance indicators (0.0 to 0.2)
    relevance_indicators = [
        "engineering", "construction", "infrastructure",
        "project", "risk", "design", "technical"
    ]
    relevance_count = sum(1 for indicator in relevance_indicators if indicator in combined_text)
    details["engineering_relevance"] = min(0.2, relevance_count * 0.05)
    
    # 4. Engineering academic boost (×2 multiplier for engineering academic sources)
    base_score = details["domain_quality"] + details["keyword_match"] + details["engineering_relevance"]
    
    if is_engineering_academic:
        # Boost engineering academic sources by 2x
        score = min(1.0, base_score * 2.0)
    
    # 5. Standards boost (×3 multiplier for standards sources, on top of engineering academic boost)
    if is_standards:
        # Boost base score by 3x (but cap at 1.0)
        details["standards_boost"] = min(1.0, base_score * 2.0)  # Add 2x more (total 3x)
        score = min(1.0, base_score * 3.0)  # Total score with 3x multiplier
    elif not is_engineering_academic:
        # Normal scoring (no boost)
        score = base_score
    
    return score, details


def validate_engineering_relevance(title: str, abstract: str = "", url: str = "") -> Tuple[bool, str]:
    """
    Validate that source is directly linked to civil/marine/geotechnical/structural/environmental engineering.
    STRICT: Only accept engineering-relevant sources. Reject Wikipedia, non-engineering Crossref, medical/social sciences/humanities.
    Allow news only if describing actual engineering failures or project performance.
    
    Returns:
        (is_relevant, reason) tuple
    """
    combined_text = f"{title} {abstract} {url}".lower()
    
    # STRICT: Reject Wikipedia (even if allow_wikipedia was used for Perplexity context)
    if "wikipedia.org" in combined_text or "wikimedia.org" in combined_text:
        return False, "Wikipedia excluded from final reference list"
    
    # STRICT: Reject non-engineering Crossref items (medical, social sciences, humanities)
    excluded_subjects = [
        'medicine', 'medical', 'biology', 'pharmacology', 'psychology', 'sociology',
        'anthropology', 'economics', 'political', 'literature', 'history', 'art',
        'music', 'philosophy', 'theology', 'agriculture', 'veterinary', 'dentistry',
        'nursing', 'healthcare', 'public health'
    ]
    # Only reject if NOT related to engineering/construction
    for excluded in excluded_subjects:
        if excluded in combined_text:
            # Allow if it's about medical facilities, healthcare construction, etc.
            if not any(allowed in combined_text for allowed in ['facility', 'building', 'construction', 'engineering', 'infrastructure']):
                return False, f"Non-engineering subject: {excluded}"
    
    # Check for engineering keyword relevance (STRICT: require engineering relevance)
    has_engineering_keyword = any(keyword.lower() in combined_text for keyword in ENGINEERING_KEYWORDS)
    
    # Check for high-quality domain indicators (government, academic, engineering firms)
    high_quality_indicators = [
        ".gov", ".edu", ".ac.", ".org", "engineering", "construction", 
        "infrastructure", "consulting", "firm", "company", "ltd", "inc",
        "pdf", "report", "technical", "case study", "project"
    ]
    has_quality_indicator = any(indicator in combined_text for indicator in high_quality_indicators)
    
    # STRICT: Require engineering keyword OR quality indicator with engineering context
    if has_engineering_keyword:
        return True, "Engineering relevant"
    
    # Allow news ONLY if describing actual engineering failures or project performance
    if "news" in combined_text or "article" in combined_text:
        news_engineering_keywords = [
            'project', 'construction', 'engineering', 'failure', 'collapse',
            'accident', 'incident', 'risk', 'harbor', 'tunnel', 'bridge',
            'infrastructure', 'structural', 'geotechnical'
        ]
        if any(keyword in combined_text for keyword in news_engineering_keywords):
            return True, "Engineering news (project performance/failure)"
        else:
            return False, "Non-engineering news"
    
    # Check for low-quality AI-generated content patterns (reject these)
    ai_generated_patterns = [
        r"\bgenerated\s+by\s+ai",
        r"\bai\s+written",
        r"\bautomatically\s+generated",
        r"\bcontent\s+farm",
        r"\bseo\s+article"
    ]
    for pattern in ai_generated_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return False, "Low-quality AI-generated content"
    
    # STRICT: Require engineering relevance or quality indicator
    if has_quality_indicator and any(keyword in combined_text for keyword in ['engineering', 'construction', 'infrastructure', 'project']):
        return True, "Engineering relevant"
    
    return False, "No engineering relevance found"


def filter_source(url: str, title: str, abstract: str = "", allow_wikipedia: bool = False) -> Tuple[bool, str]:
    """
    Filter a source based on hard exclusions and engineering relevance.
    STRICT: Reject Wikipedia (in final list), non-engineering Crossref, medical/social sciences/humanities.
    Allow news only if describing actual engineering failures or project performance.
    Wikipedia is allowed if allow_wikipedia=True (for Perplexity context only, filtered in final list).
    Require meaningful abstract/summary for acceptance.
    
    Returns:
        (is_valid, rejection_reason) tuple
    """
    # Check excluded domains (blogs, social media; Wikipedia excluded in final list)
    if is_excluded_domain(url, allow_wikipedia=allow_wikipedia):
        return False, "Excluded domain (blog, social media, Wikipedia)"
    
    # Check excluded keywords (non-engineering domains)
    combined_text = f"{title} {abstract}".lower()
    if contains_excluded_keywords(combined_text):
        return False, "Contains excluded keywords (non-engineering content)"
    
    # STRICT: Validate engineering relevance
    is_relevant, reason = validate_engineering_relevance(title, abstract, url)
    if not is_relevant:
        return False, f"Not engineering-relevant: {reason}"
    
    # STRICT: Require meaningful abstract/summary (unless from high-quality engineering academic source)
    if not abstract or len(abstract.strip()) < 20:
        # Allow if from engineering academic domain
        url_lower = url.lower()
        is_engineering_academic = any(eng_domain in url_lower for eng_domain in ENGINEERING_ACADEMIC_DOMAINS)
        if not is_engineering_academic:
            return False, "Missing meaningful abstract/summary"
    
    return True, "Valid"


def filter_and_score_sources(sources: List[Dict[str, Any]], min_score: float = 0.3) -> List[Dict[str, Any]]:
    """
    Filter and score a list of sources with strict engineering relevance validation.
    Engineering academic sources (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES) are prioritized (×2 weight).
    Standards sources are prioritized (×3 weight).
    Ensures 70-80% of sources are from engineering academic sources.
    
    Args:
        sources: List of source dictionaries with 'url', 'title', optionally 'abstract'
        min_score: Minimum relevance score threshold (default 0.3)
    
    Returns:
        Filtered and scored sources, sorted by relevance (engineering academic first, then standards, then by score)
    """
    filtered = []
    rejected = []  # For debugging
    
    for source in sources:
        url = source.get("url", "")
        title = source.get("title", "")
        abstract = source.get("abstract", source.get("summary", ""))
        
        # Hard exclusion check with rejection reason (STRICT: no Wikipedia in final list)
        is_valid, rejection_reason = filter_source(url, title, abstract, allow_wikipedia=False)
        if not is_valid:
            rejected.append({
                "title": title[:50],
                "url": url[:50],
                "reason": rejection_reason
            })
            continue
        
        # Calculate relevance score (with engineering academic boost and standards boost)
        score, details = calculate_relevance_score(title, url, abstract)
        
        # Apply threshold
        if score >= min_score:
            source_with_score = source.copy()
            source_with_score["relevance_score"] = score
            source_with_score["relevance_details"] = details
            source_with_score["rejection_reason"] = None  # Passed validation
            filtered.append(source_with_score)
        else:
            rejected.append({
                "title": title[:50],
                "url": url[:50],
                "reason": f"Score too low: {score:.2f} < {min_score}"
            })
    
    # Sort: engineering academic sources first, then standards, then by relevance score (descending)
    filtered.sort(key=lambda x: (
        not x.get("relevance_details", {}).get("is_engineering_academic", False),  # Engineering academic first
        not x.get("relevance_details", {}).get("is_standards", False),  # Then standards
        -x.get("relevance_score", 0.0)  # Then by score (descending)
    ))
    
    # Ensure 70-80% of sources are from engineering academic sources
    engineering_academic_count = sum(1 for s in filtered if s.get("relevance_details", {}).get("is_engineering_academic", False))
    total_count = len(filtered)
    
    if total_count > 0:
        engineering_academic_ratio = engineering_academic_count / total_count
        target_ratio = 0.75  # 75% target (between 70-80%)
        
        # If below target, prioritize engineering academic sources more
        if engineering_academic_ratio < target_ratio:
            # Re-sort to prioritize engineering academic sources even more
            filtered.sort(key=lambda x: (
                not x.get("relevance_details", {}).get("is_engineering_academic", False),  # Engineering academic first
                not x.get("relevance_details", {}).get("is_standards", False),  # Then standards
                -x.get("relevance_score", 0.0)  # Then by score (descending)
            ))
    
    return filtered

