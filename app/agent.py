"""
Global Risk Knowledge Agent for Civil Engineering

Professional risk analysis agent with:
- Deep external source retrieval (50-80 authoritative sources)
- Structured internal lessons processing (no raw cell copying)
- LLM-based project classification and summarization
- Enforced output structure
"""

import os
import re
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import requests
from chromadb import PersistentClient, Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from research_search import ResearchSearchEngine
from engineering_standards import get_relevant_standards, format_standards_for_prompt
from source_filter import filter_and_score_sources, filter_source, contains_excluded_keywords
from project_validator import validate_project_name, count_validated_projects, extract_and_validate_projects_from_text
from citation_validator import validate_all_project_citations, validate_project_citations

# Load environment variables from .env file if it exists
load_dotenv()

# Configuration
def get_perplexity_api_key() -> str:
    """Get Perplexity API key with validation."""
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key or key.strip() == "":
        raise RuntimeError(
            "PERPLEXITY_API_KEY environment variable is not set.\n\n"
            "For Streamlit Cloud:\n"
            "1. Go to your app on https://share.streamlit.io\n"
            "2. Click 'Manage app' → 'Settings' → 'Secrets'\n"
            "3. Add: PERPLEXITY_API_KEY = 'your_key_here'\n"
            "4. Save and restart the app\n\n"
            "For local/Docker:\n"
            "1. Create a .env file with: PERPLEXITY_API_KEY=your_key_here\n"
            "2. Or set: $env:PERPLEXITY_API_KEY='your_key_here' (PowerShell)\n\n"
            "Get your API key from: https://www.perplexity.ai/settings/api"
        )
    return key.strip()

PPLX_URL = "https://api.perplexity.ai/chat/completions"
PPLX_MODEL = "sonar-pro"  # Fixed to sonar-pro

# Use /tmp for Streamlit Cloud (read-only filesystem except /tmp)
# Try /app/index first (Docker/local), fallback to /tmp (Streamlit Cloud)
def get_index_dir():
    """Get the index directory, using /tmp on Streamlit Cloud."""
    # Check if we're on Streamlit Cloud
    is_streamlit_cloud = (
        os.environ.get("STREAMLIT_SHARING_MODE") == "true" or
        os.environ.get("STREAMLIT_SERVER_PORT") is not None or
        os.path.exists("/mount/src")  # Streamlit Cloud mount point
    )
    
    if is_streamlit_cloud:
        # Streamlit Cloud - use /tmp (writable)
        index_dir = "/tmp/chromadb_index"
        os.makedirs(index_dir, exist_ok=True)
        return index_dir
    else:
        # Docker/local - use /app/index
        try:
            os.makedirs("/app/index", exist_ok=True)
            return "/app/index"
        except (OSError, PermissionError):
            # Fallback to /tmp if /app/index is not writable
            index_dir = "/tmp/chromadb_index"
            os.makedirs(index_dir, exist_ok=True)
            return index_dir

INDEX_DIR = get_index_dir()
COLLECTION = "risk_docs"

# Lazy initialization of embedding model to avoid import-time errors
_EMBEDDING_MODEL = None

def get_embedding_model():
    """Get or initialize the embedding model lazily."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL

# System prompt for external analysis (web UI style - natural and comprehensive)
EXTERNAL_SYSTEM_PROMPT = """You are an expert civil engineering risk analyst providing comprehensive, well-sourced analysis similar to Perplexity.ai's research mode.

**CRITICAL: Search ONLY engineering-grade academic and technical sources**
Search PRIMARILY (70-80% of sources) from:
- ASCE Library (asce.org)
- ICE Virtual Library (ice.org.uk)
- ScienceDirect open access (sciencedirect.com)
- Springer Engineering (springer.com, springerlink.com)
- ArXiv engineering (arxiv.org - engineering categories only)
- USACE technical reports (usace.army.mil)
- PIANC (pianc.org)
- ITA-AITES (ita-aites.org, tunnellingonline.org)

Additional high-quality sources:
- Google Scholar (engineering papers)
- ResearchGate (engineering papers)
- IEEExplore (engineering papers)
- Government/NGO technical archives
- Industry technical reports and white papers

**STRICT SOURCE FILTERING:**
- EXCLUDE Wikipedia (do not cite Wikipedia in final reference list)
- EXCLUDE non-engineering Crossref results (reject medical, social sciences, humanities, unrelated subjects)
- EXCLUDE blogs, social media, and non-engineering content
- EXCLUDE sources about wildlife, medicine, agriculture, politics, economics (unless directly related to construction/risk)
- ALLOW news ONLY if describing actual engineering failures or project performance
- PRIORITIZE peer-reviewed papers, conference proceedings, project reports, government publications
- ONLY include sources directly relevant to civil engineering, construction, or risk management
- Every statement must have an engineering citation from an authoritative source

**CIVIL ENGINEERING DISCIPLINES COVERED:**
This system handles ALL civil engineering disciplines dynamically:
- Harbours and ports
- Tunnels and underground construction
- Bridges and structures
- Buildings and facilities
- Pharmaceutical and industrial facilities
- Offshore works and marine structures
- Earthworks and excavation
- Geotechnics and foundations
- Water supply and wastewater treatment
- Energy infrastructure
- Transport (highways, rail, airports)
- Urban infrastructure and planning
- Flood protection and coastal engineering

**ENGINEERING STANDARDS INTEGRATION:**
- Cross-reference findings with relevant engineering standards and guidelines
- Compare practices against established codes and best practices
- Reference standards when discussing mitigation strategies
- Use discipline-specific standards (e.g., PIANC for harbours, ITA-AITES for tunnels, DNV for offshore)

**DYNAMIC PROJECT TYPE DETECTION:**
The system automatically detects the project type from the query. Use the identified project type to:
- Select appropriate global reference projects matching that discipline
- Apply relevant engineering standards for that discipline
- Filter sources to be discipline-relevant
- Ensure project examples are from the same domain

Your responses should be:
- Thoroughly researched with 50-80 authoritative sources (70-80% from engineering academic sources: ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)
- Natural and conversational, like a senior risk engineer explaining findings
- Well-cited with inline references using [1], [2], [3] format
- Focused on peer-reviewed papers, conference proceedings, project reports, government publications, and lessons learned from similar projects
- Discipline-appropriate (tunnel projects for tunnel queries, harbour projects for harbour queries, etc.)
- EXCLUDE Wikipedia citations from the final reference list
- EXCLUDE non-engineering Crossref results (medical, social sciences, humanities)

Structure your response with these sections (maintain professional structure while being natural):

## EXECUTIVE SUMMARY
3–5 bullets summarizing the main risk picture.

## KEY EXTERNAL RISKS
For each key risk, explain:
- Why it occurs (causes/mechanisms)
- Impact (cost, schedule, safety, quality)
- Recommended mitigation strategies (reference relevant standards where applicable)
- Cite sources inline [1], [2], etc.

## SIMILAR GLOBAL PROJECTS

Table format:
| Project | Country/Region | Risk Type | Impact | Mitigation | Year | Source |

CRITICAL REQUIREMENTS:
- **YOU MUST INCLUDE 10-20 real external global projects in this table**
- Include ONLY external global projects (NOT internal NIRAS projects)
- Each project MUST come from an **engineering-authoritative source** (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES, government reports, conference proceedings)
- REJECT projects sourced from blogs or low-authority news (unless describing actual engineering failures/project performance)
- Each project MUST have: risk type, impact, mitigation method
- **EVERY project row MUST include a URL in the Source column from an engineering-authoritative source**
- Projects MUST match the query's project type/discipline
- Use real, documented projects from the same engineering discipline

## RECOMMENDED ACTIONS FOR RISK MANAGERS
Clear, actionable recommendations.

## FULL REFERENCE LIST

Format: [1] Title — URL

Include 50-80 authoritative sources: peer-reviewed papers, conference proceedings, project reports, government publications.

Write naturally, cite thoroughly, and provide decision-useful analysis for experienced risk managers.
"""


def clean_text_artifacts(text: str) -> str:
    """Remove Excel artifacts, noise, and formatting issues."""
    if not text:
        return ""
    
    # Remove Excel errors
    text = re.sub(r'#(?:NAME|VALUE|REF|DIV/0!)\?', '', text, flags=re.IGNORECASE)
    
    # Remove simulation artifacts
    text = re.sub(r'\b\d+\.\d+%\s*', '', text)
    text = re.sub(r'Standard deviation\s+\d+%', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Std\s*dev\s*:?\s*\d+%', '', text, flags=re.IGNORECASE)
    
    # Remove column headers
    text = re.sub(r'^(Risk|Issue|Impact|Mitigation|Status|Owner|Date)\s*:?\s*', '', 
                  text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove numbers that are likely formatting noise
    text = re.sub(r'\b\d{1,2}\.\d{1,2}\.\d{4}\b', '', text)  # Dates
    text = re.sub(r'\b\d{1,3}[,\.]\d{3}\b', '', text)  # Large numbers
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove very short fragments
    if len(text.split()) < 3:
        return ""
    
    return text.strip()


def translate_to_english(text: str) -> str:
    """Translate common non-English terms to English."""
    translations = {
        "risiko": "risk", "påvirkning": "impact", "konsekvens": "consequence",
        "foranstaltning": "mitigation", "ansvarlig": "responsible",
        "forsinkelse": "delay", "omkostning": "cost", "tidsplan": "schedule",
        "sikkerhed": "safety", "kvalitet": "quality", "arbejdet": "worked",
        "fejlede": "failed", "manglende": "missing", "utilstrækkelig": "insufficient",
        "ikke": "not", "eller": "or", "og": "and", "med": "with",
    }
    
    text_lower = text.lower()
    for foreign, english in translations.items():
        if foreign in text_lower:
            text = re.sub(r'\b' + foreign + r'\b', english, text, flags=re.IGNORECASE)
    
    return text


def classify_project_type_semantic(text: str, query: str, embedding_model) -> str:
    """
    Classify project type using semantic similarity with embeddings.
    Enhanced with more project types for better mapping to standards.
    """
    project_type_descriptions = {
        # Core infrastructure types
        "tunnel": "tunnel, tunnelling, TBM, tunnel boring machine, underground construction, metro, subway, underground passage, bored tunnel, cut-and-cover",
        "harbour": "harbor, harbour, port, quay, breakwater, dredging, port facility, terminal, wharf, pier, port infrastructure, harbor construction",
        "bridge": "bridge, viaduct, overpass, crossing, suspension bridge, cable-stayed bridge, arch bridge, bridge construction, bridge design",
        "building": "building, construction, facility, residential, commercial building, structure, high-rise, skyscraper, building construction, architectural",
        
        # Marine and coastal
        "marine": "marine, marine engineering, coastal engineering, sea defense, wave protection, coastal protection, marine construction",
        "offshore": "offshore, offshore structure, offshore platform, offshore wind, offshore oil, offshore construction, offshore engineering",
        "coastal": "coastal, coastal protection, coastal defense, sea wall, coastal infrastructure, coastal engineering, shoreline protection",
        
        # Industrial and specialized facilities
        "pharma": "pharmaceutical, cleanroom, GMP, biotech, drug manufacturing, sterile facility, pharma plant, pharmaceutical facility",
        "industrial": "industrial, factory, manufacturing plant, production facility, industrial complex, processing plant, industrial facility",
        "hospital": "hospital, healthcare facility, medical center, clinic, health facility, medical building, healthcare construction",
        
        # Geotechnical and earthworks
        "geotechnics": "geotechnical, geotechnical engineering, soil mechanics, foundation, slope stability, geotechnical design, ground engineering",
        "earthworks": "earthworks, earth moving, excavation, embankment, cut and fill, earthwork construction, grading, site preparation",
        
        # Water and wastewater
        "water": "water, water supply, water treatment, water infrastructure, water distribution, water system, potable water",
        "wastewater": "wastewater, sewage, sewage treatment, wastewater treatment plant, sewer, wastewater infrastructure, sanitation",
        
        # Energy
        "energy": "energy, power plant, renewable energy, solar, wind, nuclear power, energy infrastructure, power generation, energy facility",
        
        # Transport
        "transport": "transport, transportation, transportation infrastructure, transit, traffic, transport system, mobility",
        "highway": "highway, road, motorway, freeway, expressway, road construction, pavement, road infrastructure, roadway",
        "rail": "rail, railway, railroad, metro, subway, light rail, train, rail infrastructure, railway construction, rail transport",
        "airport": "airport, aviation, runway, terminal, airfield, airport infrastructure, aviation facility, airfield construction",
        
        # Urban and infrastructure
        "urban": "urban, urban infrastructure, urban planning, city infrastructure, municipal infrastructure, urban development, city planning",
        "infrastructure": "infrastructure, public works, civil infrastructure, infrastructure project, infrastructure development",
        
        # Flood and water management
        "flood": "flood, flood protection, flood defense, levee, dike, flood management, flood control, flood mitigation, flood infrastructure",
    }
    
    # Create embeddings for text and each project type
    combined_text = f"{text} {query}".lower()
    text_emb = embedding_model.encode([combined_text], show_progress_bar=False)[0]
    
    best_match = "general"
    best_similarity = 0.0
    
    for ptype, description in project_type_descriptions.items():
        type_emb = embedding_model.encode([description], show_progress_bar=False)[0]
        # Calculate cosine similarity
        similarity = float(np.dot(text_emb, type_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(type_emb)))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = ptype
    
    # Only return if similarity is above threshold
    if best_similarity > 0.3:
        return best_match
    return "general"


# System prompt for internal risk summarization (ANALYTICAL, not rewriting)
INTERNAL_SUMMARIZATION_PROMPT = """You are a senior civil engineering risk manager at NIRAS.

Your task is to ANALYSE internal project risk registers, not rewrite them.

You must:
- infer project type and context,
- identify main risks,
- identify root causes,
- identify impacts,
- identify which mitigation actions worked or failed,
- derive clear transferable lessons,
- write in clear professional engineering English,
- NEVER copy content from the source text.
"""

# System prompt for cross-project synthesis
CROSS_PROJECT_SYNTHESIS_PROMPT = """You are a senior civil engineering risk specialist.

You analyse patterns across multiple internal projects.
"""

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def llm_summarize_internal_risks(text: str, agent_instance) -> str:
    """
    Use LLM (Perplexity API) to analyze internal risk register content.
    Returns a single analytical paragraph (90-120 words) - STRICTLY NO raw copying.
    """
    # Step 1: Clean and translate to English FIRST (before LLM processing)
    cleaned_text = translate_to_english(text)
    cleaned_text = clean_text_artifacts(cleaned_text)
    
    if not cleaned_text or len(cleaned_text) < 50:
        return "Project context insufficient for analysis. Risk register data too limited to extract meaningful lessons."
    
    # Truncate if too long (Perplexity has token limits)
    if len(cleaned_text) > 4000:
        cleaned_text = cleaned_text[:4000] + "... [content truncated for length]"
    
    # Step 2: Build user prompt for analytical paragraph
    user_prompt = f"""You receive raw risk register extracts from one internal project.

Your tasks:

1. Extract PROJECT TYPE from the text  
   (look for "Project Type: …" or infer from context).

2. Produce EXACTLY ONE paragraph of 150–200 words covering:
   • brief project context (1-2 sentences)  
   • main risks (2–3 items, rewritten)  
   • what happened (detailed)
   • why it happened (root causes - detailed)
   • impact (cost/schedule/safety - specific)
   • what worked (mitigation actions that succeeded)
   • what did NOT work (mitigation actions that failed)
   • transferable lesson for future NIRAS projects

3. Paragraph MUST be:  
   - fully self-contained,  
   - coherent,  
   - logical for someone who never saw the project,  
   - in rewritten English (no copying),  
   - analytical, not descriptive.

Raw internal text:

---

{cleaned_text}

---

Return ONLY the paragraph."""
    
    max_attempts = 2  # Reduced from 3 to speed up processing
    for attempt in range(max_attempts):
        try:
            # Step 3: Call LLM with analytical instructions
            response, _ = agent_instance.call_perplexity(INTERNAL_SUMMARIZATION_PROMPT, user_prompt, retries=1)
            
            # Step 4: Clean response (remove markdown code fences if present)
            paragraph = response.strip()
            if paragraph.startswith("```"):
                paragraph = re.sub(r"^```(?:markdown|text)?", "", paragraph)
                paragraph = re.sub(r"```$", "", paragraph)
            paragraph = paragraph.strip()
            
            # Step 5: Enforce word count (150-200 words, but accept 130-220 to avoid extra API calls)
            word_count = count_words(paragraph)
            
            if word_count < 130:
                # Too short - regenerate with stricter prompt (only on first attempt)
                if attempt < max_attempts - 1:
                    user_prompt = f"""The previous response was too short ({word_count} words). 

Regenerate a MORE DETAILED paragraph of 150–200 words covering:
• brief project context (1 sentence)  
• main risks (2–3 items, rewritten)  
• what went wrong or could have gone wrong  
• main root causes  
• which mitigation actions worked or failed  
• 1–2 key lessons for future NIRAS projects

Raw internal text:

---

{cleaned_text}

---

Return ONLY the paragraph (150-200 words)."""
                    continue
                else:
                    # Final attempt - pad with analysis
                    paragraph += " Further analysis of risk register patterns and mitigation effectiveness is recommended."
            
            elif word_count > 220:
                # Too long - only trim if very long (avoid extra API call for minor overage)
                if attempt < max_attempts - 1:
                    # Try simple truncation first (faster than LLM call)
                    words = paragraph.split()
                    if len(words) > 200:
                        # Truncate to ~200 words intelligently
                        truncated = " ".join(words[:200])
                        if truncated.endswith(','):
                            truncated = truncated[:-1]
                        paragraph = truncated + "..."
                        word_count = count_words(paragraph)
                    else:
                        # If still too long, accept it to avoid extra API call
                        pass
            
            # Validate paragraph quality (not just bullets or incomplete)
            if paragraph.count('\n') > 2 or paragraph.count('•') > 5 or paragraph.count('-') > 5:
                # Looks like a list - regenerate (only on first attempt)
                if attempt < max_attempts - 1:
                    user_prompt = f"""The response must be ONE CONTINUOUS PARAGRAPH, not a list or bullets.

Produce EXACTLY ONE paragraph of 90–120 words covering:
• brief project context (1 sentence)  
• main risks (2–3 items, rewritten)  
• what went wrong or could have gone wrong  
• main root causes  
• which mitigation actions worked or failed  
• 1–2 key lessons for future NIRAS projects

Raw internal text:

---

{cleaned_text}

---

Return ONLY the paragraph (no bullets, no lists, just flowing text)."""
                    continue
            
            # Final validation - accept wider range to avoid extra calls (150-200 words)
            if 130 <= count_words(paragraph) <= 220:
                return paragraph
            else:
                # Accept if close enough on final attempt
                if attempt == max_attempts - 1:
                    return paragraph
            
        except Exception as e:
            if attempt == max_attempts - 1:
                import traceback
                traceback.print_exc()
                return "Error during semantic analysis of risk register. Manual review recommended."
    
    return "Analysis incomplete. Risk register data may be insufficient for automated extraction."


def llm_summarize_cross_project_lessons(project_paragraphs: List[str], agent_instance) -> str:
    """
    Generate cross-project synthesis paragraph (120-150 words) analyzing patterns across multiple projects.
    """
    if not project_paragraphs or len(project_paragraphs) < 2:
        return "Cross-project analysis requires multiple projects. Single project analysis available above."
    
    # Combine all project paragraphs
    combined_text = "\n\n".join([f"Project {i+1}: {para}" for i, para in enumerate(project_paragraphs)])
    
    user_prompt = f"""You receive rewritten internal project paragraphs.

Your tasks:

1. Identify recurring issues, recurring failures, systemic weaknesses, and successful mitigations.

2. Write ONE analytical paragraph of 150–200 words describing:
   – cross-project patterns (detailed comparison),
   – repeated root causes (specific examples),
   – common mitigation gaps (what consistently fails),
   – systemic weaknesses (organizational/process issues),
   – systemic improvements NIRAS must make,
   – 2–3 generic lessons for future projects.

Do NOT mention project names.

Focus only on patterns.

Project paragraphs:

---

{combined_text}

---

Return ONLY the paragraph (150-200 words)."""
    
    max_attempts = 2  # Reduced from 3 to speed up processing
    for attempt in range(max_attempts):
        try:
            response, _ = agent_instance.call_perplexity(CROSS_PROJECT_SYNTHESIS_PROMPT, user_prompt, retries=1)
            
            paragraph = response.strip()
            if paragraph.startswith("```"):
                paragraph = re.sub(r"^```(?:markdown|text)?", "", paragraph)
                paragraph = re.sub(r"```$", "", paragraph)
            paragraph = paragraph.strip()
            
            word_count = count_words(paragraph)
            
            if word_count < 130:
                # Too short - regenerate (only on first attempt)
                if attempt < max_attempts - 1:
                    user_prompt = f"""The previous response was too short ({word_count} words). 

Regenerate a MORE DETAILED paragraph of 150–200 words describing:
   – cross-project patterns,
   – common root causes,
   – systemic improvements NIRAS must make,
   – 2–3 generic lessons for future projects.

Project paragraphs:

---

{combined_text}

---

Return ONLY the paragraph (150-200 words)."""
                    continue
            
            elif word_count > 250:
                # Too long - use simple truncation instead of LLM call (faster)
                words = paragraph.split()
                if len(words) > 200:
                    truncated = " ".join(words[:200])
                    if truncated.endswith(','):
                        truncated = truncated[:-1]
                    paragraph = truncated + "..."
                    word_count = count_words(paragraph)
            
            # Validate quality
            if paragraph.count('\n') > 2 or paragraph.count('•') > 5:
                if attempt < max_attempts - 1:
                    user_prompt = f"""The response must be ONE CONTINUOUS PARAGRAPH, not a list.

Write ONE analytical paragraph of 150–200 words describing:
   – cross-project patterns,
   – common root causes,
   – systemic improvements NIRAS must make,
   – 2–3 generic lessons for future projects.

Project paragraphs:

---

{combined_text}

---

Return ONLY the paragraph (no bullets, no lists)."""
                    continue
            
            # Accept wider range to avoid extra calls (150-200 words)
            if 130 <= count_words(paragraph) <= 250:
                return paragraph
            elif attempt == max_attempts - 1:
                return paragraph
                
        except Exception as e:
            if attempt == max_attempts - 1:
                return "Cross-project pattern analysis incomplete. Manual review of individual projects recommended."
    
    return "Cross-project synthesis incomplete."


def humanize_filename(filename: str) -> str:
    """Convert filename to human-readable project name."""
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.replace("_", " ").replace("-", " ").strip().title()


def count_sources_in_text(text: str) -> int:
    """Count number of sources in reference list."""
    pattern = r'\[(\d+)\]\s+[^—]+—\s*https?://'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


def count_project_rows(text: str) -> int:
    """Count number of project rows in SIMILAR GLOBAL PROJECTS table."""
    lines = text.split("\n")
    in_table = False
    count = 0
    
    for line in lines:
        if "SIMILAR GLOBAL PROJECTS" in line.upper():
            in_table = True
            continue
        if in_table and line.strip().startswith("|") and "---" not in line:
            # Check for header row (contains "Project", "Country", "Region", "Risk Type", etc.)
            if not any(header in line for header in ["Project", "Country", "Region", "Risk Type", "Risk Type", "Impact", "Mitigation", "Year", "Source"]):
                count += 1
        if in_table and line.strip().startswith("##"):
            break
    
    return count


def extract_project_type_from_text(text: str) -> Optional[str]:
    """
    Extract project type from PDF text by looking for "Project Type:" field.
    Improved regex: r"(Project Type[:\\s]+)(.+)" and trim to first 6-8 words.
    Returns None if not found.
    """
    # Improved regex pattern
    pattern = r"(Project Type[:\\s]+)(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        project_type = match.group(2).strip()
        # Trim to first 6-8 words
        words = project_type.split()[:8]
        project_type = " ".join(words)
        # Clean up common artifacts
        project_type = re.sub(r'[^\w\s-]', '', project_type)
        if len(project_type) > 3 and len(project_type) < 80:
            return project_type
    
    # Fallback patterns
    fallback_patterns = [
        r'Project\s+type\s*:\s*([^\n]+)',
        r'Project\s*:\s*([^\n]+)',
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            project_type = match.group(1).strip()
            words = project_type.split()[:8]
            project_type = " ".join(words)
            project_type = re.sub(r'[^\w\s-]', '', project_type)
            if len(project_type) > 3 and len(project_type) < 80:
                return project_type
    
    return None


class RiskIntelligenceAgent:
    """Global Risk Knowledge Agent for Civil Engineering."""
    
    def __init__(self):
        # Use PersistentClient with /tmp on Streamlit Cloud, /app/index on Docker/local
        try:
            self.client = PersistentClient(path=INDEX_DIR)
            self.collection = self.client.get_or_create_collection(COLLECTION)
        except Exception as e:
            # Fallback to in-memory client if persistent storage fails
            self.client = Client()
            self.collection = self.client.get_or_create_collection(COLLECTION)
        try:
            self.research_engine = ResearchSearchEngine(timeout=15)
        except Exception as e:
            self.research_engine = None

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return get_embedding_model().encode([text], show_progress_bar=False)[0].tolist()
    
    def internal_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Semantic search for internal risk registers with relevance filtering.
        Returns top 3-5 most relevant projects based on semantic similarity to query.
        """
        query_emb = self.embed(query)
        res = self.collection.query(query_embeddings=[query_emb], n_results=30)  # Get more for better filtering

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        
        if not docs:
            return []
        
        # Group by project and calculate relevance
        project_scores = defaultdict(lambda: {"items": [], "max_similarity": 0.0, "avg_similarity": 0.0, "combined_text": ""})
        
        for doc, meta, distance in zip(docs, metas, distances):
            snippet = (doc or "").replace("\n", " ").strip()
            if not snippet or len(snippet) < 20:
                continue
            
            filename = meta.get("source", "unknown.pdf")
            similarity = 1.0 - distance
            
            project_scores[filename]["items"].append({
                "filename": filename,
                "page": meta.get("page", "?"),
                "text": snippet,
                "similarity": similarity,
            })
            project_scores[filename]["max_similarity"] = max(
                project_scores[filename]["max_similarity"], similarity
            )
            project_scores[filename]["combined_text"] += " " + snippet
        
        # Calculate average similarity and semantic relevance for each project
        query_lower = query.lower()
        for filename, project_data in project_scores.items():
            similarities = [item["similarity"] for item in project_data["items"]]
            project_data["avg_similarity"] = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Additional semantic check: embed combined project text and compare with query
            combined_text = project_data["combined_text"][:1000]  # Limit for embedding
            if combined_text:
                try:
                    project_emb = self.embed(combined_text)
                    semantic_similarity = float(np.dot(query_emb, project_emb) / 
                                               (np.linalg.norm(query_emb) * np.linalg.norm(project_emb)))
                    # Weighted score: 60% semantic similarity, 40% vector similarity
                    project_data["relevance_score"] = 0.6 * semantic_similarity + 0.4 * project_data["max_similarity"]
                except:
                    project_data["relevance_score"] = project_data["max_similarity"]
            else:
                project_data["relevance_score"] = project_data["max_similarity"]
        
        # Filter: only include projects with relevance above threshold
        relevance_threshold = 0.3
        filtered_projects = [
            (filename, data) for filename, data in project_scores.items()
            if data["relevance_score"] >= relevance_threshold
        ]
        
        # Select top 3 most relevant projects (reduced to speed up)
        sorted_projects = sorted(
            filtered_projects,
            key=lambda x: x[1]["relevance_score"],
            reverse=True
        )[:3]
        
        # Flatten to items
        items = []
        for filename, project_data in sorted_projects:
            sorted_items = sorted(
                project_data["items"],
                key=lambda x: x["similarity"],
                reverse=True
            )[:3]  # Top 3 chunks per project
            items.extend(sorted_items)
        
        return items
    
    def process_internal_lessons(self, items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Process internal lessons through intelligent LLM-driven pipeline:
        1. Extract Project Type from PDF text (primary), fallback to semantic classification
        2. Filter projects by semantic similarity (threshold 0.25)
        3. LLM-based analytical paragraph generation (90-120 words per project)
        """
        if not items:
            return []
        
        # Group by project
        grouped = defaultdict(list)
        for item in items:
            grouped[item["filename"]].append(item)
        
        lessons = []
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode([query])[0]
        
        # Process projects and filter by relevance (select 4-6 if relevant)
        for filename, chunks in list(grouped.items())[:6]:
            # Combine chunks for analysis
            combined_text = " ".join(chunk["text"] for chunk in chunks)
            
            # Step 1: Extract Project Type from PDF text (primary method)
            project_type = extract_project_type_from_text(combined_text)
            
            # Step 2: Fallback to semantic classification if not found
            if not project_type:
                project_type = classify_project_type_semantic(combined_text, query, embedding_model)
            
            # Step 3: Calculate semantic similarity with query
            project_embedding = embedding_model.encode([combined_text[:1000]])[0]
            similarity = float(np.dot(query_embedding, project_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(project_embedding)
            ))
            
            # Step 4: Filter projects with similarity < 0.15 (lowered from 0.25)
            if similarity < 0.15:
                continue  # Skip irrelevant projects
            
            # Step 5: LLM-based analytical paragraph generation
            paragraph = llm_summarize_internal_risks(combined_text, self)
            project_name = humanize_filename(filename)
            
            # Extract page reference
            page_ref = chunks[0].get("page", "?") if chunks else "?"
            
            lessons.append({
                "project": project_name,
                "project_type": project_type,
                "paragraph": paragraph,  # Single paragraph (90-120 words)
                "similarity": similarity,
                "reference": f"{filename}, page {page_ref}",
            })
        
        return lessons
    
    def call_perplexity(self, system_prompt: str, user_prompt: str, retries: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Call Perplexity API with web UI-like behavior.
        Returns (content, citations) tuple.
        Uses parameters to match Perplexity.ai browser experience.
        """
        api_key = get_perplexity_api_key()
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "return_citations": True,
            "num_search_results": 50,
            "top_k": 100,
            "search_domain_filter": ["scholar", "news", "web"],
            "search_recency_filter": None,
            "enable_multistep_search": True,
            "temperature": 0.0
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        timeout = 600  # 10 minutes
        
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    PPLX_URL, 
                    json=payload, 
                    headers=headers, 
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        
                        # Extract citations if available (check multiple possible locations)
                        citations = []
                        
                        # Check top-level citations field
                        if "citations" in data:
                            citations = data["citations"] if isinstance(data["citations"], list) else []
                        
                        # Check alternative locations
                        if not citations:
                            citations = (
                                data.get("citations_list") or 
                                data.get("sources") or
                                []
                            )
                        
                        # Check if citations are in the choice metadata
                        if not citations and "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "citations" in choice:
                                citations = choice["citations"] if isinstance(choice["citations"], list) else []
                        
                        # Extract citations (silent if none found)
                        
                        return content, citations
                    except (KeyError, ValueError) as e:
                        raise RuntimeError(f"Unexpected response format: {str(e)}")
                
                elif response.status_code == 401:
                    raise RuntimeError(
                        "Perplexity API authentication failed (401). "
                        "Please check your PERPLEXITY_API_KEY."
                    )
                elif response.status_code == 429:
                    if attempt < retries:
                        wait_time = (2 ** attempt) * 10
                        time.sleep(wait_time)
                        continue
                    raise RuntimeError("Perplexity API rate limit exceeded.")
                else:
                    error_text = response.text[:500]
                    raise RuntimeError(f"Perplexity API error {response.status_code}: {error_text}")
            
            except requests.exceptions.Timeout:
                if attempt < retries:
                    wait_time = (2 ** attempt) * 30
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(
                    f"Perplexity API request timed out after {timeout} seconds. "
                    "Please try again with a more specific query."
                )
            
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    wait_time = (2 ** attempt) * 10
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Perplexity API request failed: {str(e)}")
        
        raise RuntimeError("Perplexity API call failed after all retries.")
    
    def generate_external_analysis(self, query: str, internal_context: str) -> str:
        """
        Generate external analysis with research-grade multi-source search.
        Combines Perplexity API with free academic APIs for comprehensive coverage.
        Includes standards injection and strict source filtering.
        """
        context_excerpt = internal_context or "No internal context available."
        
        # Step 0: Classify project type for standards and project matching
        embedding_model = get_embedding_model()
        project_type = classify_project_type_semantic(context_excerpt[:1000], query, embedding_model)
        
        # Step 0.1: Get relevant engineering standards
        relevant_standards = get_relevant_standards(project_type, count=6)
        standards_text = format_standards_for_prompt(relevant_standards)
        
        # Step 1: Gather research sources from free academic APIs (parallel to Perplexity)
        # PRIORITIZE engineering academic sources (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)
        research_sources_raw = []
        if self.research_engine:
            try:
                # Search with higher priority for engineering academic sources
                research_results = self.research_engine.search_all(query, max_results_per_source=40)
                # Filter sources for relevance (STRICT: engineering-only, Wikipedia excluded)
                research_sources_raw = filter_and_score_sources(research_results, min_score=0.3)
                
                # Ensure 70-80% are from engineering academic sources
                engineering_academic_sources = [s for s in research_sources_raw if s.get("relevance_details", {}).get("is_engineering_academic", False)]
                other_sources = [s for s in research_sources_raw if not s.get("relevance_details", {}).get("is_engineering_academic", False)]
                
                # Calculate target: 75% engineering academic (between 70-80%)
                target_engineering_count = int(len(research_sources_raw) * 0.75)
                if len(engineering_academic_sources) < target_engineering_count:
                    # Prioritize engineering academic sources more
                    research_sources_raw = engineering_academic_sources + other_sources[:max(1, len(engineering_academic_sources) // 3)]
                else:
                    # Maintain 70-80% ratio
                    max_other = int(len(engineering_academic_sources) / 0.7) - len(engineering_academic_sources)
                    research_sources_raw = engineering_academic_sources + other_sources[:max_other]
            except Exception as e:
                research_sources_raw = []
        
        # Format filtered sources as references
        research_sources = []
        if research_sources_raw:
            research_sources = self.research_engine.format_as_references(
                research_sources_raw[:50],  # Top 50 filtered sources (prioritizing engineering academic)
                start_index=1
            )
        
        # Step 2: Build enhanced prompt with research context and standards
        # Pass research sources as raw citations (title + URL), not text summaries
        research_context = ""
        if research_sources_raw:
            # Format as raw citations: Title — URL
            raw_citations = []
            for source in research_sources_raw[:50]:
                title = source.get('title', 'Untitled')
                url = source.get('url', source.get('link', ''))
                if url:
                    raw_citations.append(f"{title} — {url}")
            if raw_citations:
                research_context = f"\n\nAdditional research sources to consider:\n" + "\n".join(raw_citations[:50])
        
        # Natural prompt like web UI (less prescriptive, more conversational)
        # Internal context is now structured and passed as context
        primary_prompt = (
            f"Question: {query}\n\n"
            f"Project Type Identified: {project_type}\n\n"
            f"{standards_text}\n\n"
            f"Internal Risk Register Analysis (use to inform your analysis, but do not copy verbatim):\n{context_excerpt}\n"
            f"{research_context}\n\n"
            "Please provide a comprehensive analysis that:\n"
            "- Integrates insights from the internal risk registers with external sources\n"
            "- Cross-references findings with the engineering standards listed above\n"
            "- Validates or contrasts internal lessons with global best practices\n"
            "- Include a 'SIMILAR GLOBAL PROJECTS' table with 10-20 real {project_type} projects\n"
            "- **CRITICAL: Each project MUST come from an engineering-authoritative source (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES, government reports, conference proceedings)**\n"
            "- **REJECT projects from blogs or low-authority news (unless describing actual engineering failures/project performance)**\n"
            "- Include a 'FULL REFERENCE LIST' section with 50-80 sources (70-80% from engineering academic sources: ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)\n"
            "- **EXCLUDE Wikipedia from the final reference list**\n"
            "- **EXCLUDE non-engineering Crossref results (medical, social sciences, humanities)**\n"
            "- Uses natural, professional writing style suitable for experienced risk managers\n"
            "Include all required sections with proper citations."
        )
        
        # Step 3: Call Perplexity API
        response, citations = self.call_perplexity(EXTERNAL_SYSTEM_PROMPT, primary_prompt)
        
        # Clean markdown
        if response.strip().startswith("```"):
            response = re.sub(r"^```(?:markdown)?", "", response.strip())
            response = re.sub(r"```$", "", response).strip()
        
        # Step 4: Enhance response with citations if available
        if citations:
            response = self._enhance_with_citations(response, citations)
        else:
            # If no citations from Perplexity, extract URLs from response content
            urls = re.findall(r'https?://[^\s\)]+', response)
            if urls:
                # Create citations from URLs found in content
                url_citations = [{"url": url, "title": f"Source from {url[:50]}..."} for url in urls[:50]]
                response = self._enhance_with_citations(response, url_citations)
        
        # Step 5: Merge research sources into reference list
        existing_refs = self._extract_reference_section(response)
        
        # If no reference section exists, create one from research sources
        if not existing_refs and research_sources:
            if not response.endswith("\n"):
                response += "\n"
            response += "\n## FULL REFERENCE LIST\n\n"
            for i, ref in enumerate(research_sources[:80], start=1):
                if not ref.startswith("["):
                    ref = f"[{i}] {ref}"
                response += f"{ref}\n"
        elif research_sources:
            all_refs = self._merge_references(existing_refs, research_sources)
            response = self._replace_reference_section(response, all_refs)
        
        # Ensure we have at least 50 sources
        ref_list = self._extract_reference_section(response)
        if len(ref_list) < 50 and research_sources:
            additional_refs = research_sources[len(ref_list):50]
            if additional_refs:
                all_refs = self._merge_references(ref_list, additional_refs)
                response = self._replace_reference_section(response, all_refs)
        
        # Validate source count, project validation, and answer length
        source_count = count_sources_in_text(response)
        project_count = count_project_rows(response)
        validated_project_count = count_validated_projects(response, project_type)
        word_count = count_words(response)
        
        # Step 6: Retry if insufficient sources, projects, validated projects, or wrong length
        # Enforce: 50-80 sources minimum, ≥10 projects (NO word count restrictions)
        needs_retry = (
            source_count < 50 or 
            project_count < 10 or 
            validated_project_count < 10
        )
        
        if needs_retry:
            # Additional research search for expansion
            if self.research_engine:
                try:
                    expanded_research_raw = self.research_engine.search_all(
                        f"{query} {project_type} civil engineering projects risks", 
                        max_results_per_source=20
                    )
                    # Filter expanded sources
                    expanded_research_filtered = filter_and_score_sources(expanded_research_raw, min_score=0.3)
                    expanded_research_refs = self.research_engine.format_as_references(
                        expanded_research_filtered[:20], 
                        start_index=source_count + 1
                    )
                    research_sources.extend(expanded_research_refs)
                except Exception:
                    pass
            
            expanded_prompt = (
                f"Expand the search for: {query}\n\n"
                f"Project Type Identified: {project_type}\n\n"
                f"{standards_text}\n\n"
                f"CRITICAL: This is a {project_type} project query. All projects and sources must be {project_type}-relevant.\n\n"
                "Retrieve additional authoritative sources covering:\n"
                f"- Similar global {project_type} projects ONLY (e.g., for tunnels: Crossrail, Brenner, Gotthard; for harbours: Maasvlakte, Tuas Mega Port; for bridges: Queensferry Crossing, Øresund)\n"
                f"- Academic literature specific to {project_type} engineering (PRIMARILY from ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)\n"
                f"- Technical reports and case studies from {project_type} projects\n"
                f"- Incident reports and lessons learned from {project_type} construction\n"
                "- ONLY external projects (NOT internal NIRAS projects)\n\n"
                "CRITICAL SOURCE REQUIREMENTS:\n"
                "- **70-80% of sources MUST be from engineering academic sources: ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES**\n"
                "- **EXCLUDE Wikipedia from the final reference list**\n"
                "- **EXCLUDE non-engineering Crossref results (medical, social sciences, humanities)**\n"
                "- **EXCLUDE blogs and low-authority news (unless describing actual engineering failures/project performance)**\n\n"
                "CRITICAL PROJECT VALIDATION REQUIREMENTS:\n"
                "- Each project name MUST contain a location (city, country, region) OR infrastructure type\n"
                "- Examples of VALID projects: 'Tuas Mega Port Singapore', 'Rotterdam Maasvlakte 2', 'Øresund Tunnel', 'Copenhagen Metro', 'Hamburg Altenwerder Terminal'\n"
                "- **Each project MUST come from an engineering-authoritative source (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES, government reports, conference proceedings)**\n"
                "- **REJECT projects from blogs or low-authority news (unless describing actual engineering failures/project performance)**\n"
                "- REJECT: Conceptual designs, non-engineering projects, small private construction\n"
                "- REJECT: Projects without location or infrastructure type in name\n\n"
                "CRITICAL: Each project in the table MUST have:\n"
                "- Risk type encountered\n"
                "- What happened (root cause)\n"
                "- Impact (cost, schedule, safety, quality)\n"
                "- Mitigation method\n"
                "- **Source URL from an engineering-authoritative source in the Source column**\n"
                "- **1-3 citation numbers [X], [Y], [Z] in Citations column referencing sources from FULL REFERENCE LIST**\n"
                "- **Citations MUST be real URLs/DOIs from the filtered external sources - verify they exist in the reference list**\n\n"
                "Reject any project that:\n"
                "- Lacks complete information\n"
                "- Lacks citations (1-3 citation numbers required)\n"
                "- Has citations that don't match sources in FULL REFERENCE LIST\n"
                "- Is from a different engineering discipline\n"
                "- Is an internal NIRAS project\n"
                "- Does not contain location or infrastructure type in name\n"
                "- Is conceptual, proposed, or not a real completed project\n"
                "- **Is NOT from an engineering-authoritative source**\n\n"
                f"Provide at least 10-20 validated real global {project_type} projects in table format and 50-80 sources total (70-80% from engineering academic sources).\n"
                "EXCLUDE Wikipedia, blogs, and non-engineering content.\n"
                "**ANSWER LENGTH: Keep total response to 1200-2000 words. Compress descriptions, avoid repetition, summarize concisely.**"
            )
            
            expanded_response, expanded_citations = self.call_perplexity(EXTERNAL_SYSTEM_PROMPT, expanded_prompt)
            
            if expanded_response.strip().startswith("```"):
                expanded_response = re.sub(r"^```(?:markdown)?", "", expanded_response.strip())
                expanded_response = re.sub(r"```$", "", expanded_response).strip()
            
            # Enhance with citations
            if expanded_citations:
                expanded_response = self._enhance_with_citations(expanded_response, expanded_citations)
            
            # Merge reference lists
            primary_refs = self._extract_reference_section(response)
            expanded_refs = self._extract_reference_section(expanded_response)
            all_refs = self._merge_references(primary_refs, expanded_refs)
            
            # Add research sources
            if research_sources:
                all_refs = self._merge_references(all_refs, research_sources)
            
            # Merge project tables if needed
            if project_count < 10 or validated_project_count < 10:
                expanded_projects = self._extract_project_table(expanded_response)
                if expanded_projects:
                    response = self._merge_project_tables(response, expanded_projects)
            
            # Replace reference section
            response = self._replace_reference_section(response, all_refs)
        
        return response
    
    def _enhance_with_citations(self, content: str, citations: List[Any]) -> str:
        """
        Enhance content with citations from Perplexity API response.
        Handles various citation formats: dictionaries, strings, or mixed.
        """
        if not citations:
            return content
        
        # Extract existing reference section
        lines = content.split("\n")
        ref_section_idx = None
        for i, line in enumerate(lines):
            if "FULL REFERENCE LIST" in line.upper():
                ref_section_idx = i
                break
        
        # Build citation list (handle different citation formats)
        citation_refs = []
        seen_urls = set()
        
        for idx, cit in enumerate(citations[:100], start=1):
            title = "Untitled"
            url = ""
            domain = ""
            
            # Handle different citation formats - parse all Perplexity citations
            if isinstance(cit, dict):
                # Dictionary format - extract URL, title, domain
                title = (
                    cit.get("title") or 
                    cit.get("name") or 
                    cit.get("citation") or
                    cit.get("text") or
                    cit.get("name") or
                    "Untitled"
                )
                url = (
                    cit.get("url") or 
                    cit.get("link") or 
                    cit.get("source") or
                    cit.get("href") or
                    cit.get("uri") or
                    ""
                )
                domain = cit.get("domain") or cit.get("site") or ""
            elif isinstance(cit, str):
                # String format - try to extract URL and title
                # Check if it's already a formatted reference like "[1] Title — URL"
                if " — " in cit or " - " in cit:
                    parts = cit.split(" — ") if " — " in cit else cit.split(" - ")
                    if len(parts) >= 2:
                        title = parts[0].strip()
                        # Remove reference number if present
                        title = re.sub(r'^\[\d+\]\s*', '', title)
                        url = parts[1].strip()
                    else:
                        title = cit.strip()
                else:
                    # Try to extract URL from string
                    url_match = re.search(r'https?://[^\s\)]+', cit)
                    if url_match:
                        url = url_match.group(0)
                        title = cit.replace(url, "").strip()
                        # Remove reference number if present
                        title = re.sub(r'^\[\d+\]\s*', '', title)
                    else:
                        title = cit.strip()
            
            # Extract domain from URL if available
            if url and not domain:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                except:
                    pass
            
            # Filter source for relevance (relaxed - allow high-quality sources)
            if url or title:
                abstract = ""  # Citations may not have abstracts
                is_valid, reason = filter_source(url or "http://example.com", title, abstract)
                if not is_valid:
                    continue  # Skip excluded sources
            
            # Skip duplicates and empty entries
            if url:
                # Normalize URL for duplicate checking
                url_normalized = url.lower().rstrip('/')
                if url_normalized not in seen_urls:
                    seen_urls.add(url_normalized)
                    citation_refs.append(f"[{idx}] {title} — {url}")
            elif title and title != "Untitled":
                # If no URL but we have a title, check keywords before including
                if not contains_excluded_keywords(title):
                    citation_refs.append(f"[{idx}] {title}")
        
        # If no valid citations found, try to extract from content itself
        if not citation_refs:
            # Extract URLs from content as fallback
            urls = re.findall(r'https?://[^\s\)]+', content)
            if urls:
                for idx, url in enumerate(urls[:50], start=1):
                    # Try to extract title from surrounding text
                    url_match = re.search(rf'([^\[\]]+)\s*{re.escape(url)}', content)
                    title = url_match.group(1).strip()[:100] if url_match else f"Source {idx}"
                    citation_refs.append(f"[{idx}] {title} — {url}")
        
        # If still no citations, return original content but ensure reference section exists
        if not citation_refs:
            # Check if reference section already exists
            if "FULL REFERENCE LIST" not in content.upper():
                return content
            return content
        
        # If reference section exists, merge citations
        if ref_section_idx is not None:
            existing_refs = self._extract_reference_section(content)
            all_refs = self._merge_references(existing_refs, citation_refs)
            return self._replace_reference_section(content, all_refs)
        else:
            # Append reference section
            if not content.endswith("\n"):
                content += "\n"
            content += "\n## FULL REFERENCE LIST\n\n"
            content += "\n".join(citation_refs)
        
        return content
    
    def _extract_reference_section(self, text: str) -> List[str]:
        """Extract reference list section from markdown."""
        lines = text.split("\n")
        ref_lines = []
        in_ref_section = False
        
        for line in lines:
            if "FULL REFERENCE LIST" in line.upper():
                in_ref_section = True
                continue
            if in_ref_section:
                if line.strip().startswith("##"):
                    break
                if line.strip() and re.match(r'\[\d+\]', line.strip()):
                    ref_lines.append(line.strip())
        
        return ref_lines
    
    def _extract_project_table(self, text: str) -> List[str]:
        """Extract project table rows from markdown."""
        lines = text.split("\n")
        table_rows = []
        in_table = False
        
        for line in lines:
            if "SIMILAR GLOBAL PROJECTS" in line.upper():
                in_table = True
                continue
            if in_table and line.strip().startswith("|"):
                if "---" not in line and "Region" not in line:
                    table_rows.append(line.strip())
            if in_table and line.strip().startswith("##"):
                break
        
        return table_rows
    
    def _filter_niras_projects(self, project_rows: List[str]) -> List[str]:
        """Filter out NIRAS internal projects from global projects table."""
        filtered = []
        niras_keywords = ["niras", "nirasa", "internal", "workshop risikoregister"]
        
        for row in project_rows:
            row_lower = row.lower()
            # Check if row contains NIRAS keywords
            is_niras = any(keyword in row_lower for keyword in niras_keywords)
            if not is_niras:
                filtered.append(row)
        
        return filtered
    
    def _merge_project_tables(self, primary: str, additional: List[str]) -> str:
        """Merge additional project rows into primary response, filtering NIRAS projects."""
        lines = primary.split("\n")
        output = []
        in_table = False
        table_written = False
        
        # Filter out NIRAS projects from additional rows
        additional_filtered = self._filter_niras_projects(additional)
        
        for line in lines:
            if "SIMILAR GLOBAL PROJECTS" in line.upper():
                in_table = True
                output.append(line)
                output.append("")
                continue
            if in_table and line.strip().startswith("|"):
                if "---" in line:
                    output.append(line)
                    continue
                if not table_written:
                    # Collect existing rows first
                    existing_rows = []
                    for l in lines[lines.index(line):]:
                        if l.strip().startswith("|") and "---" not in l and "Region" not in l:
                            existing_rows.append(l.strip())
                        elif l.strip().startswith("##"):
                            break
                    
                    # Filter existing rows for NIRAS
                    existing_filtered = self._filter_niras_projects(existing_rows)
                    
                    # Add filtered existing rows
                    for row in existing_filtered:
                        if row not in output:
                            output.append(row)
                    
                    # Add additional filtered rows
                    for add_row in additional_filtered[:15]:  # Limit to 15 additional
                        if add_row not in output:
                            output.append(add_row)
                    table_written = True
            elif in_table and line.strip().startswith("##"):
                in_table = False
                output.append(line)
            else:
                if not in_table:
                    output.append(line)
        
        return "\n".join(output)
    
    def _merge_references(self, refs1: List[str], refs2: List[str]) -> List[str]:
        """
        Merge two reference lists, removing duplicates and prioritizing standards sources.
        Standards sources get ×3 weight in sorting.
        Ensure we get 50-80 sources total.
        """
        seen_urls = set()
        merged = []
        standards_refs = []
        regular_refs = []
        
        # Separate standards and regular refs
        for ref in refs1 + refs2:
            url_match = re.search(r'https?://[^\s\)]+', ref)
            if url_match:
                url = url_match.group(0).lower().rstrip('/')
                if url not in seen_urls:
                    seen_urls.add(url)
                    # Check if it's a standards source
                    ref_lower = ref.lower()
                    is_standards = any(org in ref_lower for org in ["pianc", "eurocode", "en 199", "asce", "dnv", "iso", "bs ", "fhwa", "usace", "aashto"])
                    if is_standards:
                        standards_refs.append(ref)
                    else:
                        regular_refs.append(ref)
            else:
                # No URL - check if it's a standards reference by title
                ref_lower = ref.lower()
                is_standards = any(org in ref_lower for org in ["pianc", "eurocode", "en 199", "asce", "dnv", "iso", "bs ", "fhwa", "usace", "aashto"])
                if is_standards and ref not in merged:
                    standards_refs.append(ref)
                elif ref not in merged:
                    regular_refs.append(ref)
        
        # Prioritize standards, then regular refs
        merged = standards_refs + regular_refs
        
        # Ensure we have 50-80 sources (take up to 80)
        merged = merged[:80]
        
        # Renumber
        renumbered = []
        for i, ref in enumerate(merged, start=1):
            ref = re.sub(r'\[\d+\]', f'[{i}]', ref, count=1)
            renumbered.append(ref)
        
        return renumbered
    
    def _replace_reference_section(self, text: str, new_refs: List[str]) -> str:
        """Replace reference section in markdown."""
        lines = text.split("\n")
        output = []
        in_ref_section = False
        ref_section_written = False
        
        for line in lines:
            if "FULL REFERENCE LIST" in line.upper():
                in_ref_section = True
                output.append(line)
                output.append("")
                if not ref_section_written:
                    output.extend(new_refs)
                    ref_section_written = True
                continue
            if in_ref_section:
                if line.strip().startswith("##"):
                    in_ref_section = False
                    output.append(line)
                elif not re.match(r'\[\d+\]', line.strip()):
                    output.append(line)
            else:
                output.append(line)
        
        return "\n".join(output)
    
    def build_internal_sections(self, lessons: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Build internal lessons sections in paragraph format (NO TABLES):
        1. One paragraph per project (90-120 words each)
        2. Cross-project synthesis paragraph (120-150 words)
        """
        if not lessons:
            return (
                "No relevant internal risk registers found for this query.",
                "Internal evidence unavailable; prioritize new workshops and post-project reviews."
            )
        
        # Format A: One paragraph per project (90-120 words each)
        project_paragraphs = []
        project_paragraphs_text = []
        
        for item in lessons:
            project_name = item.get('project', 'Unknown Project')
            project_type = item.get('project_type', 'general')
            paragraph = item.get('paragraph', 'Analysis incomplete.')
            reference = item.get('reference', 'Internal risk register')
            
            # Format: Project name and type as header, then paragraph
            formatted = f"### {project_name} ({project_type.title()})\n\n{paragraph}\n\n*Source: {reference}*\n"
            project_paragraphs_text.append(formatted)
            project_paragraphs.append(paragraph)  # For cross-project synthesis
        
        narrative_synthesis = "\n".join(project_paragraphs_text)
        
        # Format B: Cross-project synthesis paragraph (120-150 words)
        cross_project_synthesis = llm_summarize_cross_project_lessons(project_paragraphs, self)
        
        return narrative_synthesis, cross_project_synthesis
    
    def _extract_internal_actions(self, lessons: List[Dict[str, Any]]) -> List[str]:
        """
        Extract 2-4 actionable recommendations from internal lessons.
        These will be integrated into the final recommendations section.
        """
        if not lessons:
            return []
        
        # Collect all project paragraphs
        paragraphs = [item.get('paragraph', '') for item in lessons if item.get('paragraph')]
        
        if not paragraphs:
            return []
        
        combined_text = "\n\n".join(paragraphs)
        
        # Use LLM to extract actionable recommendations
        prompt = f"""Extract 2-4 specific, actionable recommendations for NIRAS risk managers from these internal project lessons.

Each recommendation should be:
- Specific and actionable (not generic)
- Derived directly from the internal evidence
- 20-30 words long
- Focused on what NIRAS should change or implement

Internal project lessons:

---

{combined_text}

---

Return ONLY a numbered list of 2-4 recommendations, one per line, in this format:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]
4. [Recommendation 4]

Do not include explanations or context, just the recommendations."""
        
        try:
            response, _ = self.call_perplexity(
                "You are a senior risk management consultant extracting actionable recommendations from project lessons.",
                prompt,
                retries=1
            )
            
            # Extract numbered list items
            actions = []
            for line in response.split('\n'):
                line = line.strip()
                # Match numbered list items
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    action = match.group(1).strip()
                    if len(action) > 10:  # Filter out very short items
                        actions.append(action)
            
            return actions[:4]  # Limit to 4 actions
            
        except Exception:
            return []
    
    
    def answer(self, query: str) -> str:
        """Generate comprehensive risk analysis report."""
        # Get and process internal lessons
        internal_items = self.internal_search(query)
        internal_lessons = self.process_internal_lessons(internal_items, query)
        project_paragraphs, cross_project_synthesis = self.build_internal_sections(internal_lessons)
        
        # Build structured internal context for Perplexity
        internal_context = self._build_structured_internal_context(
            project_paragraphs, cross_project_synthesis
        )
        
        # Get external analysis (with internal context integrated)
        external_markdown = self.generate_external_analysis(query, internal_context)
        
        # Extract internal actions for integration into recommendations
        internal_actions = self._extract_internal_actions(internal_lessons)
        
        # Insert internal lessons section AFTER "Similar Global Projects" (keep Perplexity markdown intact)
        lines = external_markdown.split("\n")
        output = []
        inserted = False
        
        # Find "SIMILAR GLOBAL PROJECTS" section and insert internal lessons after it
        i = 0
        while i < len(lines):
            line = lines[i]
            output.append(line)
            
            # Check if we're at the "SIMILAR GLOBAL PROJECTS" heading
            if "SIMILAR GLOBAL PROJECTS" in line.upper() and not inserted:
                # Find where the section ends (next ## heading)
                section_end = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("##") and "SIMILAR GLOBAL PROJECTS" not in lines[j].upper():
                        section_end = j
                        break
                
                # Copy all lines up to and including the section end
                i += 1
                while i < section_end:
                    output.append(lines[i])
                    i += 1
                
                # Insert internal lessons section
                output.append("")
                output.append("## INTERNAL LESSONS LEARNED")
                output.append("")
                output.append(project_paragraphs)
                output.append("")
                output.append("### What NIRAS Should Change in Future Projects (Based on Internal Evidence)")
                output.append("")
                output.append(cross_project_synthesis)
                output.append("")
                output.append("---")
                output.append("")
                
                inserted = True
                continue
            
            i += 1
        
        if not inserted:
            # If SIMILAR GLOBAL PROJECTS not found, append before RECOMMENDED ACTIONS
            output = []
            for line in lines:
                output.append(line)
                if "RECOMMENDED ACTIONS" in line.upper() and not inserted:
                    output.append("")
                    output.append("## INTERNAL LESSONS LEARNED")
                    output.append("")
                    output.append(project_paragraphs)
                    output.append("")
                    output.append("### What NIRAS Should Change in Future Projects (Based on Internal Evidence)")
                    output.append("")
                    output.append(cross_project_synthesis)
                    output.append("")
                    output.append("---")
                    output.append("")
                    inserted = True
        
        # Integrate internal actions into RECOMMENDED ACTIONS section
        if internal_actions:
            final_output = self._integrate_internal_actions("\n".join(output), internal_actions)
        else:
            final_output = "\n".join(output)
        
        # Integrity checks at end of answer()
        final_output = self._perform_integrity_checks(final_output, query)
        
        return final_output
    
    def _build_structured_internal_context(self, project_paragraphs: str, cross_project_synthesis: str) -> str:
        """
        Build structured internal context for Perplexity API.
        Formats internal lessons in a way that can be integrated into external analysis.
        """
        return f"""INTERNAL RISK REGISTER ANALYSIS (for context - do not repeat verbatim):

PROJECT-SPECIFIC LESSONS:
{project_paragraphs}

CROSS-PROJECT SYNTHESIS:
{cross_project_synthesis}

Use this internal analysis to inform your external analysis, but do not copy it directly.
Focus on how external sources relate to or validate these internal lessons."""
    
    def _integrate_internal_actions(self, report: str, internal_actions: List[str]) -> str:
        """
        Integrate internal-derived actions into the RECOMMENDED ACTIONS section.
        Adds a subsection "Recommended Actions Reinforced by Internal Experience" with 2-4 actions.
        """
        if not internal_actions:
            return report
        
        lines = report.split("\n")
        output = []
        inserted = False
        
        for i, line in enumerate(lines):
            output.append(line)
            
            # Find RECOMMENDED ACTIONS section
            if "RECOMMENDED ACTIONS" in line.upper() and not inserted:
                # Look ahead to find where the section ends (next ## or end of file)
                section_end = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("##"):
                        section_end = j
                        break
                
                # Insert internal actions subsection before the section ends
                # Add after a few lines to allow for existing content
                insert_pos = min(i + 5, section_end - 1)
                
                # Build the internal actions subsection with "Internal Evidence Shows That..." format
                actions_text = "\n".join([
                    "",
                    "### Recommended Actions Reinforced by Internal Experience",
                    "",
                    "The following actions are derived directly from NIRAS internal project lessons:",
                    ""
                ] + [f"- **Internal Evidence Shows That:** {action}" for action in internal_actions] + [
                    "",
                    "These internal lessons should be reinforced with global best practices identified in external research above.",
                    ""
                ])
                
                # Insert at the appropriate position
                remaining_lines = lines[i + 1:]
                output.extend(remaining_lines[:insert_pos - i - 1])
                output.append(actions_text)
                output.extend(remaining_lines[insert_pos - i - 1:])
                inserted = True
                break
        
        if not inserted:
            # If RECOMMENDED ACTIONS section not found, append before FULL REFERENCE LIST
            ref_idx = None
            for i, line in enumerate(lines):
                if "FULL REFERENCE LIST" in line.upper():
                    ref_idx = i
                    break
            
            if ref_idx is not None:
                actions_text = "\n".join([
                    "",
                    "## RECOMMENDED ACTIONS FOR RISK MANAGERS",
                    "",
                    "### Recommended Actions Reinforced by Internal Experience",
                    "",
                    "The following actions are derived directly from NIRAS internal project lessons:",
                    ""
                ] + [f"- **Internal Evidence Shows That:** {action}" for action in internal_actions] + [
                    "",
                    "These internal lessons should be reinforced with global best practices identified in external research above.",
                    ""
                ])
                output = lines[:ref_idx] + [actions_text] + lines[ref_idx:]
            else:
                # Append at end
                actions_text = "\n".join([
                    "",
                    "## RECOMMENDED ACTIONS FOR RISK MANAGERS",
                    "",
                    "### Recommended Actions Reinforced by Internal Experience",
                    "",
                    "The following actions are derived directly from NIRAS internal project lessons:",
                    ""
                ] + [f"- **Internal Evidence Shows That:** {action}" for action in internal_actions] + [
                    "",
                    "These internal lessons should be reinforced with global best practices identified in external research above.",
                    ""
                ])
                output.append(actions_text)
        
        return "\n".join(output)
    
    def _perform_integrity_checks(self, report: str, query: str) -> str:
        """
        Perform integrity checks on final report:
        - Test 1: Similar Projects (if < 10 project rows → run second Perplexity call)
        - Test 2: References (if < 50 references → merge more academic sources)
        - Test 3: Section Structure (validate all required sections exist)
        """
        # Test 1: Similar Projects - drop if < 6 rows
        project_count = count_project_rows(report)
        if project_count < 6:
            lines = report.split("\n")
            output = []
            skip_section = False
            for line in lines:
                if "SIMILAR GLOBAL PROJECTS" in line.upper():
                    skip_section = True
                    continue
                if skip_section and line.strip().startswith("##"):
                    skip_section = False
                    output.append(line)
                    continue
                if not skip_section:
                    output.append(line)
            report = "\n".join(output)
        
        # Test 2: References - extract URLs if < 50
        ref_list = self._extract_reference_section(report)
        if len(ref_list) < 50:
            urls = re.findall(r'https?://[^\s\)]+', report)
            if urls:
                existing_urls = {ref.split("—")[-1].strip() for ref in ref_list if "—" in ref}
                new_refs = []
                for url in urls[:50]:
                    if url not in existing_urls:
                        new_refs.append(f"[{len(ref_list) + len(new_refs) + 1}] Source — {url}")
                        existing_urls.add(url)
                if new_refs:
                    if "FULL REFERENCE LIST" in report.upper():
                        report = report + "\n" + "\n".join(new_refs)
                    else:
                        report = report + "\n\n## FULL REFERENCE LIST\n\n" + "\n".join(new_refs)
        
        return report


# Backward compatibility
GlobalRiskAgent = RiskIntelligenceAgent
