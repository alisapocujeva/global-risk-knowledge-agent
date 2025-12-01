"""
Engineering Standards Database

Comprehensive list of relevant engineering standards and guidelines
for civil engineering risk analysis. Organized by domain.
"""

from typing import List, Dict, Any

ENGINEERING_STANDARDS = {
    "general": [
        {
            "code": "ISO 31000",
            "name": "Risk Management — Guidelines",
            "organization": "ISO",
            "domain": "risk_management",
            "description": "International standard for risk management principles and framework"
        },
        {
            "code": "ISO 55000",
            "name": "Asset Management — Overview, Principles and Terminology",
            "organization": "ISO",
            "domain": "asset_management",
            "description": "International standard for asset management systems"
        },
        {
            "code": "EN 1990",
            "name": "Eurocode 0: Basis of Structural Design",
            "organization": "CEN",
            "domain": "structural_design",
            "description": "European standard for structural design basis and reliability"
        }
    ],
    "tunnel": [
        {
            "code": "ITA-AITES Guidelines",
            "name": "Guidelines for Tunnelling Risk Management",
            "organization": "ITA-AITES",
            "domain": "tunnel",
            "description": "International Tunnelling Association guidelines for risk management"
        },
        {
            "code": "BS 6164",
            "name": "Code of Practice for Health and Safety in Tunnelling in the Construction Industry",
            "organization": "BSI",
            "domain": "tunnel",
            "description": "British standard for tunnel construction safety"
        },
        {
            "code": "FHWA-HRT-10-034",
            "name": "Technical Manual for Design and Construction of Road Tunnels",
            "organization": "FHWA",
            "domain": "tunnel",
            "description": "US Federal Highway Administration tunnel design manual"
        }
    ],
    "harbour": [
        {
            "code": "PIANC WG 150",
            "name": "Guidelines for the Design and Construction of Waterfront Structures",
            "organization": "PIANC",
            "domain": "harbour",
            "description": "World Association for Waterborne Transport Infrastructure guidelines"
        },
        {
            "code": "PIANC WG 121",
            "name": "Guidelines for the Design of Fender Systems",
            "organization": "PIANC",
            "domain": "harbour",
            "description": "PIANC guidelines for fender system design"
        },
        {
            "code": "CEDA",
            "name": "Environmental Aspects of Dredging",
            "organization": "CEDA",
            "domain": "harbour",
            "description": "Central Dredging Association environmental guidelines"
        },
        {
            "code": "DNV-RP-C205",
            "name": "Environmental Conditions and Environmental Loads",
            "organization": "DNV",
            "domain": "harbour",
            "description": "DNV recommended practice for marine environmental loads"
        }
    ],
    "marine": [
        {
            "code": "DNV-OS-C101",
            "name": "Design of Offshore Steel Structures",
            "organization": "DNV",
            "domain": "marine",
            "description": "DNV offshore structure design standard"
        },
        {
            "code": "DNV-RP-C201",
            "name": "Probabilistic Methods for Planning of Inspection for Fatigue Cracks in Offshore Structures",
            "organization": "DNV",
            "domain": "marine",
            "description": "DNV recommended practice for fatigue inspection planning"
        },
        {
            "code": "IMO SOLAS",
            "name": "International Convention for the Safety of Life at Sea",
            "organization": "IMO",
            "domain": "marine",
            "description": "International maritime safety standards"
        }
    ],
    "bridge": [
        {
            "code": "EN 1991-2",
            "name": "Eurocode 1: Actions on Structures — Traffic Loads on Bridges",
            "organization": "CEN",
            "domain": "bridge",
            "description": "European standard for bridge traffic loads"
        },
        {
            "code": "AASHTO LRFD",
            "name": "AASHTO LRFD Bridge Design Specifications",
            "organization": "AASHTO",
            "domain": "bridge",
            "description": "American Association of State Highway and Transportation Officials bridge design"
        },
        {
            "code": "BS 5400",
            "name": "Steel, Concrete and Composite Bridges",
            "organization": "BSI",
            "domain": "bridge",
            "description": "British standard for bridge design"
        }
    ],
    "building": [
        {
            "code": "EN 1991-1-1",
            "name": "Eurocode 1: Actions on Structures — General Actions",
            "organization": "CEN",
            "domain": "building",
            "description": "European standard for building loads"
        },
        {
            "code": "ASCE 7",
            "name": "Minimum Design Loads and Associated Criteria for Buildings and Other Structures",
            "organization": "ASCE",
            "domain": "building",
            "description": "American Society of Civil Engineers building design loads"
        },
        {
            "code": "BS 6399",
            "name": "Loading for Buildings",
            "organization": "BSI",
            "domain": "building",
            "description": "British standard for building loads"
        }
    ],
    "geotechnical": [
        {
            "code": "EN 1997-1",
            "name": "Eurocode 7: Geotechnical Design — General Rules",
            "organization": "CEN",
            "domain": "geotechnical",
            "description": "European standard for geotechnical design"
        },
        {
            "code": "FHWA GEC-5",
            "name": "Geotechnical Engineering Circular No. 5 — Evaluation of Soil and Rock Properties",
            "organization": "FHWA",
            "domain": "geotechnical",
            "description": "US Federal Highway Administration geotechnical guidelines"
        },
        {
            "code": "BS 5930",
            "name": "Code of Practice for Ground Investigations",
            "organization": "BSI",
            "domain": "geotechnical",
            "description": "British standard for ground investigations"
        }
    ],
    "industrial": [
        {
            "code": "ASME B31.3",
            "name": "Process Piping",
            "organization": "ASME",
            "domain": "industrial",
            "description": "American Society of Mechanical Engineers process piping standard"
        },
        {
            "code": "API 650",
            "name": "Welded Tanks for Oil Storage",
            "organization": "API",
            "domain": "industrial",
            "description": "American Petroleum Institute tank design standard"
        },
        {
            "code": "NFPA 70",
            "name": "National Electrical Code",
            "organization": "NFPA",
            "domain": "industrial",
            "description": "National Fire Protection Association electrical code"
        }
    ],
    "transport": [
        {
            "code": "EN 1991-2",
            "name": "Eurocode 1: Actions on Structures — Traffic Loads",
            "organization": "CEN",
            "domain": "transport",
            "description": "European standard for traffic loads"
        },
        {
            "code": "AASHTO Green Book",
            "name": "A Policy on Geometric Design of Highways and Streets",
            "organization": "AASHTO",
            "domain": "transport",
            "description": "AASHTO highway design policy"
        },
        {
            "code": "CIRIA C750",
            "name": "The Rock Manual — The Use of Rock in Hydraulic Engineering",
            "organization": "CIRIA",
            "domain": "transport",
            "description": "Construction Industry Research and Information Association rock manual"
        }
    ],
    "flood_coastal": [
        {
            "code": "FEMA P-55",
            "name": "Coastal Construction Manual",
            "organization": "FEMA",
            "domain": "flood_coastal",
            "description": "Federal Emergency Management Agency coastal construction guidelines"
        },
        {
            "code": "USACE EM 1110-2-1614",
            "name": "Risk-Based Analysis for Flood Damage Reduction Studies",
            "organization": "USACE",
            "domain": "flood_coastal",
            "description": "US Army Corps of Engineers flood risk analysis manual"
        },
        {
            "code": "CIRIA C681",
            "name": "The International Levee Handbook",
            "organization": "CIRIA",
            "domain": "flood_coastal",
            "description": "CIRIA international levee design and management guidelines"
        }
    ],
    "safety": [
        {
            "code": "CIRIA C733",
            "name": "CDM Regulations 2015 — Industry Guidance",
            "organization": "CIRIA",
            "domain": "safety",
            "description": "Construction Design and Management regulations guidance"
        },
        {
            "code": "OSHA 29 CFR 1926",
            "name": "Safety and Health Regulations for Construction",
            "organization": "OSHA",
            "domain": "safety",
            "description": "US Occupational Safety and Health Administration construction safety standards"
        },
        {
            "code": "BS OHSAS 18001",
            "name": "Occupational Health and Safety Management Systems",
            "organization": "BSI",
            "domain": "safety",
            "description": "British occupational health and safety management standard"
        }
    ],
    "offshore": [
        {
            "code": "DNV-OS-C101",
            "name": "Design of Offshore Steel Structures",
            "organization": "DNV",
            "domain": "offshore",
            "description": "DNV offshore structure design standard"
        },
        {
            "code": "API RP 2A",
            "name": "Recommended Practice for Planning, Designing and Constructing Fixed Offshore Platforms",
            "organization": "API",
            "domain": "offshore",
            "description": "American Petroleum Institute offshore platform design standard"
        },
        {
            "code": "ISO 19901",
            "name": "Petroleum and Natural Gas Industries — Specific Requirements for Offshore Structures",
            "organization": "ISO",
            "domain": "offshore",
            "description": "ISO standard for offshore structures"
        }
    ],
    "earthworks": [
        {
            "code": "BS 6031",
            "name": "Code of Practice for Earthworks",
            "organization": "BSI",
            "domain": "earthworks",
            "description": "British standard for earthworks design and construction"
        },
        {
            "code": "FHWA NHI-00-045",
            "name": "Earth Retaining Structures",
            "organization": "FHWA",
            "domain": "earthworks",
            "description": "US Federal Highway Administration earth retaining structures manual"
        },
        {
            "code": "CIRIA C580",
            "name": "Embedded Retaining Walls — Guidance for Economic Design",
            "organization": "CIRIA",
            "domain": "earthworks",
            "description": "CIRIA guidance for embedded retaining wall design"
        }
    ],
    "wastewater": [
        {
            "code": "EN 12255",
            "name": "Wastewater Treatment Plants",
            "organization": "CEN",
            "domain": "wastewater",
            "description": "European standard for wastewater treatment plant design"
        },
        {
            "code": "ASCE MOP 8",
            "name": "Design of Municipal Wastewater Treatment Plants",
            "organization": "ASCE",
            "domain": "wastewater",
            "description": "American Society of Civil Engineers wastewater treatment design manual"
        },
        {
            "code": "WEF MOP 8",
            "name": "Design of Municipal Wastewater Treatment Plants",
            "organization": "WEF",
            "domain": "wastewater",
            "description": "Water Environment Federation wastewater treatment design manual"
        }
    ],
    "urban": [
        {
            "code": "ISO 37120",
            "name": "Sustainable Development of Communities — Indicators for City Services and Quality of Life",
            "organization": "ISO",
            "domain": "urban",
            "description": "ISO standard for sustainable urban development"
        },
        {
            "code": "UN-Habitat",
            "name": "Urban Planning Guidelines for Sustainable Development",
            "organization": "UN-Habitat",
            "domain": "urban",
            "description": "United Nations guidelines for sustainable urban planning"
        },
        {
            "code": "CIRIA C723",
            "name": "The SUDS Manual",
            "organization": "CIRIA",
            "domain": "urban",
            "description": "CIRIA manual for sustainable urban drainage systems"
        }
    ]
}

# Project type to standards mapping - covers ALL civil engineering disciplines
PROJECT_TYPE_STANDARDS_MAP = {
    # Core infrastructure
    "tunnel": ["tunnel", "geotechnical", "safety", "general"],
    "harbour": ["harbour", "marine", "geotechnical", "safety", "general"],
    "bridge": ["bridge", "geotechnical", "safety", "general"],
    "building": ["building", "safety", "general"],
    
    # Marine and coastal
    "marine": ["marine", "harbour", "geotechnical", "safety", "general"],
    "offshore": ["offshore", "marine", "geotechnical", "safety", "general"],
    "coastal": ["flood_coastal", "marine", "harbour", "geotechnical", "safety", "general"],
    
    # Industrial and specialized
    "industrial": ["industrial", "safety", "general"],
    "pharma": ["industrial", "safety", "general"],
    "hospital": ["building", "safety", "general"],
    
    # Geotechnical and earthworks
    "geotechnics": ["geotechnical", "safety", "general"],
    "earthworks": ["earthworks", "geotechnical", "safety", "general"],
    
    # Water and wastewater
    "water": ["water", "safety", "general"],
    "wastewater": ["wastewater", "water", "safety", "general"],
    
    # Energy
    "energy": ["industrial", "safety", "general"],
    
    # Transport
    "transport": ["transport", "geotechnical", "safety", "general"],
    "highway": ["transport", "geotechnical", "safety", "general"],
    "rail": ["transport", "geotechnical", "safety", "general"],
    "airport": ["transport", "building", "safety", "general"],
    
    # Urban and infrastructure
    "urban": ["urban", "infrastructure", "safety", "general"],
    "infrastructure": ["general", "safety", "geotechnical"],
    
    # Flood and water management
    "flood": ["flood_coastal", "geotechnical", "safety", "general"],
}


def get_relevant_standards(project_type: str, count: int = 6) -> List[Dict[str, Any]]:
    """
    Get relevant engineering standards for a given project type.
    
    Args:
        project_type: Project type (tunnel, harbour, bridge, etc.)
        count: Number of standards to return (default 6)
    
    Returns:
        List of standard dictionaries
    """
    project_type_lower = project_type.lower()
    
    # Get relevant domains for this project type
    relevant_domains = PROJECT_TYPE_STANDARDS_MAP.get(project_type_lower, ["general", "safety"])
    
    # Collect standards from relevant domains
    standards = []
    for domain in relevant_domains:
        if domain in ENGINEERING_STANDARDS:
            standards.extend(ENGINEERING_STANDARDS[domain])
    
    # Also check if project type directly matches a domain
    if project_type_lower in ENGINEERING_STANDARDS:
        standards.extend(ENGINEERING_STANDARDS[project_type_lower])
    
    # Remove duplicates (by code)
    seen_codes = set()
    unique_standards = []
    for std in standards:
        if std["code"] not in seen_codes:
            seen_codes.add(std["code"])
            unique_standards.append(std)
    
    # Return top N standards
    return unique_standards[:count]


def format_standards_for_prompt(standards: List[Dict[str, Any]]) -> str:
    """
    Format standards list for inclusion in Perplexity prompt.
    
    Args:
        standards: List of standard dictionaries
    
    Returns:
        Formatted string for prompt
    """
    if not standards:
        return ""
    
    lines = ["**Relevant Engineering Standards and Guidelines:**"]
    for std in standards:
        lines.append(
            f"- {std['code']} ({std['organization']}): {std['name']} — {std['description']}"
        )
    
    return "\n".join(lines)

