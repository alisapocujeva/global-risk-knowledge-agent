"""
Research-Grade Multi-Source Search System

Integrates multiple free academic and research APIs:
- ArXiv (preprints and academic papers)
- Semantic Scholar (academic papers with citations)
- OpenAlex (academic metadata)
- Crossref (DOI resolution and metadata)
- Wikipedia (general knowledge)

All APIs are free and require no API keys.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from urllib.parse import quote, urlencode
from datetime import datetime

import requests


class ResearchSearchEngine:
    """Multi-source research search engine for academic and technical sources."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GlobalRiskAgent/1.0 (Research Tool)'
        })
    
    def search_arxiv(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search ArXiv for academic preprints.
        Free, no API key required.
        FILTERED: Only returns engineering-related results (cs.CE, physics.app-ph, etc.).
        """
        try:
            # ArXiv API: https://arxiv.org/help/api/user-manual
            # Filter to engineering categories: cs.CE (Computational Engineering), physics.app-ph (Applied Physics)
            # Also search general query but filter results
            params = {
                'search_query': f'(cat:cs.CE OR cat:physics.app-ph OR all:{quote(query)})',
                'start': 0,
                'max_results': max_results * 2,  # Fetch more to filter
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            url = 'https://export.arxiv.org/api/query'
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                title_text = title.text.strip().replace('\n', ' ') if title is not None else "Untitled"
                
                summary = entry.find('atom:summary', ns)
                summary_text = summary.text.strip() if summary is not None else ""
                
                # Get categories
                categories = [cat.get('term', '') for cat in entry.findall('atom:category', ns)]
                
                # Filter: Only engineering-related categories or content
                combined_text = f"{title_text} {summary_text}".lower()
                engineering_keywords = [
                    'engineering', 'construction', 'structural', 'geotechnical', 'civil',
                    'marine', 'coastal', 'tunnel', 'harbor', 'harbour', 'bridge',
                    'foundation', 'infrastructure', 'offshore', 'waterfront'
                ]
                
                # REJECT non-engineering categories (e.g., cs.AI, math, q-bio, etc.)
                excluded_categories = ['q-bio', 'q-fin', 'stat', 'math', 'astro-ph', 'cond-mat']
                if any(cat.startswith(exc) for cat in categories for exc in excluded_categories):
                    if not any(keyword in combined_text for keyword in engineering_keywords):
                        continue
                
                # ACCEPT if engineering-related
                if not any(keyword in combined_text for keyword in engineering_keywords):
                    continue
                
                # Get authors
                authors = [author.find('atom:name', ns).text 
                          for author in entry.findall('atom:author', ns)
                          if author.find('atom:name', ns) is not None]
                
                # Get links
                links = [link.get('href') for link in entry.findall('atom:link', ns)]
                pdf_url = next((l for l in links if 'pdf' in l), links[0] if links else None)
                
                # Get published date
                published = entry.find('atom:published', ns)
                published_text = published.text if published is not None else ""
                
                results.append({
                    'title': title_text,
                    'abstract': summary_text[:500],  # Limit abstract length
                    'authors': ', '.join(authors[:3]),  # First 3 authors
                    'url': pdf_url or links[0] if links else f"https://arxiv.org/abs/{entry.find('atom:id', ns).text.split('/')[-1] if entry.find('atom:id', ns) is not None else ''}",
                    'source': 'ArXiv',
                    'type': 'preprint',
                    'date': published_text[:10] if published_text else None,
                })
                
                if len(results) >= max_results:
                    break
            
            return results
        
        except Exception as e:
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for academic papers.
        Free, no API key required for basic usage.
        """
        try:
            # Semantic Scholar API: https://api.semanticscholar.org/api-docs/
            url = 'https://api.semanticscholar.org/graph/v1/paper/search'
            params = {
                'query': query,
                'limit': min(max_results, 100),  # API limit
                'fields': 'title,abstract,authors,year,url,venue,citationCount'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for paper in data.get('data', [])[:max_results]:
                authors = [author.get('name', '') for author in paper.get('authors', [])[:3]]
                
                results.append({
                    'title': paper.get('title', 'Untitled'),
                    'abstract': paper.get('abstract', '')[:500],
                    'authors': ', '.join(authors),
                    'url': paper.get('url', ''),
                    'source': 'Semantic Scholar',
                    'type': 'academic_paper',
                    'year': paper.get('year'),
                    'venue': paper.get('venue', ''),
                    'citations': paper.get('citationCount', 0),
                })
            
            return results
        
        except Exception as e:
            return []
    
    def search_openalex(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search OpenAlex for academic works.
        Free, no API key required.
        """
        try:
            # OpenAlex API: https://docs.openalex.org/
            url = 'https://api.openalex.org/works'
            params = {
                'search': query,
                'per_page': min(max_results, 200),
                'sort': 'relevance_score:desc'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for work in data.get('results', [])[:max_results]:
                authors = [author.get('author', {}).get('display_name', '') 
                          for author in work.get('authorships', [])[:3]]
                
                # Get primary location URL
                primary_location = work.get('primary_location', {})
                url = primary_location.get('landing_page_url') or work.get('doi', '')
                if url and not url.startswith('http'):
                    url = f"https://doi.org/{url}"
                
                results.append({
                    'title': work.get('title', 'Untitled'),
                    'abstract': work.get('abstract', '')[:500] if work.get('abstract') else '',
                    'authors': ', '.join(authors),
                    'url': url or work.get('doi', ''),
                    'source': 'OpenAlex',
                    'type': 'academic_work',
                    'year': work.get('publication_date', '')[:4] if work.get('publication_date') else None,
                    'venue': work.get('primary_location', {}).get('source', {}).get('display_name', ''),
                    'citations': work.get('cited_by_count', 0),
                })
            
            return results
        
        except Exception as e:
            return []
    
    def search_crossref(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search Crossref for DOI and metadata.
        Free, no API key required.
        FILTERED: Only returns engineering-related results.
        """
        # Engineering-specific subject filters for Crossref
        engineering_subjects = [
            'civil engineering', 'structural engineering', 'geotechnical engineering',
            'marine engineering', 'coastal engineering', 'environmental engineering',
            'construction', 'infrastructure', 'tunnel', 'harbor', 'harbour', 'bridge',
            'foundation', 'offshore', 'waterfront', 'breakwater', 'pier', 'quay'
        ]
        
        # Engineering journal/publisher indicators
        engineering_venues = [
            'asce', 'ice', 'ice virtual library', 'pianc', 'ita-aites', 'tunnelling',
            'springer', 'sciencedirect', 'elsevier', 'taylor', 'francis', 'wiley',
            'ieee', 'civil engineering', 'structural', 'geotechnical', 'marine',
            'coastal', 'construction', 'infrastructure', 'tunnel', 'harbor', 'bridge'
        ]
        
        try:
            # Crossref API: https://api.crossref.org/
            url = 'https://api.crossref.org/works'
            params = {
                'query': query,
                'rows': min(max_results * 3, 100),  # Fetch more to filter
                'sort': 'relevance'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('message', {}).get('items', [])[:max_results * 3]:
                title = item.get('title', ['Untitled'])[0] if item.get('title') else 'Untitled'
                venue = item.get('container-title', [''])[0] if item.get('container-title') else ''
                subject = ' '.join(item.get('subject', [])).lower()
                
                # Combine text for filtering
                combined_text = f"{title} {venue} {subject}".lower()
                
                # REJECT non-engineering content
                excluded_keywords = [
                    'medicine', 'medical', 'biology', 'biolog', 'pharmacology',
                    'psychology', 'sociology', 'anthropology', 'economics', 'political',
                    'literature', 'history', 'art', 'music', 'philosophy', 'theology',
                    'agriculture', 'veterinary', 'dentistry', 'nursing'
                ]
                if any(excluded in combined_text for excluded in excluded_keywords):
                    continue
                
                # ACCEPT only if engineering-related
                is_engineering = (
                    any(eng_subj in combined_text for eng_subj in engineering_subjects) or
                    any(eng_venue in combined_text for eng_venue in engineering_venues) or
                    'engineering' in combined_text
                )
                
                if not is_engineering:
                    continue
                
                authors = [f"{author.get('given', '')} {author.get('family', '')}".strip()
                          for author in item.get('author', [])[:3]]
                
                doi = item.get('DOI', '')
                url = f"https://doi.org/{doi}" if doi else item.get('URL', '')
                
                results.append({
                    'title': title,
                    'abstract': '',  # Crossref doesn't always have abstracts
                    'authors': ', '.join(authors),
                    'url': url,
                    'source': 'Crossref',
                    'type': 'published_work',
                    'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0] if item.get('published-print') else item.get('created', {}).get('date-parts', [[None]])[0][0],
                    'venue': venue,
                    'doi': doi,
                })
                
                if len(results) >= max_results:
                    break
            
            return results
        
        except Exception as e:
            return []
    
    def search_wikipedia(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for general knowledge.
        Free, no API key required.
        """
        try:
            # Wikipedia API: https://www.mediawiki.org/wiki/API:Main_page
            search_url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'format': 'json'
            }
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('query', {}).get('search', [])[:max_results]:
                title = item.get('title', '')
                snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                page_id = item.get('pageid', '')
                
                results.append({
                    'title': title,
                    'abstract': snippet[:500],
                    'authors': 'Wikipedia Contributors',
                    'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    'source': 'Wikipedia',
                    'type': 'encyclopedia',
                })
            
            return results
        
        except Exception as e:
            return []
    
    def search_all(self, query: str, max_results_per_source: int = 15) -> List[Dict[str, Any]]:
        """
        Search all sources in parallel and aggregate results.
        Returns deduplicated, ranked results.
        PRIORITIZES engineering academic sources (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES).
        NOTE: Wikipedia is excluded per source filtering requirements.
        """
        # Prioritize engineering academic sources with higher result counts
        sources = [
            ('arxiv', self.search_arxiv, max_results_per_source * 2),  # 2x for ArXiv engineering
            ('semantic_scholar', self.search_semantic_scholar, max_results_per_source * 2),  # 2x for academic
            ('openalex', self.search_openalex, max_results_per_source * 2),  # 2x for academic
            ('crossref', self.search_crossref, max_results_per_source),  # 1x (already filtered)
            # Wikipedia excluded per source filtering requirements
        ]
        
        all_results = []
        
        # Search all sources in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {
                executor.submit(search_func, query, max_count): name
                for name, search_func, max_count in sources
            }
            
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception:
                    pass
        
        # Deduplicate by URL or title similarity
        deduplicated = self._deduplicate_results(all_results)
        
        # Rank by relevance (prioritize engineering academic sources, citations, recency)
        ranked = self._rank_results(deduplicated, query)
        
        return ranked
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        deduplicated = []
        
        for result in results:
            url = result.get('url', '').lower().strip()
            title = result.get('title', '').lower().strip()
            
            # Normalize URL
            if url:
                url = re.sub(r'https?://', '', url)
                url = re.sub(r'www\.', '', url)
                url = url.rstrip('/')
            
            # Check for duplicates
            is_duplicate = False
            
            # Check URL
            if url and url in seen_urls:
                is_duplicate = True
            
            # Check title similarity (fuzzy match)
            if title:
                title_normalized = re.sub(r'[^\w\s]', '', title)
                if title_normalized in seen_titles:
                    is_duplicate = True
                seen_titles.add(title_normalized)
            
            if not is_duplicate:
                if url:
                    seen_urls.add(url)
                deduplicated.append(result)
        
        return deduplicated
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rank results by relevance.
        Prioritizes: engineering academic sources (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES) > citations > recency > query match
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Engineering academic source indicators
        engineering_academic_domains = [
            'asce.org', 'ice.org.uk', 'ice virtual library', 'sciencedirect.com',
            'springer.com', 'springerlink.com', 'arxiv.org', 'usace.army.mil',
            'pianc.org', 'ita-aites.org', 'tunnelling', 'dnv.com'
        ]
        
        def score_result(result: Dict[str, Any]) -> float:
            score = 0.0
            
            # Source type priority (engineering academic sources get highest priority)
            source = result.get('source', '').lower()
            url = result.get('url', '').lower()
            venue = result.get('venue', '').lower()
            combined_source = f"{source} {url} {venue}".lower()
            
            # Highest priority: Engineering academic sources (ASCE, ICE, ScienceDirect, Springer, ArXiv engineering, USACE, PIANC, ITA-AITES)
            if any(domain in combined_source for domain in engineering_academic_domains):
                score += 20.0  # Highest priority for engineering academic sources
            elif 'arxiv' in source or 'semantic' in source or 'openalex' in source:
                score += 10.0  # Academic sources
            elif 'crossref' in source:
                score += 8.0
            elif 'wikipedia' in source:
                score += 2.0  # Lower priority for Wikipedia
            
            # Citation count
            citations = result.get('citations', 0) or 0
            score += min(citations / 10.0, 5.0)  # Cap at 5 points
            
            # Title match
            title = result.get('title', '').lower()
            title_words = set(title.split())
            title_match = len(query_words.intersection(title_words)) / max(len(query_words), 1)
            score += title_match * 5.0
            
            # Abstract match
            abstract = result.get('abstract', '').lower()
            abstract_words = set(abstract.split())
            abstract_match = len(query_words.intersection(abstract_words)) / max(len(query_words), 1)
            score += abstract_match * 3.0
            
            # Recency (prefer recent papers)
            year = result.get('year')
            if year and isinstance(year, int):
                current_year = datetime.now().year
                age = current_year - year
                if age <= 5:
                    score += 2.0
                elif age <= 10:
                    score += 1.0
            
            return score
        
        # Sort by score
        scored_results = [(result, score_result(result)) for result in results]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, _ in scored_results]
    
    def format_as_references(self, results: List[Dict[str, Any]], start_index: int = 1) -> List[str]:
        """Format search results as reference list entries."""
        references = []
        for idx, result in enumerate(results, start=start_index):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            source = result.get('source', '')
            authors = result.get('authors', '')
            year = result.get('year', '')
            
            # Format: [N] Title — URL (Source, Authors, Year)
            ref_parts = [f"[{idx}] {title}"]
            if url:
                ref_parts.append(f" — {url}")
            if source or authors or year:
                meta_parts = [p for p in [source, authors, str(year) if year else None] if p]
                if meta_parts:
                    ref_parts.append(f" ({', '.join(meta_parts)})")
            
            references.append(''.join(ref_parts))
        
        return references

