"""
Streamlit UI for Global Risk Knowledge Agent

Professional engineering tool interface with clean layout and proper table rendering.
"""

import re
from typing import Optional
import streamlit as st
import pandas as pd
from io import BytesIO

# Import agent - API key check happens lazily
from agent import RiskIntelligenceAgent

# PDF generation
try:
    from xhtml2pdf import pisa
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(page_title="Global Risk Knowledge Agent", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-color: #E2001A;
    --secondary-color: #546e7a;
    --text-color: #212529;
    --border-color: #dce4ec;
    --bg-white: #ffffff;
    --bg-light: #f5f5f5;
    --spacing-section: 1.3rem;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div {
    background-color: var(--bg-light) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] > div > div {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    background-color: var(--bg-white);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.header-section {
    margin-bottom: 2.5rem;
    text-align: center;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid var(--border-color);
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    color: var(--primary-color);
}

.subtitle {
    font-size: 18px;
    color: var(--secondary-color);
    font-weight: 500;
}

.section-divider {
    height: 1px;
    background-color: var(--border-color);
    margin: var(--spacing-section) 0;
}

.input-section {
    margin-bottom: 2rem;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

.input-label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.75rem;
    font-size: 16px;
    color: var(--text-color);
}

.stTextArea > div > div > textarea {
    min-height: 120px;
    padding: 16px;
    border: 2px solid var(--border-color);
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    background-color: var(--bg-white);
}

.stTextArea > div > div > textarea:focus {
    border-color: var(--primary-color);
    outline: none;
}

.btn-run-container {
    text-align: center;
    margin-top: 1rem;
}

.stButton > button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 14px 36px;
    font-size: 16px;
    font-weight: 600;
    border-radius: 6px;
    transition: background-color 0.2s;
}

.stButton > button:hover {
    background-color: #B80015;
}

/* Section headers - large and bold */
[data-testid="stExpander"] summary {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--primary-color) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: var(--spacing-section) 0 !important;
}

[data-testid="stExpander"] {
    margin-bottom: var(--spacing-section) !important;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 0.5rem;
    background-color: var(--bg-white);
}

/* Content styling */
[data-testid="stExpander"] .stMarkdown,
[data-testid="stExpander"] p,
[data-testid="stExpander"] li {
    font-size: 15px !important;
    font-weight: 400 !important;
    line-height: 1.6;
    color: var(--text-color);
}

[data-testid="stExpander"] ul {
    list-style-type: disc;
    padding-left: 1.5rem;
    margin: 0.75rem 0;
}

[data-testid="stExpander"] li {
    margin-bottom: 0.5rem;
}

/* Table styling - enhanced for full text display and screen fitting */
[data-testid="stExpander"] table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    margin: 1rem 0;
    background-color: var(--bg-white);
    table-layout: auto;
}

[data-testid="stExpander"] table th {
    background-color: #f8f9fa;
    color: var(--primary-color);
    font-weight: 600;
    padding: 10px 8px;
    text-align: left;
    border: 1px solid var(--border-color);
    font-size: 11px;
    white-space: normal;
    word-wrap: break-word;
}

[data-testid="stExpander"] table td {
    padding: 10px 8px;
    border: 1px solid var(--border-color);
    vertical-align: top;
    word-wrap: break-word;
    white-space: normal;
    font-size: 11px;
    line-height: 1.4;
    hyphens: auto;
}

[data-testid="stExpander"] table tr:nth-child(even) {
    background-color: #fafbfc;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    margin: 1rem 0;
}

/* Consistent typography */
h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    color: var(--primary-color);
}

h2 {
    font-size: 22px;
    font-weight: 700;
    margin-top: var(--spacing-section);
    margin-bottom: 0.75rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def clean_content(text: str) -> str:
    """Clean content: remove three dots, ensure single dot, fix formatting."""
    if not text:
        return text
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'•\s*', '\n• ', text)
    return text.strip()


def parse_markdown_table(text: str) -> Optional[pd.DataFrame]:
    """Parse markdown table into pandas DataFrame."""
    lines = text.split("\n")
    table_lines = []
    in_table = False
    
    for line in lines:
        if line.strip().startswith("|") and "---" not in line:
            in_table = True
            table_lines.append(line)
        elif in_table and not line.strip().startswith("|"):
            break
    
    if not table_lines:
        return None
    
    try:
        # Parse table
        rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells:
                rows.append(cells)
        
        if len(rows) < 2:
            return None
        
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    except Exception:
        return None


def parse_markdown_sections(markdown_text: str) -> dict:
    """Parse markdown into sections for structured display."""
    sections = {}
    current_section = None
    current_content = []
    
    lines = markdown_text.split("\n")
    
    for line in lines:
        line_upper = line.strip().upper()
        found_section = None
        
        if line.strip().startswith("##"):
            if "EXECUTIVE SUMMARY" in line_upper:
                found_section = "EXECUTIVE SUMMARY"
            elif "KEY EXTERNAL RISKS" in line_upper:
                found_section = "KEY EXTERNAL RISKS"
            elif "SIMILAR GLOBAL PROJECTS" in line_upper or "SIMILAR PROJECTS" in line_upper:
                found_section = "SIMILAR GLOBAL PROJECTS"
            elif "INTERNAL LESSONS LEARNED" in line_upper:
                found_section = "INTERNAL LESSONS LEARNED"
            elif "RECOMMENDED ACTIONS" in line_upper:
                found_section = "RECOMMENDED ACTIONS FOR RISK MANAGERS"
            elif "FULL REFERENCE LIST" in line_upper:
                found_section = "FULL REFERENCE LIST"
        
        if found_section:
            if current_section:
                sections[current_section] = clean_content("\n".join(current_content).strip())
            current_section = found_section
            current_content = []
        else:
            if current_section and line.strip() and not line.strip().startswith("---"):
                current_content.append(line)
    
    if current_section:
        sections[current_section] = clean_content("\n".join(current_content).strip())
    
    return sections


def main():
    """Main application entry point."""
    # Header
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Global Risk Knowledge Agent</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered risk analysis for civil engineering projects.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<label class="input-label">Enter your risk question or scenario</label>', unsafe_allow_html=True)
    query = st.text_area(
        "Query input",
        placeholder="e.g., Assess the impact of unexpected groundwater contamination on TBM operations in urban tunnel projects...",
        height=120,
        label_visibility="collapsed"
    )
    st.markdown('<div class="btn-run-container">', unsafe_allow_html=True)
    generate = st.button("Run Analysis", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider" style="margin-bottom: 0;"></div>', unsafe_allow_html=True)
    
    if generate and query.strip():
        try:
            agent = RiskIntelligenceAgent()
            with st.spinner("Generating comprehensive analysis with 50-100 sources…"):
                report = agent.answer(query)
        except RuntimeError as e:
            error_msg = str(e)
            st.error(f"**Error:** {error_msg}")
            if "401" in error_msg or "authentication" in error_msg.lower() or "API key" in error_msg:
                st.info(
                    "💡 **To fix this:**\n"
                    "1. Check that `PERPLEXITY_API_KEY` is set in your environment\n"
                    "2. Verify the API key is valid at https://www.perplexity.ai/settings/api\n"
                    "3. If using Docker, ensure the environment variable is passed: `docker run -e PERPLEXITY_API_KEY=your_key ...`"
                )
            return
        
        # Parse sections
        sections = parse_markdown_sections(report)
        
        # Display sections
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Executive Summary
        with st.expander("**EXECUTIVE SUMMARY**", expanded=True):
            content = sections.get("EXECUTIVE SUMMARY", "No executive summary available.")
            st.markdown(content, unsafe_allow_html=True)
        
        # Key External Risks
        with st.expander("**KEY EXTERNAL RISKS**", expanded=True):
            content = sections.get("KEY EXTERNAL RISKS", "No external risks data available.")
            st.markdown(content, unsafe_allow_html=True)
        
        def render_table_with_full_text(content: str):
            """Render table with full text visible, properly wrapped to fit screen."""
            df = parse_markdown_table(content)
            if df is not None:
                # Calculate column widths based on content and number of columns
                num_cols = len(df.columns)
                # Dynamic width calculation: ensure all columns fit on screen
                base_width = min(200, max(120, (100 / num_cols) * 10))  # Responsive width
                
                # Convert DataFrame to HTML with proper styling for full text
                html_table = '<div style="overflow-x: auto; width: 100%; margin: 1rem 0;">'
                html_table += '<table style="width: 100%; border-collapse: collapse; font-size: 12px; table-layout: auto;">'
                
                # Header row
                html_table += '<thead><tr style="background-color: #f8f9fa;">'
                for col in df.columns:
                    html_table += f'<th style="border: 1px solid #ddd; padding: 10px 8px; text-align: left; font-weight: 600; color: #E2001A; font-size: 12px; white-space: normal; word-wrap: break-word;">{col}</th>'
                html_table += '</tr></thead>'
                
                # Data rows with text wrapping and proper line breaks
                html_table += '<tbody>'
                for idx, row in df.iterrows():
                    html_table += '<tr>'
                    for col in df.columns:
                        cell_value = str(row[col]) if pd.notna(row[col]) else ""
                        # Break long sentences into multiple lines for better readability
                        if len(cell_value) > 80:
                            # Insert line breaks after sentences (periods, exclamation, question marks)
                            cell_value = re.sub(r'([.!?])\s+', r'\1<br>', cell_value)
                            # Also break after commas in very long text
                            if len(cell_value) > 150:
                                # Break after every 2nd comma in long text
                                parts = cell_value.split(', ')
                                if len(parts) > 3:
                                    cell_value = ', '.join(parts[:2]) + ',<br>' + ', '.join(parts[2:])
                        
                        # Escape HTML to prevent injection
                        cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        # Re-apply line breaks after escaping
                        cell_value = cell_value.replace('&lt;br&gt;', '<br>')
                        
                        # Allow text to wrap and show full content - responsive column widths
                        html_table += f'<td style="border: 1px solid #ddd; padding: 10px 8px; text-align: left; word-wrap: break-word; white-space: normal; vertical-align: top; font-size: 11px; line-height: 1.4; hyphens: auto; max-width: none;">{cell_value}</td>'
                    html_table += '</tr>'
                html_table += '</tbody></table></div>'
                
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.markdown(content, unsafe_allow_html=True)
        
        # Similar Global Projects
        with st.expander("**SIMILAR GLOBAL PROJECTS**", expanded=True):
            content = sections.get("SIMILAR GLOBAL PROJECTS", "No similar projects data available.")
            render_table_with_full_text(content)
        
        # Internal Lessons Learned
        with st.expander("**INTERNAL LESSONS LEARNED**", expanded=False):
            content = sections.get("INTERNAL LESSONS LEARNED", "No internal lessons available.")
            render_table_with_full_text(content)
        
        # Recommended Actions
        with st.expander("**RECOMMENDED ACTIONS FOR RISK MANAGERS**", expanded=True):
            content = sections.get("RECOMMENDED ACTIONS FOR RISK MANAGERS", "No recommended actions available.")
            if content:
                content = content.replace('**', '')
                content = re.sub(r'•\s*', '\n• ', content)
            st.markdown(content, unsafe_allow_html=True)
        
        # Full Reference List
        with st.expander("**FULL REFERENCE LIST**", expanded=False):
            content = sections.get("FULL REFERENCE LIST", "No references available.")
            st.markdown(content, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export functionality
        st.markdown('<div class="footer-actions">', unsafe_allow_html=True)
        
        def convert_markdown_to_html(markdown_text: str) -> str:
            """Convert markdown to HTML for PDF generation."""
            html = markdown_text
            
            # Convert headers
            html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
            html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
            
            # Convert tables
            lines = html.split('\n')
            in_table = False
            table_html = []
            result_lines = []
            
            for line in lines:
                if line.strip().startswith('|') and '---' not in line:
                    if not in_table:
                        in_table = True
                        table_html = ['<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">']
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if cells:
                        row_tag = 'th' if len(table_html) == 1 else 'td'
                        row = '<tr>' + ''.join([f'<{row_tag} style="border: 1px solid #ddd; padding: 8px; text-align: left;">{cell}</{row_tag}>' for cell in cells]) + '</tr>'
                        table_html.append(row)
                else:
                    if in_table:
                        table_html.append('</table>')
                        result_lines.append('\n'.join(table_html))
                        table_html = []
                        in_table = False
                    result_lines.append(line)
            
            if in_table:
                table_html.append('</table>')
                result_lines.append('\n'.join(table_html))
            
            html = '\n'.join(result_lines)
            
            # Convert bullets
            html = re.sub(r'^• (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
            html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
            
            # Convert bold
            html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
            
            # Convert line breaks
            html = html.replace('\n', '<br>')
            
            return html
        
        def generate_pdf():
            """Generate actual PDF from report."""
            if not PDF_AVAILABLE:
                # Fallback to HTML
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Global Risk Knowledge Agent Report</title>
    <style>
        body {{ font-family: 'Inter', sans-serif; padding: 40px; line-height: 1.6; }}
        h1 {{ color: #E2001A; font-size: 24px; }}
        h2 {{ color: #E2001A; font-size: 20px; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; page-break-inside: avoid; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; word-wrap: break-word; }}
        th {{ background-color: #f8f9fa; }}
        @media print {{ @page {{ margin: 2cm; }} }}
    </style>
</head>
<body>
    <h1>Global Risk Knowledge Agent Report</h1>
    <p><strong>Query:</strong> {query}</p>
    <hr>
    {convert_markdown_to_html(report)}
</body>
</html>"""
                return html_content.encode('utf-8'), "text/html", "risk_analysis_report.html"
            
            # Generate actual PDF
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        body {{
            font-family: 'DejaVu Sans', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #212529;
        }}
        h1 {{
            color: #E2001A;
            font-size: 24pt;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #E2001A;
            font-size: 18pt;
            margin-top: 30px;
            margin-bottom: 15px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #546e7a;
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
            font-size: 9pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            word-wrap: break-word;
            vertical-align: top;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #E2001A;
        }}
        tr {{
            page-break-inside: avoid;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 5px 0;
        }}
        p {{
            margin: 10px 0;
        }}
        strong {{
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Global Risk Knowledge Agent Report</h1>
    <p><strong>Query:</strong> {query}</p>
    <hr>
    {convert_markdown_to_html(report)}
</body>
</html>"""
            
            # Convert HTML to PDF
            result = BytesIO()
            pdf = pisa.pisaDocument(BytesIO(html_content.encode('utf-8')), result)
            
            if pdf.err:
                # Fallback to HTML if PDF generation fails
                return html_content.encode('utf-8'), "text/html", "risk_analysis_report.html"
            
            return result.getvalue(), "application/pdf", "risk_analysis_report.pdf"
        
        pdf_data, mime_type, filename = generate_pdf()
        st.download_button(
            label="📄 Export as PDF",
            data=pdf_data,
            file_name=filename,
            mime=mime_type,
            use_container_width=False,
            help="Download the report as PDF" if PDF_AVAILABLE else "Download as HTML (PDF library not available)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif not generate:
        # Show placeholder
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        with st.expander("**EXECUTIVE SUMMARY**", expanded=True):
            st.markdown("• Enter a risk question above and click 'Run Analysis' to generate insights.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
