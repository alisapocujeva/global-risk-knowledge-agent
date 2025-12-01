# How to Run and Test the Global Risk Knowledge Agent

## Quick Test (Code Validation)

Run a quick syntax check without dependencies:
```bash
python quick_test.py
```

## Full Agent Testing

For comprehensive testing, use Docker (see below).

## Prerequisites

1. **Perplexity API Key**: Get your API key from https://www.perplexity.ai/settings/api
2. **Python 3.10+** (if running locally)
3. **Docker** (if using containerized deployment)

## Option 1: Run with Docker (Recommended)

### Step 1: Set Environment Variable

**Windows PowerShell:**
```powershell
$env:PERPLEXITY_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set PERPLEXITY_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export PERPLEXITY_API_KEY="your-api-key-here"
```

### Step 2: Build Docker Image

```bash
docker build -t global-risk-agent .
```

This will:
- Install all dependencies (sentence-transformers, chromadb, etc.)
- Set up the application

**Build Time Notes:**
- **First build:** 5-10 minutes (compiling C++ extensions for sentence-transformers/chromadb)
- **Subsequent builds:** 2-3 minutes (if only code changed, dependencies are cached)
- **Model download:** Happens automatically on first query (lazy loading, ~30 seconds)

**To speed up builds:**
- Use Docker BuildKit: `DOCKER_BUILDKIT=1 docker build -t global-risk-agent .`
- Rebuild only when dependencies change (Docker caches layers automatically)

### Step 3: Run Container

**Option 1: Using .env file (Recommended)**
```powershell
docker run -p 8501:8501 --env-file .env global-risk-agent
```

**Option 2: Using environment variable directly**
```powershell
docker run -p 8501:8501 -e PERPLEXITY_API_KEY=pplx-your-key-here global-risk-agent
```

**Option 3: Using PowerShell environment variable**
```powershell
docker run -p 8501:8501 -e PERPLEXITY_API_KEY=$env:PERPLEXITY_API_KEY global-risk-agent
```

**Note:** If port 8501 is already in use, you can:
- Stop existing containers: `docker stop $(docker ps -q)`
- Use a different port: `docker run -p 8502:8501 --env-file .env global-risk-agent` (then access at http://localhost:8502)

### Step 4: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

---

## Option 2: Run Locally (Without Docker)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Environment Variable

**Windows PowerShell:**
```powershell
$env:PERPLEXITY_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set PERPLEXITY_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export PERPLEXITY_API_KEY="your-api-key-here"
```

### Step 3: Verify Index Exists

The index should already be built in the `index/` directory. If not, run:

```bash
python build_index.py
```

This will:
- Load all PDFs from `internal_docs/`
- Extract text and build ChromaDB vector index
- Save to `index/` directory

### Step 4: Run Streamlit App

```bash
cd app
streamlit run app.py
```

Or from project root:
```bash
streamlit run app/app.py
```

### Step 5: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

---

## Testing the Agent

### Test Query Examples

1. **Tunnel Projects:**
   ```
   Assess the impact of unexpected groundwater contamination on TBM operations in urban tunnel projects
   ```

2. **Marine Projects:**
   ```
   What are the main risks in harbor construction projects and how have similar projects mitigated them?
   ```

3. **Pharmaceutical Facilities:**
   ```
   Analyze risks in pharmaceutical cleanroom construction and GMP compliance
   ```

4. **General Infrastructure:**
   ```
   What are common risk drivers in large-scale infrastructure projects?
   ```

### What to Expect

1. **Processing Time:** 2-5 minutes per query
   - Internal document search: ~5-10 seconds
   - LLM summarization: ~30-60 seconds per project
   - External research APIs: ~10-20 seconds
   - Perplexity API: ~60-180 seconds

2. **Output Sections:**
   - ✅ Executive Summary
   - ✅ Key External Risks (with citations)
   - ✅ Similar Global Projects (10-20 projects table)
   - ✅ Internal Lessons Learned (3 formats)
   - ✅ Benchmarking Against Regions
   - ✅ Recommended Actions
   - ✅ Full Reference List (50-100 sources)

3. **Internal Lessons Format:**
   - Narrative Synthesis (detailed project summaries)
   - Summary Table (concise overview)
   - Cross-Project Themes (pattern analysis)

### Troubleshooting

**Error: "PERPLEXITY_API_KEY environment variable is not set"**
- Solution: Make sure you've set the environment variable before running
- For Docker: Use `-e PERPLEXITY_API_KEY=your-key`

**Error: "401 Authorization Required"**
- Solution: Check that your API key is valid at https://www.perplexity.ai/settings/api
- Make sure there are no extra spaces in the key

**Error: "No internal evidence found"**
- Solution: Verify that `index/` directory exists and contains ChromaDB files
- Rebuild index: `python build_index.py`

**Slow Performance:**
- First run: Models download (~500MB), expect 5-10 minutes
- Subsequent runs: Should be faster (2-5 minutes per query)
- LLM calls take time - this is normal

**Port 8501 already in use:**
- Change port: `streamlit run app/app.py --server.port 8502`
- Or stop the existing Streamlit process

---

## Testing the Agent

### Option 1: Quick Syntax Check (No Dependencies)

```bash
python quick_test.py
```

This validates code syntax without requiring dependencies.

### Option 2: Full Test Suite (In Docker)

Once Docker container is running:

```bash
# Get container ID
docker ps

# Run comprehensive test suite
docker exec <container-id> python /app/test_agent.py
```

Or run individual tests:

```bash
# Test agent initialization
docker exec <container-id> python -c "from app.agent import RiskIntelligenceAgent; agent = RiskIntelligenceAgent(); print('✓ Agent initialized')"

# Test internal search
docker exec <container-id> python -c "from app.agent import RiskIntelligenceAgent; agent = RiskIntelligenceAgent(); results = agent.internal_search('tunnel risks'); print(f'Found {len(results)} results')"

# Test full answer (requires API key)
docker exec <container-id> python -c "from app.agent import RiskIntelligenceAgent; agent = RiskIntelligenceAgent(); answer = agent.answer('What are tunnel construction risks?'); print(f'Generated {len(answer)} chars')"
```

### Option 3: Manual Test Script

Create a file `test_agent.py`:

```python
import os
from app.agent import RiskIntelligenceAgent

# Set API key
os.environ["PERPLEXITY_API_KEY"] = "your-api-key-here"

# Initialize agent
agent = RiskIntelligenceAgent()

# Test query
query = "What are the main risks in tunnel construction projects?"

# Get answer
print("Generating analysis...")
report = agent.answer(query)

# Save to file
with open("test_output.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Report saved to test_output.md")
print(f"Report length: {len(report)} characters")
```

Run with:
```bash
python test_agent.py
```

---

## Verification Checklist

- [ ] Environment variable `PERPLEXITY_API_KEY` is set
- [ ] Index directory exists with ChromaDB files
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Streamlit app starts without errors
- [ ] Test query produces a report with all sections
- [ ] Internal lessons section appears with 3 formats
- [ ] Reference list contains 50+ sources
- [ ] Similar projects table has 10+ rows

---

## Performance Notes

- **First Query:** May take longer (model initialization)
- **Subsequent Queries:** Faster (models cached)
- **API Rate Limits:** Perplexity has rate limits - if you hit them, wait a few minutes
- **Timeout:** Perplexity calls have 10-minute timeout (600 seconds)

---

## Next Steps

Once running:
1. Try different query types (tunnel, marine, pharma, etc.)
2. Verify internal lessons are rewritten (not copied)
3. Check that external sources are diverse and authoritative
4. Confirm cross-project themes are extracted
5. Verify all sections appear in the final report

