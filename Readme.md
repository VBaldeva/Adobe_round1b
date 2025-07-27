# Challenge 1B: Persona-Driven Document Intelligence

## Approach Overview

This solution builds an intelligent document analyst that extracts and prioritizes the most relevant sections from PDF collections based on a specific persona and their job-to-be-done. **The system is designed to work completely offline** with all models pre-downloaded during Docker build.

## Architecture

### 1. PDF Parser (`pdf_parser.py`)
**Improved Heading Detection Strategy:**
- **Font Analysis**: Analyzes font statistics across the document to identify common body text properties
- **Multi-criteria Heading Detection**: 
  - Font size comparison (larger than body text)
  - Bold formatting detection
  - Text pattern analysis (numbered headings, title case, etc.)
  - **Strict Bullet Point Filtering**: Bullet points are never treated as headings, always as content
- **Context Extraction**: Captures text following each heading as section content
- **Quality Filtering**: Removes sections with insufficient content

**Key Improvements:**
- More robust font analysis using statistical methods
- Better pattern recognition for different heading styles
- **Fixed bullet point handling** - bullets are always treated as content, not headings
- Improved filtering to reduce false positives
- Enhanced context extraction for better content understanding

### 2. Relevance Model (`relevance_model.py`)
**Hybrid Scoring Approach:**
- **Semantic Similarity (40%)**: Uses sentence transformers for deep semantic understanding
- **TF-IDF Relevance (30%)**: Captures keyword importance and document frequency
- **Keyword Matching (10%)**: Direct overlap between query and section terms
- **Position Bonus (10%)**: Earlier sections often contain key information
- **Heading Quality (5%)**: Rewards descriptive, well-formed headings
- **Content Length (5%)**: Prefers sections with substantial content

**Offline Features:**
- Graceful fallback to TF-IDF-only mode if semantic model fails
- All models pre-cached during Docker build
- No network calls during execution

### 3. Output Generator (`output_generator.py`)
**Smart Section Selection:**
- Ensures diversity across documents
- Limits sections per document for balanced representation
- Handles edge cases (empty sections, missing data)
- Generates refined text summaries for subsection analysis

## Models and Libraries Used (All Pre-Downloaded)

- **PyMuPDF (fitz)**: PDF text extraction and font analysis
- **sentence-transformers**: Semantic similarity using `all-MiniLM-L6-v2` model (200MB)
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NLTK**: Text tokenization for summarization (punkt tokenizer pre-downloaded)
- **Custom Summarizer**: TF-IDF based extractive summarization

## Offline Setup

### Model Download
All required models are automatically downloaded during Docker build:
- Sentence transformer model: `all-MiniLM-L6-v2` (~90MB)
- NLTK data: punkt tokenizer, stopwords
- PyTorch CPU version with dependencies

### Docker Build (Offline Ready)
```bash
# Build with all models pre-downloaded
docker build --platform linux/amd64 -t challenge1b:offline .
```

The build process will:
1. Download all Python dependencies
2. Download and cache the sentence transformer model
3. Download NLTK data
4. Verify all components work offline
5. Set offline environment variables

### Verification
Test offline functionality:
```bash
docker run --rm --network none challenge1b:offline python verify_offline.py
```

## How to Build and Run

### Docker Build
```bash
docker build --platform linux/amd64 -t challenge1b:latest .
```

### Docker Run (Completely Offline)
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b:latest \
  python main.py "PhD Researcher in Computational Biology" "Prepare a comprehensive literature review focusing on methodologies"
```

### Debug Mode
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  challenge1b:latest \
  python main.py "Researcher" "Find methods" --debug
```

### Local Development
```bash
pip install -r requirements.txt
python download_models.py  # Download models for offline use
python main.py "Persona" "Job Description"
```

## Offline Compliance

✅ **No network calls during execution**  
✅ **All models pre-downloaded** (~200MB total)  
✅ **Works with `--network none`**  
✅ **Graceful fallback if models fail**  
✅ **NLTK data cached locally**  
✅ **Environment variables set for offline mode**

## File Structure
```
├── main.py                 # Main execution script
├── pdf_parser.py          # PDF text extraction and heading detection
├── relevance_model.py     # Section scoring and ranking
├── output_generator.py    # JSON output generation
├── summarizer.py         # Text summarization
├── download_models.py    # Model download script
├── verify_offline.py     # Offline verification
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container build instructions
└── README.md           # This file
```

## Constraints Met

- ✅ **CPU only**: Uses CPU-optimized models
- ✅ **Model size ≤ 1GB**: Total model size ~200MB
- ✅ **Processing time ≤ 60 seconds**: Optimized for speed
- ✅ **No internet access**: Fully offline after build
- ✅ **Bullet point handling**: Fixed to treat bullets as content only