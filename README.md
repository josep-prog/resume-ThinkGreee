# Resume Parser and ATS System

Professional resume screening and candidate ranking system that processes resumes (PDF, DOCX, TXT) and grades them against job criteria to save organizations time identifying top applicants.

## Overview

This system uses Natural Language Processing (NER) and machine learning to:
- Extract structured information from resumes (name, email, phone, skills, experience, education)
- Match candidates against job descriptions
- Rank candidates based on similarity and skill match scores
- Generate comprehensive ATS reports and analytics dashboards

## Features

- **Multi-format Support**: PDF, DOCX, and TXT files
- **OCR Integration**: Extract text from images in resumes
- **NER-based Extraction**: Uses pretrained models for accurate entity recognition
- **TF-IDF Ranking**: Semantic similarity matching between resumes and job descriptions
- **Skill Matching**: Database-driven skill extraction and matching
- **Analytics Dashboard**: Interactive visualizations of candidate rankings
- **Model Retraining**: Fine-tune NER model with your resume data

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. (Optional) Install Tesseract OCR for image text extraction:
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Basic Usage

```python
from resume_processor_optimized import ResumeParser

# Initialize parser
parser = ResumeParser()

# Process all resumes in a folder
parser.process_resume_folder("./resumes")

# Define job requirements
job_description = """
Looking for a Python Developer with machine learning experience.
Required: Python, pandas, scikit-learn, Django, REST APIs, SQL.
Preferred: AWS, cloud platforms, problem-solving skills.
"""

# Rank candidates
top_candidates = parser.rank_candidates(job_description, top_k=10)

# Generate ATS report
report_df = parser.generate_ats_report(top_candidates)

# Create analytics dashboard
parser.create_analytics_dashboard(report_df)

# Save processed data
parser.save_processed_data()
```

### Command Line Usage

```bash
python resume_processor_optimized.py
```

When prompted, enter the path to your resume folder.

## Output Files

The system generates several output files:

1. **ATS Report** (`ats_report_YYYYMMDD_HHMMSS.csv`): CSV file with candidate rankings
2. **Analytics Dashboard** (`analytics_dashboard_YYYYMMDD_HHMMSS.html`): Interactive HTML dashboard
3. **Processed Data** (`processed_resumes_YYYYMMDD_HHMMSS.json`): JSON file with extracted data

## Scoring System

Candidates are scored using two metrics:

1. **Similarity Score (70% weight)**: TF-IDF cosine similarity between resume and job description
2. **Skill Match Score (30% weight)**: Percentage of required skills matched

**Total Score** = (Similarity × 0.7) + (Skill Match × 0.3)

## Skills Database

The system includes a comprehensive skills database covering:
- Programming languages (Python, Java, JavaScript, C++, etc.)
- Web development (React, Angular, Django, Flask, etc.)
- Data science (pandas, TensorFlow, PyTorch, etc.)
- Databases (MySQL, PostgreSQL, MongoDB, etc.)
- Cloud platforms (AWS, Azure, GCP, Docker, Kubernetes, etc.)
- Soft skills (leadership, communication, problem-solving, etc.)

You can extend the skills database by modifying `self.skills_database` in the `ResumeParser` class.

## Model Retraining

To improve accuracy with your specific resume format:

```python
# After processing resumes
parser.retrain_model()
```

This fine-tunes the NER model on your processed resumes.

## Performance Optimizations

- Removed duplicate imports and code
- Streamlined text extraction and cleaning
- Optimized skill matching with set operations
- Reduced redundant regex operations
- Simplified OCR processing
- Efficient TF-IDF vectorization

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended for model training)
- GPU optional (speeds up OCR and model training)

## Architecture

```
ResumeParser
├── Document Processing
│   ├── PDF extraction (PyPDF2, PyMuPDF)
│   ├── DOCX extraction (python-docx)
│   ├── Image extraction
│   └── OCR (EasyOCR, Tesseract)
│
├── Entity Extraction
│   ├── NER model (Transformers)
│   ├── Regex patterns
│   └── Skills database matching
│
├── Candidate Ranking
│   ├── TF-IDF vectorization
│   ├── Cosine similarity
│   └── Skill matching
│
└── Output Generation
    ├── ATS report (CSV)
    ├── Analytics dashboard (HTML)
    └── Processed data (JSON)
```

## Troubleshooting

**Model loading fails**: The system will automatically fall back to spaCy. Ensure you've installed the spaCy model.

**OCR not working**: Install EasyOCR or Tesseract. The system will work without OCR but won't extract text from images.

**Memory errors during training**: Reduce batch size in `TrainingArguments` or process fewer resumes at once.

## License

MIT License

## Support

For issues or questions, please review the code documentation or create an issue in the project repository.
