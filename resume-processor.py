#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume Parser and ATS System
Simplified version with minimal dependencies
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Try to import required libraries, with helpful error messages
try:
    # Document processing - core dependencies
    import PyPDF2
    from docx import Document
    
    # ML libraries - make sklearn optional
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction import text as sklearn_text
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
        print("⚠️  scikit-learn not available. Using basic text matching instead.")
    
    # Make other ML libraries optional
    try:
        import spacy
        SPACY_AVAILABLE = True
    except ImportError:
        SPACY_AVAILABLE = False
        print("⚠️  spaCy not available. Using regex-based extraction only.")
    
    try:
        import torch
        from transformers import pipeline
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        print("⚠️  transformers not available. Using basic extraction methods.")

except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("\n📦 Please install the required packages first:")
    print("pip install PyPDF2 python-docx pandas numpy")
    print("Optional: pip install scikit-learn spacy transformers torch")
    sys.exit(1)


class ResumeParser:
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the Resume Parser

        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu
        self.resume_data = []
        
        # Initialize components
        self._initialize_skills_database()
        self._initialize_custom_stop_words()
        self.load_models()

    def _initialize_skills_database(self) -> None:
        """Initialize comprehensive skills database"""
        self.skills_database = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                          'swift', 'kotlin', 'php', 'ruby', 'scala', 'r', 'matlab'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
                              'django', 'flask', 'spring', 'fastapi', 'graphql', 'rest api'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                           'matplotlib', 'seaborn', 'plotly', 'jupyter', 'spark', 'hadoop'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'sqlite', 'redis', 'elasticsearch',
                         'cassandra', 'oracle', 'sql server', 'dynamodb'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins',
                     'gitlab', 'ansible', 'prometheus', 'grafana'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving',
                           'analytical', 'creative', 'management', 'collaboration', 'adaptability']
        }

        # Flatten skills for quick lookup
        self.all_skills = [skill for skills in self.skills_database.values() for skill in skills]

    def _initialize_custom_stop_words(self) -> None:
        """Initialize custom stop words for resume processing"""
        if SKLEARN_AVAILABLE:
            self.custom_stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)
        else:
            # Basic stop words if sklearn not available
            self.custom_stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
                "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
                "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', 
                "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
            }
        
        # Add resume-specific stop words
        resume_stop_words = {'resume', 'cv', 'curriculum', 'vitae', 'page', 'email', 'phone',
                           'linkedin', 'github', 'contact', 'references'}
        self.custom_stop_words.update(resume_stop_words)

    def load_models(self) -> None:
        """Load available ML models"""
        self.nlp = None
        self.nlp_pipeline = None
        
        # Try to load spaCy
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
        
        # Try to load transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                self.nlp_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                logger.info("Transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")

    def process_single_resume(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single resume file
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Processed candidate data or None if failed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        logger.info(f"Processing single resume: {file_path.name}")
        
        text = self.preprocess_resume(file_path)
        if not text:
            logger.error(f"Failed to extract text from: {file_path.name}")
            return None
            
        candidate_data = self.extract_candidate_data(text, file_path.name)
        if candidate_data:
            self.resume_data.append(candidate_data)
            logger.info(f"Successfully processed: {file_path.name}")
            return candidate_data
        else:
            logger.error(f"Failed to extract data from: {file_path.name}")
            return None

    def preprocess_resume(self, file_path: Path) -> Optional[str]:
        """
        Preprocess resume files with optimized text extraction

        Args:
            file_path: Path to resume file

        Returns:
            Extracted and cleaned text or None if failed
        """
        try:
            file_extension = file_path.suffix.lower()

            if file_extension == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_extension == '.docx':
                return self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                return self._extract_txt_text(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return None

        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {e}")
            return None

    def _extract_pdf_text(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return self._clean_text(text) if text.strip() else None
        except Exception as e:
            logger.error(f"Error extracting PDF text from {file_path}: {e}")
            return None

    def _extract_docx_text(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return self._clean_text(text) if text.strip() else None
        except Exception as e:
            logger.error(f"Error extracting DOCX text from {file_path}: {e}")
            return None

    def _extract_txt_text(self, file_path: Path) -> Optional[str]:
        """Extract text from TXT files with encoding fallback"""
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    if text.strip():
                        return self._clean_text(text)
            except UnicodeDecodeError:
                continue

        logger.error(f"Failed to decode text file: {file_path}")
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s@.+/-]', '', text)

        # Remove multiple dots
        text = re.sub(r'\.{2,}', '.', text)

        return text

    def extract_candidate_data(self, text: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from resume text

        Args:
            text: Resume text content
            filename: Original filename

        Returns:
            Structured candidate data or None if extraction fails
        """
        if not text:
            return None

        try:
            candidate_data = {
                'filename': filename,
                'processed_date': datetime.now().isoformat(),
                'raw_text': text,
                'text_length': len(text),
                'word_count': len(text.split()),
                'name': self._extract_name(text),
                'email': self._extract_email(text),
                'phone': self._extract_phone(text),
                'skills': self._extract_skills(text),
                'experience': self._extract_experience(text),
                'education': self._extract_education(text),
                'organizations': self._extract_organizations(text),
                'locations': self._extract_locations(text)
            }

            # Post-process and validate
            candidate_data = self._post_process_candidate_data(candidate_data)

            return candidate_data

        except Exception as e:
            logger.error(f"Error extracting candidate data from {filename}: {e}")
            return None

    def _extract_name(self, text: str) -> str:
        """Extract candidate name"""
        lines = text.split('\n')[:10]

        for line in lines:
            line = line.strip()
            if self._is_valid_name(line):
                return line

        return ""

    def _is_valid_name(self, name: str) -> bool:
        """Validate if extracted text is a valid name"""
        if len(name) < 2 or len(name) > 50:
            return False
        if re.search(r'[\d@]', name):
            return False
        if re.search(r'(resume|cv|curriculum|http|www|\.com)', name.lower()):
            return False
        return len(name.split()) >= 2

    def _extract_email(self, text: str) -> str:
        """Extract email address"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else ""

    def _extract_phone(self, text: str) -> str:
        """Extract phone number"""
        phone_patterns = [
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(\+?\d{1,3}[-.\s]?)?\d{2,5}[-.\s]?\d{2,5}[-.\s]?\d{4,5}',
        ]

        for pattern in phone_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                phone = match.group().strip()
                # Validate phone number length
                digits = re.sub(r'\D', '', phone)
                if 10 <= len(digits) <= 15:
                    return phone

        return ""

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = set()

        # Check against skills database
        for skill in self.all_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)

        # Extract from skills sections
        skills_section_patterns = [
            r'(?i)(?:skills?|technical skills?|core competencies|technologies|expertise)[:\s]*(.*?)(?=\n\s*\n|\n[A-Z][a-z]|\Z)',
            r'(?i)(?:programming languages|frameworks|tools)[:\s]*(.*?)(?=\n\s*\n|\n[A-Z][a-z]|\Z)'
        ]

        for pattern in skills_section_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    skills_text = ' '.join(match)
                else:
                    skills_text = match

                # Extract skills from the section
                skills = re.findall(r'[•\-\*]?\s*([A-Za-z][A-Za-z0-9+#\.\s\-]{2,30})(?=[,\n•\-\*]|$)', skills_text)
                found_skills.update([skill.strip() for skill in skills if len(skill.strip()) > 2])

        return list(found_skills)

    def _extract_experience(self, text: str) -> List[str]:
        """Extract work experience"""
        experience_patterns = [
            r'(?i)(?:experience|work history|employment|professional experience)(.*?)(?=education|skills|projects|\Z)',
            r'(\d{4}[-\s]*(?:to|–|-)?\s*(?:\d{4}|present|current|now)).*?([A-Za-z].*?)(?=\n\d{4}|\n[A-Z]|\Z)'
        ]

        experiences = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    exp_text = ' '.join([m.strip() for m in match if m.strip()])
                else:
                    exp_text = match.strip()

                if exp_text and len(exp_text) > 10:
                    experiences.append(exp_text)

        return experiences[:5]

    def _extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_patterns = [
            r'(?i)(?:education|academic background|qualifications)(.*?)(?=experience|skills|projects|\Z)',
            r'\b(?:bachelor|master|phd|doctorate|diploma|degree|bs?|ms?|mba)[\s\w.,]*?(?=\n|,|\Z)',
            r'\b(19|20)\d{2}\b.*?(?:bachelor|master|phd|bs?|ms?|mba)'
        ]

        education = []
        for pattern in education_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                edu_text = match.group().strip()
                if edu_text and len(edu_text) > 5:
                    education.append(edu_text)

        return education[:3]

    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organizations/companies"""
        org_pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Ltd|Corp|Company|Corporation))\b'
        return list(set(re.findall(org_pattern, text)))

    def _extract_locations(self, text: str) -> List[str]:
        """Extract locations"""
        location_pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b)'
        return list(set(re.findall(location_pattern, text)))

    def _post_process_candidate_data(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and validate extracted candidate data"""
        # Remove duplicates from lists
        for key in ['skills', 'experience', 'education', 'organizations', 'locations']:
            if key in candidate_data and isinstance(candidate_data[key], list):
                candidate_data[key] = list(set(candidate_data[key]))

        # Validate email format
        if candidate_data.get('email'):
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, candidate_data['email']):
                candidate_data['email'] = ""

        return candidate_data

    def rank_candidates(self, job_description: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rank candidates based on job description similarity

        Args:
            job_description: Job description text
            top_k: Number of top candidates to return

        Returns:
            List of ranked candidates
        """
        if not self.resume_data:
            return []

        job_desc_clean = self._clean_text(job_description.lower())

        # Calculate scores for each candidate
        for candidate in self.resume_data:
            if SKLEARN_AVAILABLE:
                candidate['similarity_score'] = self._calculate_text_similarity(candidate, job_desc_clean)
            else:
                candidate['similarity_score'] = self._calculate_basic_similarity(candidate, job_desc_clean)
            
            candidate['skill_match_score'] = self._calculate_skill_match(candidate, job_desc_clean)
            candidate['experience_score'] = self._calculate_experience_score(candidate, job_desc_clean)
            
            # Combined score
            candidate['total_score'] = (
                candidate['similarity_score'] * 0.5 +
                candidate['skill_match_score'] * 0.3 +
                candidate['experience_score'] * 0.2
            )

        # Sort by total score
        ranked_candidates = sorted(self.resume_data, key=lambda x: x.get('total_score', 0), reverse=True)
        return ranked_candidates[:top_k]

    def _calculate_text_similarity(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Calculate TF-IDF based text similarity"""
        candidate_text = self._prepare_candidate_text(candidate)

        if not candidate_text:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(self.custom_stop_words),
                max_features=1000,
                ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform([job_description, candidate_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            return 0.0

    def _calculate_basic_similarity(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Calculate basic text similarity using word overlap"""
        candidate_text = self._prepare_candidate_text(candidate).lower()
        job_words = set(job_description.lower().split())
        candidate_words = set(candidate_text.split())
        
        if not job_words:
            return 0.0
            
        overlap = job_words.intersection(candidate_words)
        return len(overlap) / len(job_words)

    def _prepare_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Prepare candidate text for similarity calculation"""
        text_parts = []

        # Add skills
        text_parts.extend(candidate.get('skills', []))

        # Add experience (first 200 chars of each experience)
        for exp in candidate.get('experience', [])[:3]:
            text_parts.append(exp[:200])

        # Add education
        text_parts.extend(candidate.get('education', []))

        return ' '.join(text_parts).lower()

    def _calculate_skill_match(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Calculate skill match score"""
        candidate_skills = set(skill.lower() for skill in candidate.get('skills', []))

        # Extract skills from job description
        job_skills = set()
        for skill in self.all_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', job_description.lower()):
                job_skills.add(skill.lower())

        if not job_skills:
            return 0.0

        matched_skills = candidate_skills & job_skills
        return len(matched_skills) / len(job_skills)

    def _calculate_experience_score(self, candidate: Dict[str, Any], job_description: str) -> float:
        """Calculate experience relevance score"""
        experience_text = ' '.join(candidate.get('experience', [])).lower()

        if not experience_text:
            return 0.0

        # Simple keyword matching for experience relevance
        experience_keywords = ['years', 'experience', 'developed', 'managed', 'led', 'created', 'built']
        job_keywords = re.findall(r'\b[a-z]{4,15}\b', job_description.lower())

        relevant_keywords = set(experience_keywords) & set(job_keywords)

        if not job_keywords:
            return 0.0

        return len(relevant_keywords) / len(job_keywords)

    def generate_report(self, ranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate ATS report

        Args:
            ranked_candidates: List of ranked candidates

        Returns:
            DataFrame with candidate rankings
        """
        report_data = []
        for i, candidate in enumerate(ranked_candidates, 1):
            report_data.append({
                'Rank': i,
                'Name': candidate.get('name', 'N/A'),
                'Email': candidate.get('email', 'N/A'),
                'Phone': candidate.get('phone', 'N/A'),
                'Total Score': round(candidate.get('total_score', 0), 3),
                'Similarity Score': round(candidate.get('similarity_score', 0), 3),
                'Skill Match Score': round(candidate.get('skill_match_score', 0), 3),
                'Experience Score': round(candidate.get('experience_score', 0), 3),
                'Skills Count': len(candidate.get('skills', [])),
                'Experience Count': len(candidate.get('experience', [])),
                'Top Skills': ', '.join(candidate.get('skills', [])[:5]),
                'Filename': candidate.get('filename', 'N/A')
            })

        return pd.DataFrame(report_data)

    def save_data(self) -> None:
        """Save processed resume data"""
        if not self.resume_data:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save as JSON
        json_file = f"processed_resumes_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.resume_data, f, indent=2, ensure_ascii=False, default=str)

        # Save as CSV
        csv_data = []
        for candidate in self.resume_data:
            flat_candidate = {
                'name': candidate.get('name', ''),
                'email': candidate.get('email', ''),
                'phone': candidate.get('phone', ''),
                'filename': candidate.get('filename', ''),
                'similarity_score': candidate.get('similarity_score', 0),
                'skill_match_score': candidate.get('skill_match_score', 0),
                'total_score': candidate.get('total_score', 0),
                'skills_count': len(candidate.get('skills', [])),
                'experience_count': len(candidate.get('experience', [])),
                'education_count': len(candidate.get('education', [])),
                'skills': ', '.join(candidate.get('skills', [])),
                'top_skills': ', '.join(candidate.get('skills', [])[:5])
            }
            csv_data.append(flat_candidate)

        csv_file = f"processed_resumes_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)

        print(f"💾 Data saved to: {json_file} and {csv_file}")


def get_resume_path():
    """Prompt user for resume file path"""
    print("🚀 Resume Parser and ATS System")
    print("=" * 50)
    
    # Check if file path was provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if Path(file_path).exists():
            return file_path
        else:
            print(f"❌ File not found: {file_path}")
    
    # Interactive prompt
    while True:
        print("\n📁 Please enter the path to your resume file:")
        print("   Supported formats: PDF, DOCX, TXT")
        print("   Example: /home/user/resume.pdf or ./resume.docx")
        
        file_path = input("\n📄 Resume file path: ").strip()
        
        # Handle quoted paths and expand user home directory
        file_path = file_path.strip('"\'')
        file_path = os.path.expanduser(file_path)
        
        if not file_path:
            print("❌ Please enter a file path")
            continue
            
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        if file_path.suffix.lower() not in ['.pdf', '.docx', '.txt']:
            print(f"❌ Unsupported file format: {file_path.suffix}")
            print("   Please use PDF, DOCX, or TXT files")
            continue
            
        return str(file_path)


def main():
    """Main function with interactive file path input"""
    try:
        # Get resume file path from user
        resume_file = get_resume_path()
        
        print(f"\n📄 Processing resume: {resume_file}")
        
        # Initialize parser
        parser = ResumeParser()
        
        # Process the resume
        start_time = time.time()
        candidate_data = parser.process_single_resume(resume_file)
        processing_time = time.time() - start_time

        print(f"✅ Processing completed in {processing_time:.2f} seconds")

        if not candidate_data:
            print("❌ Failed to process resume. Please check the file format and content.")
            return

        # Display extracted data
        print("\n📊 EXTRACTED CANDIDATE DATA:")
        print("=" * 50)
        print(f"Name: {candidate_data.get('name', 'N/A')}")
        print(f"Email: {candidate_data.get('email', 'N/A')}")
        print(f"Phone: {candidate_data.get('phone', 'N/A')}")
        print(f"Skills Count: {len(candidate_data.get('skills', []))}")
        print(f"Experience Count: {len(candidate_data.get('experience', []))}")
        print(f"Education Count: {len(candidate_data.get('education', []))}")
        
        if candidate_data.get('skills'):
            print(f"\n🛠️ Top Skills: {', '.join(candidate_data.get('skills', [])[:10])}")
        
        if candidate_data.get('experience'):
            print(f"\n💼 Recent Experience: {candidate_data.get('experience', [])[0][:100]}...")
        
        if candidate_data.get('education'):
            print(f"🎓 Education: {candidate_data.get('education', [])[0]}")

        # Example job description
        job_description = """
        We are looking for a Senior Python Developer with 5+ years of experience in
        machine learning and data science. Required skills include Python, pandas,
        scikit-learn, TensorFlow, Django, REST APIs, and SQL. Experience with cloud
        platforms like AWS and Docker is preferred. Strong problem-solving skills
        and experience leading technical teams is required. The ideal candidate
        should have a Bachelor's or Master's degree in Computer Science or related field.
        """

        print("\n📋 JOB DESCRIPTION:")
        print(job_description.strip())
        print("=" * 50)

        # Rank candidates
        start_time = time.time()
        top_candidates = parser.rank_candidates(job_description, top_k=5)
        ranking_time = time.time() - start_time

        print(f"✅ Ranking completed in {ranking_time:.2f} seconds")

        # Generate report
        report_df = parser.generate_report(top_candidates)

        # Display results
        print("\n🏆 RANKING RESULTS:")
        print("=" * 50)
        for i, candidate in enumerate(top_candidates, 1):
            print(f"{i}. {candidate.get('name', 'N/A')}")
            print(f"   Total Score: {candidate.get('total_score', 0):.3f}")
            print(f"   Similarity Score: {candidate.get('similarity_score', 0):.3f}")
            print(f"   Skill Match Score: {candidate.get('skill_match_score', 0):.3f}")
            print(f"   Experience Score: {candidate.get('experience_score', 0):.3f}")
            print(f"   Skills: {', '.join(candidate.get('skills', [])[:5])}")
            print("-" * 40)

        # Save processed data
        print("\n💾 Saving processed data...")
        parser.save_data()

        print("\n✅ Resume Parser completed successfully!")
        print("📁 Generated files in current directory:")
        print("   - Processed Resumes (JSON & CSV)")

    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check that all dependencies are installed:")
        print("pip install PyPDF2 python-docx pandas numpy")


if __name__ == "__main__":
    main()
