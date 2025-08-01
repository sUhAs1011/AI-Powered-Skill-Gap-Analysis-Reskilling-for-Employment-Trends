import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import fitz  # PyMuPDF
import docx
import re
import os
import json
import warnings
import cv2
import pytesseract
from PIL import Image
import io

warnings.filterwarnings('ignore')

# --- Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
MODEL_SAVE_PATH = BASE_DIR / "trained_model"
DSSM_MODEL_PATH = MODEL_SAVE_PATH / "dssm_best_model.pth"
JOB_EMBEDDINGS_COLLECTION = "jobs_embeddings"
COURSE_EMBEDDINGS_COLLECTION = "courses_embeddings"
MAPPING_FILE_PATH = BASE_DIR / "job_dataset" / "job_to_course_mapping.json"

# --- Model and App Configuration ---
DSSM_CONFIG = {
    'query_dim': 384,
    'doc_dim': 384,
    "hidden_dims": [256,128,64],  # This matches the saved model
    'dropout': 0.1,  # Use 0.2 to match the optimized training
}

# --- Utility Functions ---

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        document = docx.Document(file)
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

def extract_text_with_ocr(image):
    """Extract text from image using OCR."""
    try:
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get better text recognition
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract for OCR
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        
        return text.strip()
    except Exception as e:
        st.error(f"Error in OCR processing: {e}")
        return None

def extract_text_from_pdf_with_ocr(file):
    """Extract text from PDF using both text extraction and OCR for images."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # First try to extract text normally
            page_text = page.get_text()
            
            # If no text found, try OCR on the page image
            if not page_text.strip():
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Use OCR on the image
                ocr_text = extract_text_with_ocr(img)
                if ocr_text:
                    page_text = ocr_text
            
            text += page_text + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF with OCR: {e}")
        return None

def extract_text_from_image(file):
    """Extract text from image file using OCR."""
    try:
        image = Image.open(file)
        return extract_text_with_ocr(image)
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

def validate_job_search(job_title, client, embedding_model):
    """Simple validation: Accept any job title that finds results in the database."""
    if not job_title or len(job_title.strip()) < 2:
        return False, "Job title too short. Please enter a valid job title."
    
    try:
        # Try to find jobs in the database
        collection = client.get_collection(name=JOB_EMBEDDINGS_COLLECTION)
        query_embedding = embedding_model.encode([job_title])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Check top 5 results
            include=["metadatas"]
        )
        
        if not results or not results['ids'][0]:
            return False, f"No jobs found for '{job_title}'. Please try a different job title."
        
        # Count how many jobs were found
        job_count = len(results['ids'][0])
        
        # If we found any jobs, accept the job title
        if job_count > 0:
            return True, f"Found {job_count} jobs for '{job_title}'"
        
        return False, f"'{job_title}' doesn't match any jobs in our database. Please try a professional job title."
        
    except Exception as e:
        return False, f"Error validating job title: {str(e)}"

def extract_skills_from_resume(text):
    """
    Enhanced skill extractor that captures more comprehensive skills from resume text.
    """
    if not text:
        return []
    
    # Method 1: Extract specific technical skills using regex
    technical_pattern = r"""
        \b(
            # Programming languages
            Python|Java|C\+\+|C\#|JavaScript|TypeScript|PHP|Ruby|Go|Swift|Kotlin|Scala|R|
            # Frameworks and libraries
            React|Angular|Vue\.js|Node\.js|Django|Flask|FastAPI|Spring|Express|Laravel|
            # Databases
            SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Oracle|Redis|Cassandra|Elasticsearch|
            # Cloud platforms
            AWS|Azure|GCP|Docker|Kubernetes|Terraform|Ansible|Jenkins|
            # ML/AI
            TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Matplotlib|Seaborn|
            Machine\sLearning|Deep\sLearning|AI|NLP|Computer\sVision|
            # Tools and methodologies
            Git|CI/CD|DevOps|Agile|Scrum|RESTful\sAPIs|GraphQL|Microservices|
            # Data and analytics
            Data\sAnalysis|Data\sVisualization|Big\sData|Hadoop|Spark|Tableau|Power\sBI|
            # Other technical skills
            Linux|Windows|MacOS|Network\sEngineering|Cyber\sSecurity|Cloud\sComputing|
            UI/UX|Frontend|Backend|Full\sStack|Mobile\sDevelopment|API\sDevelopment
        )\b
    """
    
    technical_skills = re.findall(technical_pattern, text, re.IGNORECASE | re.VERBOSE)
    
    # Method 2: Extract business and domain skills
    business_pattern = r"""
        \b(
            # Business skills
            Marketing|Sales|Finance|Accounting|HR|Human\sResources|Operations|Supply\sChain|
            Customer\sService|Business\sDevelopment|Strategy|Consulting|Project\sManagement|
            Product\sManagement|Business\sAnalysis|Market\sResearch|Competitive\sAnalysis|
            # Domain expertise
            E-commerce|Retail|Healthcare|Finance|Banking|Insurance|Education|Government|
            Manufacturing|Logistics|Transportation|Real\sEstate|Media|Entertainment|
            # Soft skills
            Leadership|Communication|Teamwork|Problem\sSolving|Analytical|Critical\sThinking|
            Time\sManagement|Organization|Planning|Coordination|Collaboration|
            # Web and digital
            Web\sDevelopment|Web\sApplication|Digital\sMarketing|SEO|SEM|Social\sMedia|
            Content\sCreation|Email\sMarketing|Affiliate\sMarketing|PPC|Google\sAds|
            # Operations and processes
            Operations|Process\sImprovement|Quality\sAssurance|Six\sSigma|Lean|Kaizen|
            Workflow|Automation|Efficiency|Optimization|Standardization|
            # Qualifications and certifications
            Certification|Qualification|Degree|Bachelor|Master|PhD|MBA|CPA|PMP|PHR|
            AWS\sCertified|Azure\sCertified|Google\sCloud|CISSP|CEH|CompTIA
        )\b
    """
    
    business_skills = re.findall(business_pattern, text, re.IGNORECASE | re.VERBOSE)
    
    # Method 3: Extract multi-word skills and phrases
    # Look for patterns like "X Y" where both words are capitalized
    multi_word_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    multi_word_skills = re.findall(multi_word_pattern, text)
    
    # Filter multi-word skills to keep only relevant ones
    relevant_multi_words = []
    for skill in multi_word_skills:
        skill_lower = skill.lower()
        
        # Skip if it contains irrelevant terms
        irrelevant_terms = [
            'alarm', 'burglar', 'sem', 'using', 'powered', 'skill', 'gap',
            'analysis', 'education', 'healthcare', 'system', 'storage',
            'best', 'practice', 'role', 'rotational', 'shift', 'strict', 'adherence',
            'position', 'rotation', 'compliance', 'policy', 'procedure', 'guideline',
            'standard', 'protocol', 'requirement', 'mandatory', 'obligatory',
            'compulsory', 'essential', 'necessary', 'important', 'critical',
            'vital', 'crucial', 'primary', 'secondary', 'tertiary', 'main',
            'major', 'minor', 'senior', 'junior', 'entry', 'level', 'mid',
            'lead', 'principal', 'associate', 'assistant', 'coordinator',
            'specialist', 'expert', 'consultant', 'advisor', 'analyst',
            'technician', 'operator', 'administrator', 'supervisor', 'manager',
            'director', 'executive', 'officer', 'representative', 'agent',
            'member', 'participant', 'contributor', 'stakeholder', 'partner',
            'collaborator', 'colleague', 'peer', 'subordinate', 'superior',
            'report', 'direct', 'indirect', 'matrix', 'functional', 'line',
            'staff', 'support', 'service', 'maintenance', 'operation',
            'production', 'manufacturing', 'assembly', 'quality', 'control',
            'assurance', 'testing', 'validation', 'verification', 'inspection',
            'audit', 'review', 'assessment', 'evaluation', 'appraisal',
            'feedback', 'input', 'output', 'result', 'outcome', 'impact',
            'effect', 'influence', 'contribution', 'value', 'benefit',
            'advantage', 'disadvantage', 'pro', 'con', 'positive', 'negative',
            'good', 'bad', 'excellent', 'poor', 'average', 'above', 'below',
            'high', 'low', 'medium', 'moderate', 'extreme', 'intense',
            'mild', 'strong', 'weak', 'powerful', 'effective', 'efficient',
            'productive', 'successful', 'unsuccessful', 'failed', 'succeeded',
            'achieved', 'accomplished', 'completed', 'finished', 'done',
            'ongoing', 'continuous', 'regular', 'periodic', 'occasional',
            'frequent', 'rare', 'common', 'uncommon', 'typical', 'atypical',
            'normal', 'abnormal', 'standard', 'non-standard', 'custom',
            'default', 'optional', 'mandatory', 'required', 'necessary',
            'essential', 'important', 'critical', 'vital', 'crucial',
            'primary', 'secondary', 'tertiary', 'main', 'major', 'minor'
        ]
        if any(term in skill_lower for term in irrelevant_terms):
            continue
            
        # Keep if it contains relevant keywords
        relevant_keywords = [
            'development', 'management', 'analysis', 'design', 'engineering',
            'marketing', 'operations', 'application', 'web', 'digital',
            'business', 'project', 'product', 'customer', 'service',
            'data', 'software', 'network', 'security', 'cloud', 'database', 
            'api', 'mobile', 'frontend', 'backend', 'machine', 'learning',
            'deep', 'artificial', 'intelligence', 'nlp', 'computer', 'vision'
        ]
        if any(keyword in skill_lower for keyword in relevant_keywords):
            relevant_multi_words.append(skill)
    
    # Combine all skills
    all_skills = technical_skills + business_skills + relevant_multi_words
    
    # Filter out generic words and irrelevant terms that are not actual skills
    generic_words = {
        # Generic job-related words
        'position', 'responsibility', 'duty', 'task', 'work', 'job', 'role',
        'function', 'purpose', 'objective', 'goal', 'target', 'aim',
        'requirement', 'need', 'demand', 'expectation', 'standard',
        'quality', 'performance', 'efficiency', 'effectiveness',
        'experience', 'knowledge', 'ability', 'capability',
        'competency', 'proficiency', 'expertise', 'mastery',
        'innovation', 'creativity', 'problem', 'solution', 'approach',
        'method', 'technique', 'strategy', 'plan', 'process',
        
        # Food and culinary terms
        'savoury', 'food', 'beverage', 'cooking', 'culinary',
        
        # Personal information
        'years', 'month', 'day', 'time', 'date', 'location', 'address',
        'phone', 'email', 'contact', 'reference', 'resume', 'cv',
        
        # Irrelevant technical terms
        'sem', 'alarm', 'burglar', 'system', 'using', 'powered',
        'skill', 'gap', 'analysis', 'education', 'healthcare',
        
        # Common OCR artifacts
        'page', 'section', 'header', 'footer', 'title', 'subtitle',
        'bullet', 'point', 'list', 'item', 'number', 'letter',
        
        # Resume formatting terms
        'summary', 'objective', 'profile', 'background', 'overview',
        'highlights', 'achievements', 'accomplishments', 'responsibilities',
        
        # Generic business terms
        'company', 'organization', 'department', 'team', 'group',
        'project', 'initiative', 'program', 'campaign', 'effort',
        
        # Time-related terms
        'duration', 'period', 'timeline', 'schedule', 'deadline',
        'start', 'end', 'begin', 'finish', 'complete', 'ongoing',
        
        # Location and contact terms
        'location', 'address', 'city', 'state', 'country', 'region',
        'remote', 'onsite', 'hybrid', 'travel', 'relocation',
        
        # Education and certification terms
        'degree', 'certificate', 'diploma', 'license', 'accreditation',
        'graduation', 'enrollment', 'attendance', 'completion',
        
        # Generic action words
        'developed', 'created', 'implemented', 'managed', 'led',
        'coordinated', 'facilitated', 'supported', 'assisted', 'helped',
        'improved', 'enhanced', 'optimized', 'streamlined', 'reduced',
        'increased', 'maintained', 'monitored', 'tracked', 'analyzed',
        'researched', 'investigated', 'evaluated', 'assessed', 'reviewed',
        
        # Additional irrelevant terms from your examples
        'best', 'practice', 'role', 'rotational', 'shift', 'strict', 'adherence',
        'position', 'shift', 'rotation', 'strict', 'adherence', 'compliance',
        'policy', 'procedure', 'guideline', 'standard', 'protocol',
        'requirement', 'mandatory', 'obligatory', 'compulsory', 'essential',
        'necessary', 'important', 'critical', 'vital', 'crucial',
        'primary', 'secondary', 'tertiary', 'main', 'major', 'minor',
        'senior', 'junior', 'entry', 'level', 'mid', 'senior', 'lead',
        'principal', 'associate', 'assistant', 'coordinator', 'specialist',
        'expert', 'consultant', 'advisor', 'analyst', 'technician',
        'operator', 'administrator', 'supervisor', 'manager', 'director',
        'executive', 'officer', 'representative', 'agent', 'associate',
        'member', 'participant', 'contributor', 'stakeholder', 'partner',
        'collaborator', 'colleague', 'peer', 'subordinate', 'superior',
        'report', 'direct', 'indirect', 'matrix', 'functional', 'line',
        'staff', 'support', 'service', 'maintenance', 'operation',
        'production', 'manufacturing', 'assembly', 'quality', 'control',
        'assurance', 'testing', 'validation', 'verification', 'inspection',
        'audit', 'review', 'assessment', 'evaluation', 'appraisal',
        'feedback', 'input', 'output', 'result', 'outcome', 'impact',
        'effect', 'influence', 'contribution', 'value', 'benefit',
        'advantage', 'disadvantage', 'pro', 'con', 'positive', 'negative',
        'good', 'bad', 'excellent', 'poor', 'average', 'above', 'below',
        'high', 'low', 'medium', 'moderate', 'extreme', 'intense',
        'mild', 'strong', 'weak', 'powerful', 'effective', 'efficient',
        'productive', 'successful', 'unsuccessful', 'failed', 'succeeded',
        'achieved', 'accomplished', 'completed', 'finished', 'done',
        'ongoing', 'continuous', 'regular', 'periodic', 'occasional',
        'frequent', 'rare', 'common', 'uncommon', 'typical', 'atypical',
        'normal', 'abnormal', 'standard', 'non-standard', 'custom',
        'default', 'optional', 'mandatory', 'required', 'necessary',
        'essential', 'important', 'critical', 'vital', 'crucial',
        'primary', 'secondary', 'tertiary', 'main', 'major', 'minor'
    }
    
    # Normalize and filter skills
    processed_skills = []
    for skill in all_skills:
        skill_lower = skill.lower().strip()
        
        # Skip if it's in generic words
        if skill_lower in generic_words:
            continue
            
        # Skip if it's too short or is a number
        if len(skill_lower) <= 2 or skill_lower.isdigit():
            continue
            
        # Skip if it contains generic words (for multi-word skills)
        skill_words = skill_lower.split()
        if any(word in generic_words for word in skill_words):
            continue
            
        # Skip if it's a generic phrase (contains words like "skill", "expert", "advanced", etc.)
        generic_phrase_words = [
            'skill', 'expert', 'advanced', 'basic', 'intermediate', 'beginner', 'professional', 
            'senior', 'junior', 'level', 'proficiency', 'competency', 'capability', 'ability', 
            'knowledge', 'experience', 'expertise', 'mastery', 'proficiency', 'competence', 
            'qualification', 'certification', 'degree', 'diploma', 'certificate', 'license', 
            'accreditation', 'requirement', 'solution', 'technology', 'business', 'role',
            'position', 'function', 'duty', 'task', 'work', 'job', 'responsibility',
            'purpose', 'objective', 'goal', 'target', 'aim', 'need', 'demand',
            'expectation', 'standard', 'quality', 'performance', 'efficiency',
            'effectiveness', 'innovation', 'creativity', 'problem', 'solution',
            'approach', 'method', 'technique', 'strategy', 'plan', 'process',
            'company', 'organization', 'department', 'team', 'group', 'project',
            'initiative', 'program', 'campaign', 'effort', 'duration', 'period',
            'timeline', 'schedule', 'deadline', 'start', 'end', 'begin', 'finish',
            'complete', 'ongoing', 'continuous', 'regular', 'periodic', 'occasional',
            'frequent', 'rare', 'common', 'uncommon', 'typical', 'atypical',
            'normal', 'abnormal', 'standard', 'non-standard', 'custom', 'default',
            'optional', 'mandatory', 'required', 'necessary', 'essential', 'important',
            'critical', 'vital', 'crucial', 'primary', 'secondary', 'tertiary',
            'main', 'major', 'minor', 'lead', 'principal', 'associate', 'assistant',
            'coordinator', 'specialist', 'expert', 'consultant', 'advisor', 'analyst',
            'technician', 'operator', 'administrator', 'supervisor', 'manager',
            'director', 'executive', 'officer', 'representative', 'agent', 'member',
            'participant', 'contributor', 'stakeholder', 'partner', 'collaborator',
            'colleague', 'peer', 'subordinate', 'superior', 'report', 'direct',
            'indirect', 'matrix', 'functional', 'line', 'staff', 'support',
            'service', 'maintenance', 'operation', 'production', 'manufacturing',
            'assembly', 'quality', 'control', 'assurance', 'testing', 'validation',
            'verification', 'inspection', 'audit', 'review', 'assessment',
            'evaluation', 'appraisal', 'feedback', 'input', 'output', 'result',
            'outcome', 'impact', 'effect', 'influence', 'contribution', 'value',
            'benefit', 'advantage', 'disadvantage', 'pro', 'con', 'positive',
            'negative', 'good', 'bad', 'excellent', 'poor', 'average', 'above',
            'below', 'high', 'low', 'medium', 'moderate', 'extreme', 'intense',
            'mild', 'strong', 'weak', 'powerful', 'effective', 'efficient',
            'productive', 'successful', 'unsuccessful', 'failed', 'succeeded',
            'achieved', 'accomplished', 'completed', 'finished', 'done', 'ongoing',
            'continuous', 'regular', 'periodic', 'occasional', 'frequent', 'rare',
            'common', 'uncommon', 'typical', 'atypical', 'normal', 'abnormal',
            'standard', 'non-standard', 'custom', 'default', 'optional',
            'mandatory', 'required', 'necessary', 'essential', 'important',
            'critical', 'vital', 'crucial', 'primary', 'secondary', 'tertiary',
            'main', 'major', 'minor'
        ]
        if any(word in skill_lower for word in generic_phrase_words):
            continue
            
        processed_skills.append(skill_lower)
    
    return sorted(list(set(processed_skills)))

# --- DSSM Model Definition ---
class DSSMModel(nn.Module):
    """Deep Structured Semantic Model (DSSM). Mirrors the training script's model."""
    
    def __init__(self, query_dim, doc_dim, hidden_dims, dropout=0.1):
        super(DSSMModel, self).__init__()
        self.query_tower = self._build_tower(query_dim, hidden_dims, dropout)
        self.doc_tower = self._build_tower(doc_dim, hidden_dims, dropout)
        
    def _build_tower(self, input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        return nn.Sequential(*layers)
    
    def forward(self, query_emb, doc_emb):
        query_features = self.query_tower(query_emb)
        doc_features = self.doc_tower(doc_emb)
        return query_features, doc_features

# --- Backend Loading Functions ---

@st.cache_resource
def get_chroma_client():
    """Establishes a connection to the ChromaDB persistent client."""
    if not CHROMA_DATA_PATH.exists():
        st.error(f"ChromaDB data not found at {CHROMA_DATA_PATH}. Please run populate_chromadb.py.")
        return None
    return chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))

@st.cache_resource
def load_dssm_model():
    """Loads the pre-trained DSSM model from disk."""
    if not DSSM_MODEL_PATH.exists():
        st.error(f"DSSM model not found at {DSSM_MODEL_PATH}. Please run model_training.py.")
        return None
    
    model = DSSMModel(
        query_dim=DSSM_CONFIG['query_dim'],
        doc_dim=DSSM_CONFIG['doc_dim'],
        hidden_dims=DSSM_CONFIG['hidden_dims'],
        dropout=DSSM_CONFIG['dropout']
    )
    try:
        # Load on CPU, as we are doing inference
        model.load_state_dict(torch.load(DSSM_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading DSSM model: {e}")
        return None

@st.cache_resource
def get_embedding_model():
    """Loads the sentence transformer model for creating embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_job_course_mapping():
    """Loads the job-to-course mapping from JSON file."""
    if not MAPPING_FILE_PATH.exists():
        st.warning(f"Job-to-course mapping file not found at {MAPPING_FILE_PATH}")
        return {}
    
    try:
        with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        st.success(f"Loaded {len(mapping)} job-course mappings from JSON file")
        return mapping
    except Exception as e:
        st.error(f"Error loading mapping file: {e}")
        return {}

# --- Core Logic Functions ---

def find_target_job(job_title, client, embedding_model):
    """Finds the most relevant job in ChromaDB based on a title query."""
    if not client:
        return None, "ChromaDB client not available."
    
    try:
        collection = client.get_collection(name=JOB_EMBEDDINGS_COLLECTION)
        query_embedding = embedding_model.encode([job_title])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "documents"]
        )
        
        if not results or not results['ids'][0]:
            return None, "No matching job found for this title."
            
        job_metadata = results['metadatas'][0][0]
        job_description = results['documents'][0][0]
        
        return job_metadata, job_description
    except Exception as e:
        return None, f"Error querying ChromaDB for jobs: {e}"


def get_skills_from_job_metadata(metadata):
    """
    Extracts a list of skills from job metadata by searching multiple possible keys
    and parsing text fields.
    """
    # Normalize all keys in metadata to lowercase for case-insensitive matching.
    metadata_lower = {k.lower(): v for k, v in metadata.items()}

    # Priority 1: Check for direct skill keys.
    possible_skill_keys = ["key skills", "skills", "job_skills", "required_skills", "preferred_skills"]
    for key in possible_skill_keys:
        skills_text = metadata_lower.get(key)
        if skills_text:
            if isinstance(skills_text, list):
                return sorted(list(set([str(skill).strip().lower() for skill in skills_text])))
            elif isinstance(skills_text, str):
                return sorted(list(set([skill.strip().lower() for skill in skills_text.split(',') if skill.strip()])))

    # Priority 2: Check inside a "job_text" or "description" field.
    text_to_search = metadata_lower.get("job_text", "") or metadata_lower.get("description", "")
    if isinstance(text_to_search, str) and text_to_search:
        # Use regex to find "Key Skills: ..." and extract the list.
        match = re.search(r"key skills:\s*(.*)", text_to_search, re.IGNORECASE)
        if match:
            skills_part = match.group(1)
            # The extracted part might have other fields after it, so we stop at the next period or newline.
            skills_list_str = re.split(r'[.\\n]', skills_part)[0]
            return sorted(list(set([skill.strip().lower() for skill in skills_list_str.split(',') if skill.strip()])))
    
    # Priority 3: Try to find skills from JSON mapping file
    try:
        json_mapping = load_job_course_mapping()
        if json_mapping:
            # Extract job title from metadata
            job_title = metadata.get('job_title', metadata.get('Job Title', ''))
            if not job_title and 'job_text' in metadata_lower:
                # Try to extract from job_text field
                job_text = metadata_lower['job_text']
                if 'Job Title:' in job_text:
                    job_title = job_text.split('Job Title:')[1].split('.')[0].strip()
            
            # Search in JSON mapping
            for job_entry in json_mapping:
                if job_entry.get('job_title', '').lower() == job_title.lower():
                    required_skills = job_entry.get('required_skills', [])
                    if required_skills:
                        return sorted(list(set([str(skill).strip().lower() for skill in required_skills])))
    except:
        pass
            
    # If skills are still not found, return an empty list.
    return []


def find_course_recommendations(skill_gap, dssm_model, client, embedding_model, top_n=3, use_json_mapping=True):
    """Finds the best course recommendations for a list of missing skills."""
    if not all([client, dssm_model, embedding_model]):
        return {}, "A required model or client is missing."

    # Try to use JSON mapping first if available and requested
    if use_json_mapping:
        json_mapping = load_job_course_mapping()
        if json_mapping:
            st.info("Using pre-computed job-course mappings for recommendations...")
            return find_course_recommendations_from_json(skill_gap, json_mapping, client, top_n)
    
    # Fallback to DSSM model-based recommendations
    st.info("Using DSSM model for course recommendations...")
    return find_course_recommendations_from_dssm(skill_gap, dssm_model, client, embedding_model, top_n)

def find_course_recommendations_from_json(skill_gap, json_mapping, client, top_n=3):
    """Find course recommendations using the pre-computed JSON mapping."""
    recommendations = {}
    
    # Get all courses from ChromaDB for metadata lookup
    course_collection = client.get_collection(name=COURSE_EMBEDDINGS_COLLECTION)
    all_courses = course_collection.get(include=["metadatas"])
    
    # Create a mapping from course_id to metadata
    course_id_to_meta = {}
    for i, course_id in enumerate(all_courses['ids']):
        course_id_to_meta[course_id] = all_courses['metadatas'][i]
    
    for skill in skill_gap:
        skill_recs = []
        seen_courses = set()  # Track unique courses to avoid duplicates
        
        # Search through all job entries in the mapping
        for job_entry in json_mapping:
            job_title = job_entry.get('job_title', '').lower()
            job_skills = job_entry.get('required_skills', [])
            
            # Check if this job is related to the skill
            skill_lower = skill.lower()
            job_related = (
                skill_lower in job_title or 
                any(skill_lower in str(s).lower() for s in job_skills) or
                any(skill_lower in course_info.get('title', '').lower() 
                    for course_info in job_entry.get('top_courses', []))
            )
            
            if job_related:
                # Get courses for this job
                for course_info in job_entry.get('top_courses', []):
                    course_id = course_info.get('course_id')
                    course_title = course_info.get('title', 'Unknown Course')
                    similarity = course_info.get('similarity', 0.0)
                    
                    # Skip if we've already seen this course
                    if course_title in seen_courses:
                        continue
                    seen_courses.add(course_title)
                    
                    # Check if course metadata is available
                    if course_id in course_id_to_meta:
                        meta = course_id_to_meta[course_id]
                        organization = meta.get('organization', 'Coursera')
                        # Try to get organization from other fields if not found
                        if organization == 'Unknown Organization':
                            organization = meta.get('Organization', meta.get('organization_name', 'Coursera'))
                    else:
                        # Default to Coursera for course recommendations
                        organization = 'Coursera'
                    
                    skill_recs.append({
                        "title": course_title,
                        "organization": organization,
                        "similarity": similarity,
                        "source": "JSON Mapping"
                    })
        
        # Sort by similarity and take top N
        skill_recs.sort(key=lambda x: x['similarity'], reverse=True)
        recommendations[skill] = skill_recs[:top_n]
    
    return recommendations, None

def find_course_recommendations_from_dssm(skill_gap, dssm_model, client, embedding_model, top_n=3):
    """Find course recommendations using the DSSM model."""
    course_collection = client.get_collection(name=COURSE_EMBEDDINGS_COLLECTION)
    all_courses = course_collection.get(include=["metadatas", "embeddings"])

    recommendations = {}
    
    # Create embeddings for all courses once
    course_ids = all_courses['ids']
    course_embeddings = torch.tensor(all_courses['embeddings'], dtype=torch.float32)
    
    with st.spinner("Searching for the best courses using DSSM model..."):
        for skill in skill_gap:
            # Use the skill as the "job query"
            query_text = f"A course about {skill}"
            query_embedding = torch.tensor(embedding_model.encode([query_text]), dtype=torch.float32)
            
            # Pass through the DSSM model
            with torch.no_grad():
                dssm_model.eval()
                # The query tower expects a batch, so we need to repeat the query embedding
                repeated_query_emb = query_embedding.repeat(course_embeddings.shape[0], 1)
                
                job_features, course_features = dssm_model(repeated_query_emb, course_embeddings)
                
                # Calculate cosine similarity on the DSSM output
                similarities = F.cosine_similarity(job_features, course_features)
            
            # Get top N recommendations
            top_indices = torch.topk(similarities, k=top_n).indices.tolist()
            
            skill_recs = []
            for idx in top_indices:
                rec = {
                    "title": all_courses['metadatas'][idx].get('course_title', 'Unknown Course'),
                    "organization": all_courses['metadatas'][idx].get('organization', 'Unknown'),
                    "similarity": similarities[idx].item(),
                    "source": "DSSM Model"
                }
                skill_recs.append(rec)
            
            recommendations[skill] = skill_recs
            
    return recommendations, None

# --- Main Application ---
st.set_page_config(page_title="Career Copilot", layout="wide", initial_sidebar_state="expanded")

st.title("🚀 Career Copilot: Your Skill & Course Advisor")
st.markdown("""
Welcome to Career Copilot! This tool helps you bridge the gap between your current skills and your dream job.
1.  *Upload your resume* (PDF or DOCX).
2.  *Enter a job title* you're interested in.
3.  We'll analyze the required skills, identify any gaps, and recommend relevant courses to help you succeed.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Step 1: Your Information")
    
    # Resume Upload
    resume_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx", "png", "jpg", "jpeg"])
    
    # OCR Toggle
    use_ocr = st.checkbox("Use OCR for better text extraction", value=True, 
                          help="Enable OCR to extract text from scanned PDFs or images. Recommended for better skill detection.")
    
    # Job Title Input
    job_title_input = st.text_input("Enter Desired Job Title", placeholder="e.g., Senior Data Scientist")

    # Recommendation Method Toggle
    st.header("Step 2: Recommendation Method")
    use_json_mapping = st.checkbox("Use Pre-computed Mappings (Faster)", value=True, 
                                  help="Use the job-to-course mapping JSON file for faster recommendations. Uncheck to use the DSSM model.")

    # Analyze Button
    analyze_button = st.button("Analyze & Recommend Courses", use_container_width=True)

# --- Main Content Area ---
if analyze_button and resume_file and job_title_input:
    # Load models and client first for validation
    client = get_chroma_client()
    embedding_model = get_embedding_model()
    
    if not client or not embedding_model:
        st.error("Could not load necessary models. Please check the console for errors.")
        st.stop()
    
    # Validate job title using actual database search
    is_valid, validation_message = validate_job_search(job_title_input, client, embedding_model)
    
    if not is_valid:
        st.error("❌ *Job Title Validation Failed*")
        st.warning(validation_message)
        st.markdown("""
        *Try these examples:*
        - Data Scientist, Machine Learning Engineer
        - Software Developer, Full Stack Engineer  
        - DevOps Engineer, Cloud Architect
        - Product Manager, Business Analyst
        - UX Designer, Frontend Developer
        """)
        st.stop()
    
    # 1. Process Resume
    st.header("📄 Your Resume Analysis")
    resume_text = ""
    
    with st.spinner("Extracting text from your resume..."):
        if resume_file.type == "application/pdf":
            if use_ocr:
                resume_text = extract_text_from_pdf_with_ocr(resume_file)
                if resume_text:
                    st.success("✅ Text extracted using OCR for better accuracy")
            else:
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    st.success("✅ Text extracted from PDF")
                    
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)
            if resume_text:
                st.success("✅ Text extracted from DOCX")
                
        elif resume_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            if use_ocr:
                resume_text = extract_text_from_image(resume_file)
                if resume_text:
                    st.success("✅ Text extracted from image using OCR")
            else:
                st.warning("Please enable OCR to extract text from images")
                st.stop()

    if resume_text:
        # Show extracted text for debugging
        with st.expander("🔍 Debug: Extracted Text (First 500 chars)"):
            st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        
        user_skills = extract_skills_from_resume(resume_text)
        
        if user_skills:
            st.subheader("Your Extracted Skills:")
            # Display skills in numbered list format
            skills_text = ""
            for i, skill in enumerate(user_skills, 1):
                skills_text += f"{i}. {skill}\n"
            st.info(skills_text)
        else:
            st.warning("Could not extract any skills from your resume. Please ensure it contains relevant technical and professional skills.")

    else:
        st.error("Failed to read the resume file. Please try a different file.")

    # Placeholder for next steps
    if resume_text and user_skills:
        st.header("🔍 Job & Skill Gap Analysis")
        
        # Load DSSM model (client and embedding_model already loaded)
        dssm_model = load_dssm_model()

        if dssm_model:
            with st.spinner(f"Searching for '{job_title_input}' in our database..."):
                target_job_meta, job_desc = find_target_job(job_title_input, client, embedding_model)
            
            if target_job_meta:
                # Additional check for job relevance
                job_title_found = target_job_meta.get('Job Title', target_job_meta.get('job_title', ''))
                job_text = target_job_meta.get('job_text', '')
                
                # Check if the found job is actually relevant
                job_title_lower = job_title_input.lower()
                found_job_lower = job_title_found.lower()
                
                # Define clearly irrelevant terms
                irrelevant_terms = ['burger', 'pizza', 'food', 'restaurant', 'cooking', 'culinary', 'chef', 'waiter', 'server', 'cashier', 'janitor', 'cleaner', 'driver', 'delivery']
                
                # Check if the search term is clearly irrelevant
                if any(term in job_title_lower for term in irrelevant_terms):
                    st.error("❌ *Invalid Job Title*")
                    st.warning(f"'{job_title_input}' is not a professional job title. Please enter a valid job title like:")
                    st.markdown("""
                    - *Data Scientist* or *Data Analyst*
                    - *Software Engineer* or *Developer*
                    - *Product Manager* or *Business Analyst*
                    - *DevOps Engineer* or *Cloud Architect*
                    """)
                    st.stop()
                
                # Check if the found job is relevant
                relevant_keywords = ['analyst', 'data', 'scientist', 'engineer', 'developer', 'manager', 'specialist', 'consultant', 'architect', 'designer', 'administrator', 'coordinator', 'director', 'lead']
                
                job_is_relevant = (
                    any(keyword in job_title_lower for keyword in relevant_keywords) or
                    any(keyword in found_job_lower for keyword in relevant_keywords) or
                    'data' in job_title_lower or 'data' in found_job_lower
                )
                
                if not job_is_relevant:
                    st.error("❌ *No Relevant Jobs Found*")
                    st.warning(f"'{job_title_input}' doesn't match any professional jobs in our database. Please try a different job title.")
                    st.stop()
                
                st.subheader(f"Best Match Found: {job_title_found}")
                
                # --- DEBUGGING: Show full metadata ---
                with st.expander("🔍 Debug: Full Job Metadata"):
                    st.json(target_job_meta)
                # --- END DEBUGGING ---
                
                with st.expander("View Job Description"):
                    st.write(job_desc)

                required_skills = get_skills_from_job_metadata(target_job_meta)
                
                # --- DEBUGGING: Show skill extraction process ---
                st.write(f"*Debug: Extracted skills count: {len(required_skills)}*")
                if required_skills:
                    st.write(f"*Debug: Skills found: {required_skills}*")
                else:
                    st.write("*Debug: No skills extracted - checking metadata structure...*")
                    metadata_lower = {k.lower(): v for k, v in target_job_meta.items()}
                    st.write(f"*Debug: Available keys (lowercase): {list(metadata_lower.keys())}*")
                    
                    # Check for job_text field specifically
                    job_text = metadata_lower.get("job_text", "")
                    if job_text:
                        st.write(f"*Debug: job_text field found (first 200 chars): {str(job_text)[:200]}...*")
                    else:
                        st.write("*Debug: No job_text field found*")
                # --- END DEBUGGING ---
                
                if required_skills:
                    st.subheader("Required Skills for this Role:")
                    st.info(", ".join(required_skills))

                    # Calculate skill gap
                    skill_gap = [skill for skill in required_skills if skill not in user_skills]

                    if not skill_gap:
                        st.balloons()
                        st.success("🎉 *Congratulations!* Your skills are a great match for this role. No immediate skill gap found.")
                    else:
                        st.subheader("Your Skill Gap:")
                        st.warning(", ".join(skill_gap))

                        st.header("📚 Recommended Courses to Bridge the Gap")
                        
                        # Get job-specific recommendations instead of skill-specific ones
                        job_title = target_job_meta.get('job_title', target_job_meta.get('Job Title', job_title_input))
                        
                        if use_json_mapping:
                            json_mapping = load_job_course_mapping()
                            if json_mapping:
                                # Find this specific job in the JSON mapping
                                job_recommendations = []
                                for job_entry in json_mapping:
                                    if job_entry.get('job_title', '').lower() == job_title.lower():
                                        job_recommendations = job_entry.get('top_courses', [])
                                        break
                                
                                if job_recommendations:
                                    st.success("✅ *Most Relevant Courses for This Role:*")
                                    for i, course_info in enumerate(job_recommendations[:5], 1):  # Show top 5
                                        course_title = course_info.get('title', 'Unknown Course')
                                        organization = course_info.get('organization', 'Coursera')
                                        similarity = course_info.get('similarity', 0.0)
                                        st.markdown(f"{i}. *{course_title}* by {organization}")
                                        st.markdown(f"   📊 Relevance: {similarity:.3f} 🔄 JSON")
                                        st.markdown("---")
                                else:
                                    st.info("No specific course recommendations found for this role.")
                            else:
                                st.error("Could not load course recommendations.")
                        else:
                            # Fallback to skill-based recommendations
                            recommendations, error = find_course_recommendations(skill_gap, dssm_model, client, embedding_model, use_json_mapping=False)
                            
                            if error:
                                st.error(error)
                            elif recommendations:
                                st.success("✅ *Recommended Courses for Missing Skills:*")
                                for skill, recs in recommendations.items():
                                    st.subheader(f"Courses for: {skill}")
                                    for i, rec in enumerate(recs[:3], 1):  # Show top 3 per skill
                                        source_badge = "🤖 DSSM"
                                        st.markdown(f"{i}. *{rec['title']}* by {rec['organization']}")
                                        st.markdown(f"   📊 Similarity: {rec['similarity']:.3f} {source_badge}")
                                        st.markdown("---")
                            else:
                                st.info("No specific course recommendations found for the identified skill gap.")
                            
                            # Fallback: Show general course recommendations for this job from JSON mapping
                            if use_json_mapping:
                                json_mapping = load_job_course_mapping()
                                if json_mapping:
                                    # Find this job in JSON mapping
                                    job_title = target_job_meta.get('job_title', target_job_meta.get('Job Title', ''))
                                    if not job_title and 'job_text' in target_job_meta:
                                        job_text = target_job_meta['job_text']
                                        if 'Job Title:' in job_text:
                                            job_title = job_text.split('Job Title:')[1].split('.')[0].strip()
                                    
                                    for job_entry in json_mapping:
                                        if job_entry.get('job_title', '').lower() == job_title.lower():
                                            top_courses = job_entry.get('top_courses', [])
                                            if top_courses:
                                                st.subheader("📚 General Course Recommendations for this Role:")
                                                for i, course_info in enumerate(top_courses[:3], 1):
                                                    course_title = course_info.get('title', 'Unknown Course')
                                                    organization = course_info.get('organization', 'Coursera')
                                                    similarity = course_info.get('similarity', 0.0)
                                                    st.markdown(f"{i}. *{course_title}* by {organization} (Similarity: {similarity:.3f})")
                                            break

                else:
                    st.warning("Could not determine the required skills for this job title.")
            else:
                st.error("Could not find a matching job in the database. Please try a different title.")
        else:
            st.error("Could not load necessary models or connect to the database. Please check the console for errors.")

elif analyze_button:
    st.warning("Please upload your resume and enter a job title to begin.")

st.markdown("---")
st.markdown("Powered by DSSM and Streamlit")
