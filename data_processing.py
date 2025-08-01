import pandas as pd
from pathlib import Path
import numpy as np
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer
import os
# --- Import improved normalization from utils.py ---
from utils import normalize_skill_name

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# --- Define Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
JOB_DATASET_DIR = BASE_DIR / "job_dataset"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
PREPROCESSED_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB and model for semantic matching
try:
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Get collections
    course_skill_collection = chroma_client.get_collection("course_skills")
    print("✓ ChromaDB and model initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Could not initialize ChromaDB/model: {e}")
    print("  Semantic matching will be disabled, falling back to exact matching")
    model = None
    course_skill_collection = None

JOB_CSVS = [
    "naukri_com_job_sample.csv",
    "NYC_Fresh_Jobs_Postings.csv",
    "Data_Science_Analytics.csv",
    "Engineering_Hardware_Networks.csv",
    "Engineering_Software_QA.csv",
    "IT_Information_Security.csv",
    "Project_Program_Management.csv",
    "Product_Management.csv",
    "Research _ Development.csv",
    "UX Design _ Architecture.csv",
    "postings.csv"
]

# Initialize NLTK components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Enhanced skill synonyms mapping
SKILL_SYNONYMS = {
    # Programming Languages
    'python': ['python3', 'python programming', 'py', 'pythonic'],
    'javascript': ['js', 'javascript programming', 'ecmascript'],
    'java': ['java programming', 'j2ee', 'j2se'],
    'c++': ['cpp', 'c plus plus', 'c++ programming'],
    'c#': ['csharp', 'c sharp', 'c# programming'],
    'sql': ['structured query language', 'mysql', 'postgresql', 'oracle sql'],
    'r': ['r programming', 'r language', 'r stats'],
    
    # Data Science & ML
    'machine learning': ['ml', 'machine learning algorithms', 'ml algorithms'],
    'deep learning': ['dl', 'neural networks', 'neural network', 'ai'],
    'artificial intelligence': ['ai', 'artificial intelligence algorithms'],
    'data science': ['data analytics', 'data analysis', 'analytics'],
    'statistics': ['statistical analysis', 'stats', 'statistical modeling'],
    'data visualization': ['visualization', 'data viz', 'charts', 'graphs'],
    
    # Web Technologies
    'html': ['html5', 'hypertext markup language'],
    'css': ['css3', 'cascading style sheets', 'styling'],
    'react': ['reactjs', 'react.js', 'react javascript'],
    'angular': ['angularjs', 'angular.js', 'angular javascript'],
    'node.js': ['nodejs', 'node', 'node javascript'],
    
    # Cloud & DevOps
    'aws': ['amazon web services', 'amazon aws', 'aws cloud'],
    'azure': ['microsoft azure', 'azure cloud'],
    'docker': ['containerization', 'containers', 'docker containers'],
    'kubernetes': ['k8s', 'kubernetes orchestration'],
    'git': ['version control', 'git version control'],
    
    # Databases
    'mongodb': ['mongo', 'nosql', 'document database'],
    'redis': ['redis cache', 'caching'],
    'elasticsearch': ['elastic search', 'search engine'],
    
    # Frameworks & Libraries
    'pandas': ['python pandas', 'data manipulation'],
    'numpy': ['numerical python', 'numerical computing'],
    'scikit-learn': ['sklearn', 'scikit learn', 'machine learning library'],
    'tensorflow': ['tf', 'tensor flow', 'deep learning framework'],
    'pytorch': ['torch', 'py torch', 'deep learning library'],
    
    # Soft Skills
    'leadership': ['team leadership', 'leadership skills', 'managing teams'],
    'communication': ['communication skills', 'verbal communication', 'written communication'],
    'problem solving': ['problem-solving', 'analytical thinking', 'critical thinking'],
    'project management': ['project planning', 'agile', 'scrum', 'kanban'],
    'teamwork': ['collaboration', 'team collaboration', 'working in teams'],
    
    # Business & Domain
    'business analysis': ['business analyst', 'requirements gathering', 'business requirements'],
    'product management': ['product owner', 'product strategy', 'product development'],
    'marketing': ['digital marketing', 'marketing strategy', 'brand management'],
    'finance': ['financial analysis', 'accounting', 'financial modeling'],
    'sales': ['sales management', 'business development', 'client relations'],
    
    # Tools & Platforms
    'excel': ['microsoft excel', 'spreadsheets', 'data analysis'],
    'power bi': ['powerbi', 'business intelligence', 'data visualization'],
    'tableau': ['tableau desktop', 'data visualization tool'],
    'jira': ['project tracking', 'agile tools', 'issue tracking'],
    'slack': ['team communication', 'collaboration tools'],
    
    # Methodologies
    'agile': ['agile methodology', 'scrum', 'kanban', 'lean'],
    'scrum': ['agile scrum', 'scrum methodology', 'sprint planning'],
    'waterfall': ['traditional methodology', 'sequential development'],
    'devops': ['development operations', 'ci/cd', 'continuous integration'],
    
    # Security
    'cybersecurity': ['information security', 'cyber security', 'security'],
    'penetration testing': ['pen testing', 'security testing', 'vulnerability assessment'],
    'encryption': ['data encryption', 'cryptography', 'security protocols'],
    
    # Mobile Development
    'ios': ['iphone development', 'swift', 'objective-c'],
    'android': ['android development', 'kotlin', 'java android'],
    'react native': ['react native development', 'mobile app development'],
    
    # Testing
    'unit testing': ['unit tests', 'test driven development', 'tdd'],
    'integration testing': ['system testing', 'end to end testing', 'e2e testing'],
    'manual testing': ['qa testing', 'quality assurance', 'test cases'],
    'automated testing': ['test automation', 'selenium', 'automated qa']
}

# Generic words to remove from skills
GENERIC_WORDS = {
    'skill', 'skills', 'knowledge', 'experience', 'expertise', 'proficiency',
    'ability', 'capability', 'competency', 'competencies', 'understanding',
    'familiarity', 'familiar', 'basic', 'intermediate', 'advanced', 'expert',
    'level', 'years', 'year', 'plus', 'good', 'strong', 'excellent',
    'working', 'hands-on', 'hands on', 'practical', 'theoretical',
    'development', 'design', 'analysis', 'management', 'administration',
    'programming', 'coding', 'software', 'hardware', 'system', 'systems',
    'tool', 'tools', 'technology', 'technologies', 'framework', 'frameworks',
    'language', 'languages', 'platform', 'platforms', 'database', 'databases',
    'application', 'applications', 'web', 'mobile', 'desktop', 'cloud',
    'data', 'information', 'content', 'user', 'client', 'server', 'api',
    'interface', 'architecture', 'infrastructure', 'network', 'networking',
    'security', 'testing', 'deployment', 'maintenance', 'support', 'training'
}

def create_skill_mapping(skills_list):
    """
    Create a mapping from normalized skills to their original variants.
    """
    skill_mapping = defaultdict(list)
    
    for skill in skills_list:
        if isinstance(skill, str) and skill.strip():
            normalized = normalize_skill_name(skill)
            if normalized:
                skill_mapping[normalized].append(skill.strip())
    
    return skill_mapping

def find_duplicate_skills(skills_list):
    """
    Find duplicate skills after normalization.
    """
    normalized_skills = {}
    duplicates = defaultdict(list)
    
    for skill in skills_list:
        if isinstance(skill, str) and skill.strip():
            normalized = normalize_skill_name(skill)
            if normalized:
                if normalized in normalized_skills:
                    duplicates[normalized].append(skill.strip())
                else:
                    normalized_skills[normalized] = skill.strip()
    
    return duplicates

def clean_skills_column_advanced(df, skill_col):
    """
    Advanced skills column cleaning with normalization and duplicate removal.
    """
    if skill_col not in df.columns:
        return df
    
    print(f"  Cleaning skills column: {skill_col}")
    
    # Track original skills for analysis
    all_original_skills = []
    for skills_str in df[skill_col].dropna():
        if isinstance(skills_str, str) and skills_str != 'No skills specified':
            skills = [s.strip() for s in skills_str.split(',')]
            all_original_skills.extend(skills)
    
    print(f"    Original unique skills: {len(set(all_original_skills))}")
    
    # Find duplicates before normalization
    duplicates = find_duplicate_skills(all_original_skills)
    if duplicates:
        print(f"    Found {len(duplicates)} skill groups with duplicates:")
        for canonical, variants in list(duplicates.items())[:5]:  # Show first 5
            print(f"      '{canonical}': {variants}")
    
    # Process each row
    cleaned_skills = []
    for skills_str in df[skill_col]:
        if pd.isna(skills_str) or skills_str == 'No skills specified':
            cleaned_skills.append('No skills specified')
            continue
        
        if isinstance(skills_str, str):
            # Split and normalize each skill
            skills = [s.strip() for s in skills_str.split(',')]
            normalized_skills = []
            
            for skill in skills:
                if skill and skill != 'No skills specified':
                    normalized = normalize_skill_name(skill)
                    if normalized and normalized not in normalized_skills:
                        normalized_skills.append(normalized)
            
            # Join back together
            if normalized_skills:
                cleaned_skills.append(','.join(sorted(normalized_skills)))
            else:
                cleaned_skills.append('No skills specified')
        else:
            cleaned_skills.append('No skills specified')
    
    df[skill_col] = cleaned_skills
    
    # Track final skills for analysis
    all_final_skills = []
    for skills_str in df[skill_col].dropna():
        if isinstance(skills_str, str) and skills_str != 'No skills specified':
            skills = [s.strip() for s in skills_str.split(',')]
            all_final_skills.extend(skills)
    
    print(f"    Final unique skills: {len(set(all_final_skills))}")
    print(f"    Skills reduction: {len(set(all_original_skills)) - len(set(all_final_skills))} skills removed")
    
    return df

def load_csv_with_encoding(fpath):
    """Load CSV with multiple encoding attempts."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(fpath, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {fpath} with any encoding")

def standardize_columns(df):
    """Standardize column names to lowercase with underscores."""
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    return df

def handle_missing_values(df, dataset_name):
    """Handle missing values for job datasets with improved strategies."""
    print(f"Handling missing values for {dataset_name}...")
    
    # Track missing data before processing
    missing_before = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percent_before = (missing_before / total_cells) * 100
    
    print(f"  Missing data before: {missing_percent_before:.1f}% ({missing_before:,} cells)")

    # Enhanced fill defaults based on column patterns
    fill_defaults = {
        # Skills-related columns
        'skills': 'No skills specified',
        'preferred_skills': 'No skills specified',
        'job_skills': 'No skills specified',
        'required_skills': 'No skills specified',
        'skill': 'No skills specified',
        'Skill': 'No skills specified',
        
        # Company/Organization columns
        'company': 'Company not specified',
        'Company': 'Company not specified',
        'organization': 'Organization not specified',
        'Organization': 'Organization not specified',
        'employer': 'Employer not specified',
        'Employer': 'Employer not specified',
        'agency': 'Agency not specified',
        'Agency': 'Agency not specified',
        
        # Job title columns
        'jobtitle': 'Title not specified',
        'job_title': 'Title not specified',
        'business_title': 'Title not specified',
        'Business Title': 'Title not specified',
        'title': 'Title not specified',
        'Title': 'Title not specified',
        'position': 'Position not specified',
        'Position': 'Position not specified',
        
        # Description columns
        'jobdescription': 'No description available',
        'job_description': 'No description available',
        'description': 'No description available',
        'Description': 'No description available',
        
        # Experience and requirements
        'experience': 'Experience not specified',
        'Experience': 'Experience not specified',
        'exp': 'Experience not specified',
        'Exp': 'Experience not specified',
        
        # Location and industry
        'industry': 'Industry not specified',
        'Industry': 'Industry not specified',
        'joblocation_address': 'Location not specified',
        'location': 'Location not specified',
        'Location': 'Location not specified',
        
        # Salary and compensation
        'payrate': 'Not specified',
        'salary': 'Not specified',
        'Salary': 'Not specified',
        'salary_range_from': 'Not specified',
        'salary_range_to': 'Not specified',
        
        # Other metadata
        'postdate': 'Date not available',
        'post_date': 'Date not available',
        'numberofpositions': 1,
        'education': 'Not specified',
        'Education': 'Not specified',
        'level': 'Level not specified',
        'Level': 'Level not specified',
        'difficulty': 'Difficulty not specified',
        'Difficulty': 'Difficulty not specified'
    }
    
    # Apply fill defaults
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    # Handle numeric columns with median/mean imputation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Use median for skewed data, mean for normal distribution
                if df[col].skew() > 1 or df[col].skew() < -1:
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
    
    # Handle categorical columns with mode imputation
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
    
    # Drop rows with >70% missing values (more lenient than before)
    threshold = int(len(df.columns) * 0.7)
    df = df.dropna(thresh=threshold)
    
    # Track missing data after processing
    missing_after = df.isnull().sum().sum()
    total_cells_after = df.shape[0] * df.shape[1]
    missing_percent_after = (missing_after / total_cells_after) * 100 if total_cells_after > 0 else 0
    
    print(f"  Missing data after: {missing_percent_after:.1f}% ({missing_after:,} cells)")
    print(f"  Improvement: {missing_percent_before - missing_percent_after:.1f}% reduction")

    return df

def handle_course_missing_values(df, dataset_name):
    """Handle missing values for course datasets."""
    print(f"Handling missing values for course dataset {dataset_name}...")
    
    # Track missing data before processing
    missing_before = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percent_before = (missing_before / total_cells) * 100
    
    print(f"  Missing data before: {missing_percent_before:.1f}% ({missing_before:,} cells)")

    # Course-specific fill defaults
    course_fill_defaults = {
        # Course metadata
        'title': 'Title not available',
        'Title': 'Title not available',
        'course_title': 'Title not available',
        'Course Title': 'Title not available',
        'name': 'Name not available',
        'Name': 'Name not available',
        
        # Organization/Instructor
        'organization': 'Organization not specified',
        'Organization': 'Organization not specified',
        'instructor': 'Instructor not specified',
        'Instructor': 'Instructor not specified',
        'provider': 'Provider not specified',
        'Provider': 'Provider not specified',
        
        # Course details
        'description': 'No description available',
        'Description': 'No description available',
        'course_description': 'No description available',
        'summary': 'No summary available',
        'Summary': 'No summary available',
        
        # Skills and topics
        'skills': 'No skills specified',
        'Skills': 'No skills specified',
        'skill': 'No skills specified',
        'Skill': 'No skills specified',
        'topics': 'No topics specified',
        'Topics': 'No topics specified',
        
        # Course ratings and metrics
        'rating': 0.0,
        'Rating': 0.0,
        'ratings': 0.0,
        'Ratings': 0.0,
        'enrolled': 0,
        'Enrolled': 0,
        'enrollment': 0,
        'Enrollment': 0,
        'num_review': 0,
        'reviews': 0,
        'Reviews': 0,
        
        # Course level and difficulty
        'level': 'Level not specified',
        'Level': 'Level not specified',
        'difficulty': 'Difficulty not specified',
        'Difficulty': 'Difficulty not specified',
        
        # Course duration and format
        'duration': 'Duration not specified',
        'Duration': 'Duration not specified',
        'length': 'Length not specified',
        'Length': 'Length not specified',
        'format': 'Format not specified',
        'Format': 'Format not specified',
        
        # Pricing and availability
        'price': 'Price not specified',
        'Price': 'Price not specified',
        'cost': 'Cost not specified',
        'Cost': 'Cost not specified',
        'free': 'Free',
        'Free': 'Free',
        
        # Satisfaction and completion
        'satisfaction_rate': 0.0,
        'Satisfaction Rate': 0.0,
        'completion_rate': 0.0,
        'Completion Rate': 0.0
    }
    
    # Apply course-specific fill defaults
    for col, val in course_fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    # Handle numeric columns with appropriate imputation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            if 'rating' in col.lower() or 'satisfaction' in col.lower():
                # Ratings should be 0 if missing
                df[col] = df[col].fillna(0.0)
            elif 'enrolled' in col.lower() or 'enrollment' in col.lower() or 'review' in col.lower():
                # Counts should be 0 if missing
                df[col] = df[col].fillna(0)
            else:
                # Use median for other numeric columns
                df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns with mode imputation
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
    
    # Drop rows with >80% missing values (more lenient for courses)
    threshold = int(len(df.columns) * 0.8)
    df = df.dropna(thresh=threshold)
    
    # Track missing data after processing
    missing_after = df.isnull().sum().sum()
    total_cells_after = df.shape[0] * df.shape[1]
    missing_percent_after = (missing_after / total_cells_after) * 100 if total_cells_after > 0 else 0
    
    print(f"  Missing data after: {missing_percent_after:.1f}% ({missing_after:,} cells)")
    print(f"  Improvement: {missing_percent_before - missing_percent_after:.1f}% reduction")

    return df

def preprocess_and_save_job(filename):
    """Preprocess and save a single job dataset."""
    fpath = JOB_DATASET_DIR / filename
    key = filename.replace('.csv', '').replace(' ', '_').replace('-', '_').lower()

    try:
        print(f"Processing {filename}...")
        df = load_csv_with_encoding(fpath)
        print(f"  Original shape: {df.shape}")

        # Standardize column names
        df = standardize_columns(df)

        # Remove duplicates
        df = df.drop_duplicates()
        print(f"  After removing duplicates: {df.shape}")

        # Handle missing values
        df = handle_missing_values(df, key)

        # Clean skills columns
        skill_columns = ['skills', 'preferred_skills', 'job_skills', 'required_skills']
        for col in skill_columns:
            if col in df.columns:
                df = clean_skills_column_advanced(df, col)

        # Create combined text for embedding
        text_columns = []
        if 'company' in df.columns:
            text_columns.append('company')
        if 'jobtitle' in df.columns:
            text_columns.append('jobtitle')
        elif 'business_title' in df.columns:
            text_columns.append('business_title')
        if 'jobdescription' in df.columns:
            text_columns.append('jobdescription')
        elif 'job_description' in df.columns:
            text_columns.append('job_description')

        if text_columns:
            df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)

        # Final cleanup
        df = df.fillna('Unknown')  # Fill any remaining missing values

        # Save cleaned file
        out_path = PREPROCESSED_DIR / f"{key}_preprocessed.csv"
        df.to_csv(out_path, index=False)
        print(f"  Final shape: {df.shape}")
        print(f"✓ Preprocessed and saved: {out_path}")

        return df

    except Exception as e:
        print(f"✗ Error preprocessing {filename}: {e}")
        return None

def preprocess_and_save_course(filename):
    """Preprocess and save a single course dataset."""
    fpath = JOB_DATASET_DIR / filename
    key = filename.replace('.csv', '').replace(' ', '_').replace('-', '_').lower()

    try:
        print(f"Processing course dataset {filename}...")
        df = load_csv_with_encoding(fpath)
        print(f"  Original shape: {df.shape}")

        # Standardize column names
        df = standardize_columns(df)

        # Remove duplicates
        df = df.drop_duplicates()
        print(f"  After removing duplicates: {df.shape}")

        # Handle missing values for courses
        df = handle_course_missing_values(df, key)

        # --- Handle new Coursera course columns ---
        # Ensure all expected columns exist
        expected_cols = ['title', 'organization', 'skills', 'ratings', 'review_counts', 'metadata']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 'Not specified'

        # Clean skills columns
        skill_columns = ['skills', 'Skills', 'skill', 'Skill', 'topics', 'Topics']
        for col in skill_columns:
            if col in df.columns:
                df = clean_skills_column_advanced(df, col)

        # Create combined text for embedding
        text_columns = []
        if 'title' in df.columns:
            text_columns.append('title')
        elif 'course_title' in df.columns:
            text_columns.append('course_title')
        if 'description' in df.columns:
            text_columns.append('description')
        elif 'course_description' in df.columns:
            text_columns.append('course_description')
        if 'organization' in df.columns:
            text_columns.append('organization')
        elif 'instructor' in df.columns:
            text_columns.append('instructor')
        # Add ratings, review_counts, metadata if present
        if 'ratings' in df.columns:
            text_columns.append('ratings')
        if 'review_counts' in df.columns:
            text_columns.append('review_counts')
        if 'metadata' in df.columns:
            text_columns.append('metadata')

        if text_columns:
            df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)

        # Final cleanup
        df = df.fillna('Unknown')  # Fill any remaining missing values

        # Save cleaned file
        out_path = PREPROCESSED_DIR / f"{key}_preprocessed.csv"
        df.to_csv(out_path, index=False)
        print(f"  Final shape: {df.shape}")
        print(f"✓ Preprocessed and saved: {out_path}")

        return df

    except Exception as e:
        print(f"✗ Error preprocessing course dataset {filename}: {e}")
        return None

def main():
    """Main preprocessing function for both job and course datasets."""
    print("🚀 Starting Comprehensive Data Preprocessing...")

    processed_datasets = {}

    # Define course datasets
    COURSE_CSVS = [
        "coursera1.csv",
        "coursera2.csv",
        "coursera3.csv",
        "coursera_data_analyst.csv",
        "coursera_hardware.csv",
        "coursera_info_sec.csv",
        "coursera_product.csv",
        "coursera_project.csv",
        "coursera_software.csv",
        "coursera_ui_ux.csv",
        "coursera_cyber_security.csv"
    ]

    # Process job datasets
    print("\n📋 Processing Job Datasets...")
    for fname in JOB_CSVS:
        df = preprocess_and_save_job(fname)
        if df is not None:
            key = fname.replace('.csv', '').replace(' ', '_').replace('-', '_').lower()
            processed_datasets[f"job_{key}"] = df

    # Process course datasets
    print("\n📚 Processing Course Datasets...")
    for fname in COURSE_CSVS:
        df = preprocess_and_save_course(fname)
        if df is not None:
            key = fname.replace('.csv', '').replace(' ', '_').replace('-', '_').lower()
            processed_datasets[f"course_{key}"] = df

    print(f"\n✅ Comprehensive Data Preprocessing Complete. Processed {len(processed_datasets)} datasets.")

    # Print summary statistics
    print("\n📊 Preprocessing Summary:")
    job_count = 0
    course_count = 0
    for name, df in processed_datasets.items():
        if name.startswith('job_'):
            job_count += 1
        elif name.startswith('course_'):
            course_count += 1
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\n📈 Summary:")
    print(f"  Job datasets: {job_count}")
    print(f"  Course datasets: {course_count}")
    print(f"  Total datasets: {len(processed_datasets)}")

    # Analyze skill gaps
    if model is not None and course_skill_collection is not None:
        skill_analysis = analyze_skill_gaps_with_semantic_matching(processed_datasets)
    else:
        skill_analysis = analyze_skill_gaps(processed_datasets)

    return processed_datasets

def analyze_skill_gaps_with_semantic_matching(processed_datasets):
    """Analyze skill gaps using semantic similarity instead of exact matching."""
    print("\n🔍 Analyzing Skill Gaps with Semantic Matching...")
    
    # Check if ChromaDB and model are available
    if model is None or course_skill_collection is None:
        print("  ChromaDB/model not available, falling back to exact matching...")
        return analyze_skill_gaps(processed_datasets)
    
    # Extract all job skills (already normalized)
    job_skills = set()
    for name, df in processed_datasets.items():
        if name.startswith('job_'):
            skill_columns = ['skills', 'preferred_skills', 'job_skills', 'required_skills']
            for col in skill_columns:
                if col in df.columns:
                    for skills_str in df[col].dropna():
                        if isinstance(skills_str, str) and skills_str != 'No skills specified':
                            skills = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
                            for skill in skills:
                                norm_skill = normalize_skill_name(skill)
                                if norm_skill:
                                    job_skills.add(norm_skill)
    
    print(f"  Unique job skills for analysis: {len(job_skills)}")

    semantic_overlap = set()
    semantic_gaps = set()
    
    print("  Performing semantic skill matching...")
    for job_skill in job_skills:
        try:
            # Query ChromaDB for similar course skills
            job_skill_embedding = model.encode(job_skill)
            
            results = course_skill_collection.query(
                query_embeddings=[job_skill_embedding.tolist()],
                n_results=1,  # Get the top 1 most similar course skill
                include=["metadatas", "distances"]
            )
            
            if results and results['metadatas'] and results['metadatas'][0]:
                best_match_metadata = results['metadatas'][0][0]
                similarity = 1 - results['distances'][0][0] # Convert distance to similarity
                
                # Consider a match if similarity is above a certain threshold
                if similarity >= 0.7:  # Threshold for semantic match
                    semantic_overlap.add((job_skill, best_match_metadata['skill_name'], similarity))
                else:
                    semantic_gaps.add(job_skill)
            else:
                semantic_gaps.add(job_skill)
                
        except Exception as e:
            print(f"    Error processing skill '{job_skill}': {e}")
            semantic_gaps.add(job_skill)
    
    # Calculate statistics
    overlap_percentage = (len(semantic_overlap) / len(job_skills)) * 100 if job_skills else 0
    
    print(f"  Semantic skill overlap: {len(semantic_overlap)} ({overlap_percentage:.1f}% of job skills)")
    print(f"  Semantic skill gaps: {len(semantic_gaps)} skills not covered by courses")
    
    # Save semantic analysis
    import json
    semantic_analysis_path = PREPROCESSED_DIR / "semantic_skill_analysis.json"
    
    # Get course skills from ChromaDB (limit to avoid memory issues)
    try:
        course_skills_in_chroma = []
        peek_results = course_skill_collection.peek(10000)  # Peek up to 10k skills
        if peek_results and 'metadatas' in peek_results:
            course_skills_in_chroma = [m['skill_name'] for m in peek_results['metadatas'] if m]
    except Exception as e:
        print(f"    Warning: Could not retrieve course skills from ChromaDB: {e}")
        course_skills_in_chroma = []
    
    semantic_analysis = {
        'job_skills': list(job_skills),
        'course_skills_in_chroma': course_skills_in_chroma,
        'semantic_overlap': [(job, course, float(sim)) for job, course, sim in semantic_overlap],
        'semantic_gaps': list(semantic_gaps),
        'overlap_percentage': overlap_percentage
    }
    with open(semantic_analysis_path, 'w') as f:
        json.dump(semantic_analysis, f, indent=2)
    
    print(f"✓ Semantic skill analysis saved to: {semantic_analysis_path}")
    
    # Print top semantic matches
    if semantic_overlap:
        print(f"\n🔗 Top Semantic Skill Matches (first 10):")
        sorted_overlap = sorted(semantic_overlap, key=lambda x: x[2], reverse=True)
        for i, (job_skill, course_skill, similarity) in enumerate(sorted_overlap[:10], 1):
            print(f"  {i}. '{job_skill}' ↔ '{course_skill}' (similarity: {similarity:.3f})")
    
    # Print top skill gaps
    if semantic_gaps:
        print(f"\n📋 Top Skill Gaps (first 20):")
        for i, skill in enumerate(list(semantic_gaps)[:20], 1):
            print(f"  {i}. {skill}")
    
    return semantic_analysis

def analyze_skill_gaps(processed_datasets):
    """Fallback: Analyze skill gaps between job requirements and course offerings using exact matching and improved normalization."""
    print("\n🔍 Analyzing Skill Gaps (Exact Matching Fallback)...")
    # Extract job skills
    job_skills = set()
    for name, df in processed_datasets.items():
        if name.startswith('job_'):
            skill_columns = ['skills', 'preferred_skills', 'job_skills', 'required_skills']
            for col in skill_columns:
                if col in df.columns:
                    for skills_str in df[col].dropna():
                        if isinstance(skills_str, str) and skills_str != 'No skills specified':
                            skills = [skill.strip() for skill in skills_str.split(',')]
                            for skill in skills:
                                norm_skill = normalize_skill_name(skill)
                                if norm_skill:
                                    job_skills.add(norm_skill)
    # Extract course skills
    course_skills = set()
    for name, df in processed_datasets.items():
        if name.startswith('course_'):
            skill_columns = ['skills', 'skill', 'topics']
            for col in skill_columns:
                if col in df.columns:
                    for skills_str in df[col].dropna():
                        if isinstance(skills_str, str) and skills_str != 'No skills specified':
                            skills = [skill.strip() for skill in skills_str.split(',')]
                            for skill in skills:
                                norm_skill = normalize_skill_name(skill)
                                if norm_skill:
                                    course_skills.add(norm_skill)
    # Calculate overlap and gaps
    skill_overlap = job_skills & course_skills
    skill_gaps = job_skills - course_skills
    print(f"  Job skills found: {len(job_skills)}")
    print(f"  Course skills found: {len(course_skills)}")
    print(f"  Skill overlap: {len(skill_overlap)} ({len(skill_overlap)/len(job_skills)*100:.1f}% of job skills)")
    print(f"  Skill gaps: {len(skill_gaps)} skills not covered by courses")
    # Save skill analysis
    import json
    skill_analysis_path = PREPROCESSED_DIR / "skill_analysis.json"
    skill_analysis = {
        'job_skills': list(job_skills),
        'course_skills': list(course_skills),
        'skill_overlap': list(skill_overlap),
        'skill_gaps': list(skill_gaps),
        'overlap_percentage': len(skill_overlap) / len(job_skills) * 100 if job_skills else 0
    }
    with open(skill_analysis_path, 'w') as f:
        json.dump(skill_analysis, f, indent=2)
    print(f"✓ Exact skill analysis saved to: {skill_analysis_path}")
    # Print top skill gaps
    if skill_gaps:
        print(f"\n📋 Top Skill Gaps (first 20):")
        for i, skill in enumerate(list(skill_gaps)[:20], 1):
            print(f"  {i}. {skill}")
    return skill_analysis

if __name__ == "__main__":
    main()
