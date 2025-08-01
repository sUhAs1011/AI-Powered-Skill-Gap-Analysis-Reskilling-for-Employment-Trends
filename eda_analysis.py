import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- Define Paths ---
BASE_DIR = Path(__file__).resolve().parent
JOB_DATASET_DIR = BASE_DIR / "job_dataset"
COURSERA_CSV = JOB_DATASET_DIR / "coursera2.csv"
NAUKRI_CSV = JOB_DATASET_DIR / "naukri_com_job_sample.csv"
NYC_JOBS_CSV = JOB_DATASET_DIR / "NYC_Fresh_Jobs_Postings.csv"
COURSES_CSV = JOB_DATASET_DIR / "coursera3.csv"
COURSERA_2024_CSV = JOB_DATASET_DIR / "coursera1.csv"

# Add new dataset filenames (remove coursera_*.csv from here, will be loaded automatically)
ADDITIONAL_CSVS = [
    "Data_Science_Analytics.csv",
    "Engineering_Hardware_Networks.csv",
    "Engineering_Software_QA.csv",
    "IT_Information_Security.csv",
    "Project_Program_Management.csv",
    "Research _ Development.csv",
    "UX Design _ Architecture.csv",
    "postings.csv"
]

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_skill(skill):
    skill = skill.lower().strip()
    skill = skill.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(w) for w in skill.split() if w not in stop_words]
    return ' '.join(tokens)

import spacy
nlp = spacy.load('en_core_web_lg')  # use the large model for better vectors

def is_similar(skill1, skill2, threshold=0.85):
    doc1 = nlp(skill1)
    doc2 = nlp(skill2)
    return doc1.similarity(doc2) >= threshold

def get_skills_by_dataset(datasets):
    """Aggregate skills from all relevant datasets into a dictionary."""
    skills_by_dataset = {}

    # Combine skills from all course datasets
    all_course_skills = set()

    # Coursera v3
    if not datasets['coursera'].empty and 'Skills' in datasets['coursera'].columns:
        all_course_skills.update(
            preprocess_skill(skill)
            for skill in datasets['coursera']['Skills'].dropna().str.split(',').explode()
        )
        # Extract from description if Skills column is missing or incomplete
        if 'Description' in datasets['coursera'].columns:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            for desc in datasets['coursera']['Description'].dropna():
                for word in desc.split():
                    word = preprocess_skill(word)
                    if word and word not in ENGLISH_STOP_WORDS and len(word) > 2:
                        all_course_skills.add(word)

    # Coursera 2024
    if not datasets['coursera_2024'].empty and 'Skills' in datasets['coursera_2024'].columns:
        all_course_skills.update(
            preprocess_skill(skill)
            for skill in datasets['coursera_2024']['Skills'].dropna().str.split(',').explode()
        )
        if 'Description' in datasets['coursera_2024'].columns:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            for desc in datasets['coursera_2024']['Description'].dropna():
                for word in desc.split():
                    word = preprocess_skill(word)
                    if word and word not in ENGLISH_STOP_WORDS and len(word) > 2:
                        all_course_skills.add(word)

    # courses.csv
    if not datasets['courses'].empty and 'Skills' in datasets['courses'].columns:
        all_course_skills.update(
            preprocess_skill(skill)
            for skill in datasets['courses']['Skills'].dropna().str.split(',').explode()
        )
        if 'description' in datasets['courses'].columns:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            for desc in datasets['courses']['description'].dropna():
                for word in desc.split():
                    word = preprocess_skill(word)
                    if word and word not in ENGLISH_STOP_WORDS and len(word) > 2:
                        all_course_skills.add(word)

    # Add all new coursera_*.csv datasets
    for key, df in datasets.items():
        if key.startswith('coursera_') and key not in ['coursera', 'coursera_2024']:
            if not df.empty and 'Skills' in df.columns:
                all_course_skills.update(
                    preprocess_skill(skill)
                    for skill in df['Skills'].dropna().str.split(',').explode()
                )

    skills_by_dataset['Courses'] = set(filter(None, all_course_skills))

    # Naukri skills
    if not datasets['naukri'].empty and 'skills' in datasets['naukri'].columns:
        naukri_skills = [preprocess_skill(skill)
                         for skills_str in datasets['naukri']['skills'].dropna()
                         for skill in skills_str.split(',')]
        skills_by_dataset['Naukri'] = set(filter(None, naukri_skills))

    # NYC Jobs skills
    if not datasets['nyc_jobs'].empty and 'Preferred Skills' in datasets['nyc_jobs'].columns:
        nyc_skills = [preprocess_skill(skill)
                      for skills_str in datasets['nyc_jobs']['Preferred Skills'].dropna()
                      for skill in skills_str.split(';')]
        skills_by_dataset['NYC Jobs'] = set(filter(None, nyc_skills))

    return skills_by_dataset

def load_datasets():
    """Load all datasets and return them as a dictionary."""
    datasets = {}
    
    try:
        datasets['coursera'] = pd.read_csv(COURSERA_CSV)
        print(f"✓ Coursera data loaded: {datasets['coursera'].shape}")
    except Exception as e:
        print(f"✗ Error loading Coursera data: {e}")
        datasets['coursera'] = pd.DataFrame()
    
    try:
        datasets['naukri'] = pd.read_csv(NAUKRI_CSV)
        print(f"✓ Naukri data loaded: {datasets['naukri'].shape}")
    except Exception as e:
        print(f"✗ Error loading Naukri data: {e}")
        datasets['naukri'] = pd.DataFrame()
    
    try:
        datasets['nyc_jobs'] = pd.read_csv(NYC_JOBS_CSV)
        print(f"✓ NYC Jobs data loaded: {datasets['nyc_jobs'].shape}")
    except Exception as e:
        print(f"✗ Error loading NYC Jobs data: {e}")
        datasets['nyc_jobs'] = pd.DataFrame()
    
    try:
        datasets['courses'] = pd.read_csv(COURSES_CSV)
        print(f"✓ Courses data loaded: {datasets['courses'].shape}")
    except Exception as e:
        print(f"✗ Error loading Courses data: {e}")
        datasets['courses'] = pd.DataFrame()
    
    try:
        datasets['coursera_2024'] = pd.read_csv(COURSERA_2024_CSV, encoding='utf-8')
        print(f"✓ Coursera 2024 data loaded: {datasets['coursera_2024'].shape}")
    except UnicodeDecodeError:
        try:
            datasets['coursera_2024'] = pd.read_csv(COURSERA_2024_CSV, encoding='latin1')
            print(f"✓ Coursera 2024 data loaded with latin1 encoding: {datasets['coursera_2024'].shape}")
        except Exception as e:
            print(f"✗ Error loading Coursera 2024 data with latin1: {e}")
            datasets['coursera_2024'] = pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading Coursera 2024 data: {e}")
        datasets['coursera_2024'] = pd.DataFrame()
    
    # Load additional datasets
    for fname in ADDITIONAL_CSVS:
        key = fname.replace('.csv', '').replace(' ', '_').replace('-', '_').lower()
        fpath = JOB_DATASET_DIR / fname
        try:
            datasets[key] = pd.read_csv(fpath, encoding='utf-8')
            print(f"✓ {fname} loaded: {datasets[key].shape}")
        except UnicodeDecodeError:
            try:
                datasets[key] = pd.read_csv(fpath, encoding='latin1')
                print(f"✓ {fname} loaded with latin1 encoding: {datasets[key].shape}")
            except Exception as e:
                print(f"✗ Error loading {fname} with latin1: {e}")
                datasets[key] = pd.DataFrame()
        except Exception as e:
            print(f"✗ Error loading {fname}: {e}")
            datasets[key] = pd.DataFrame()
    
    # Automatically load all new coursera_*.csv course datasets
    for f in JOB_DATASET_DIR.glob('coursera_*.csv'):
        key = f.stem
        if key not in datasets:
            try:
                datasets[key] = pd.read_csv(f, encoding='utf-8')
                print(f"✓ {f.name} loaded: {datasets[key].shape}")
            except Exception as e:
                print(f"✗ Error loading {f.name}: {e}")
                datasets[key] = pd.DataFrame()
    return datasets

def basic_dataset_info(datasets):
    """Display basic information about each dataset."""
    print("\n" + "="*60)
    print("BASIC DATASET INFORMATION")
    print("="*60)
    
    for name, df in datasets.items():
        if not df.empty:
            print(f"\n📊 {name.upper()} DATASET:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"   Data types:")
            for col, dtype in df.dtypes.items():
                print(f"     {col}: {dtype}")

def missing_values_analysis(datasets):
    """Analyze missing values in all datasets."""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    for name, df in datasets.items():
        if not df.empty:
            print(f"\n🔍 {name.upper()} - Missing Values:")
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing_Count': missing_data.values,
                'Missing_Percent': missing_percent.values
            }).sort_values('Missing_Percent', ascending=False)
            
            # Only show columns with missing values
            missing_df = missing_df[missing_df['Missing_Percent'] > 0]
            
            if len(missing_df) > 0:
                print(missing_df.to_string(index=False))
            else:
                print("   ✓ No missing values found!")

def coursera_eda(df):
    """Perform detailed EDA on Coursera dataset."""
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("COURSERA DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Total courses: {len(df)}")
    print(f"   Unique organizations: {df['Organization'].nunique()}")
    print(f"   Average rating: {df['Ratings'].mean():.2f}")
    print(f"   Rating range: {df['Ratings'].min():.1f} - {df['Ratings'].max():.1f}")
    
    # Difficulty distribution
    if 'Difficulty' in df.columns:
        print(f"\n📊 Difficulty Distribution:")
        difficulty_counts = df['Difficulty'].value_counts()
        print(difficulty_counts)
        
        # Plot difficulty distribution
        plt.figure(figsize=(10, 6))
        difficulty_counts.plot(kind='bar')
        plt.title('Course Difficulty Distribution')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Number of Courses')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("coursera_difficulty_distribution.png")
        plt.close()
    
    # Top organizations
    print(f"\n🏢 Top 10 Organizations by Course Count:")
    top_orgs = df['Organization'].value_counts().head(10)
    print(top_orgs)
    
    # Skills analysis
    if 'Skills' in df.columns:
        print(f"\n💡 Skills Analysis:")
        # Extract skills from the Skills column
        all_skills = []
        for skills_str in df['Skills'].dropna():
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"   Total unique skills: {len(skill_counts)}")
            print(f"   Most common skills:")
            for skill, count in skill_counts.most_common(10):
                print(f"     {skill}: {count}")

def naukri_eda(df):
    """Perform detailed EDA on Naukri dataset."""
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("NAUKRI DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Total job postings: {len(df)}")
    print(f"   Unique companies: {df['company'].nunique()}")
    print(f"   Unique job titles: {df['jobtitle'].nunique()}")
    
    # Top companies
    print(f"\n🏢 Top 10 Companies by Job Postings:")
    top_companies = df['company'].value_counts().head(10)
    print(top_companies)
    
    # Job titles analysis
    print(f"\n💼 Most Common Job Titles:")
    job_title_counts = df['jobtitle'].value_counts().head(10)
    print(job_title_counts)
    
    # Skills analysis
    if 'skills' in df.columns:
        print(f"\n💡 Skills Analysis:")
        all_skills = []
        for skills_str in df['skills'].dropna():
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"   Total unique skills: {len(skill_counts)}")
            print(f"   Most common skills:")
            for skill, count in skill_counts.most_common(10):
                print(f"     {skill}: {count}")
    
    # Experience requirements
    if 'experience' in df.columns:
        print(f"\n⏰ Experience Requirements:")
        exp_counts = df['experience'].value_counts().head(10)
        print(exp_counts)

def nyc_jobs_eda(df):
    """Perform detailed EDA on NYC Jobs dataset."""
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("NYC JOBS DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Total job postings: {len(df)}")
    print(f"   Unique agencies: {df['Agency'].nunique()}")
    print(f"   Unique business titles: {df['Business Title'].nunique()}")
    
    # Top agencies
    print(f"\n🏛️ Top 10 Agencies by Job Postings:")
    top_agencies = df['Agency'].value_counts().head(10)
    print(top_agencies)
    
    # Business titles analysis
    print(f"\n💼 Most Common Business Titles:")
    title_counts = df['Business Title'].value_counts().head(10)
    print(title_counts)
    
    # Salary analysis (if available)
    if 'Salary Range From' in df.columns and 'Salary Range To' in df.columns:
        print(f"\n💰 Salary Analysis:")
        df['Salary Range From'] = pd.to_numeric(df['Salary Range From'], errors='coerce')
        df['Salary Range To'] = pd.to_numeric(df['Salary Range To'], errors='coerce')
        
        print(f"   Average salary range: ${df['Salary Range From'].mean():.0f} - ${df['Salary Range To'].mean():.0f}")
        print(f"   Median salary range: ${df['Salary Range From'].median():.0f} - ${df['Salary Range To'].median():.0f}")
    
    # Preferred skills analysis
    if 'Preferred Skills' in df.columns:
        print(f"\n💡 Preferred Skills Analysis:")
        all_skills = []
        for skills_str in df['Preferred Skills'].dropna():
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(';')]
                all_skills.extend(skills)
        
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"   Total unique skills: {len(skill_counts)}")
            print(f"   Most common skills:")
            for skill, count in skill_counts.most_common(10):
                print(f"     {skill}: {count}")

def courses_eda(df):
    """Perform detailed EDA on Courses dataset."""
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("COURSES DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Total courses: {len(df)}")
    print(f"   Unique titles: {df['title'].nunique()}")
    
    # Title analysis
    print(f"\n📚 Course Title Analysis:")
    print(f"   Average title length: {df['title'].str.len().mean():.1f} characters")
    print(f"   Longest title: {df['title'].str.len().max()} characters")
    print(f"   Shortest title: {df['title'].str.len().min()} characters")
    
    # Description analysis (if available)
    if 'description' in df.columns:
        print(f"\n📝 Description Analysis:")
        df['desc_length'] = df['description'].str.len()
        print(f"   Average description length: {df['desc_length'].mean():.1f} characters")
        print(f"   Longest description: {df['desc_length'].max()} characters")
        print(f"   Shortest description: {df['desc_length'].min()} characters")

def cross_dataset_analysis(datasets):
    """Perform analysis across multiple datasets."""
    print("\n" + "="*60)
    print("CROSS-DATASET ANALYSIS")
    print("="*60)
    
    # Skills comparison across datasets
    print(f"\n🔍 Skills Comparison Across Datasets:")
    
    skills_by_dataset = get_skills_by_dataset(datasets)
    
    # Print skills statistics
    for dataset_name, skills in skills_by_dataset.items():
        print(f"   {dataset_name}: {len(skills)} unique skills")
    
    # Find common skills across datasets
    if len(skills_by_dataset) > 1:
        common_skills = set.intersection(*skills_by_dataset.values())
        print(f"\n   Common skills across all datasets: {len(common_skills)}")
        if common_skills:
            print(f"   Top common skills: {list(common_skills)[:10]}")

def generate_insights_report(datasets):
    """Generate a summary insights report."""
    print("\n" + "="*60)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*60)

    skills_by_dataset = get_skills_by_dataset(datasets)

    insights = []
    recommendations = []

    # Data quality insights
    total_records = sum(len(df) for df in datasets.values() if not df.empty)
    insights.append(f"📊 Total records across all datasets: {total_records:,}")

    # Missing data insights
    for name, df in datasets.items():
        if not df.empty:
            missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            insights.append(f"⚠️  {name.title()} missing data: {missing_percent:.1f}%")
            if missing_percent > 5:
                recommendations.append(
                    f"Consider addressing missing data in the {name.title()} dataset (currently {missing_percent:.1f}% missing)."
                )

    # Skills overlap and gaps
    if 'Courses' in skills_by_dataset and 'Naukri' in skills_by_dataset:
        overlap = skills_by_dataset['Courses'] & skills_by_dataset['Naukri']
        gap = skills_by_dataset['Naukri'] - skills_by_dataset['Courses']
        overlap_percent = (len(overlap) / len(skills_by_dataset['Naukri'])) * 100 if skills_by_dataset['Naukri'] else 0
        insights.append(f"🔗 Skill overlap between all course datasets and Naukri: {len(overlap)} skills ({overlap_percent:.1f}%)")
        if overlap_percent < 30:
            recommendations.append(
                f"Expand course offerings to cover more in-demand job skills. Only {overlap_percent:.1f}% of job-required skills are covered by current courses."
            )
        if gap:
            recommendations.append(
                f"Develop new courses for {len(gap)} job-required skills not currently covered by any course dataset."
            )
            # Print missing skills for user review
            print("\nJob-required skills not covered by any course (first 50):")
            for skill in list(gap)[:50]:
                print(f"  - {skill}")

    # Skill naming consistency
    if 'Courses' in skills_by_dataset:
        lower_skills = [s.lower() for s in skills_by_dataset['Courses']]
        if len(lower_skills) != len(set(lower_skills)):
            recommendations.append(
                "Standardize skill naming conventions in the course datasets to avoid duplicates (e.g., 'python' vs 'Python')."
            )

    # Print insights and recommendations
    for insight in insights:
        print(insight)

    if recommendations:
        print("\n🎯 DATA-DRIVEN RECOMMENDATIONS:")
        for idx, rec in enumerate(recommendations, 1):
            print(f"{idx}. {rec}")
    else:
        print("\n🎯 No critical recommendations based on current data.")

def coursera_2024_eda(df):
    """Perform detailed EDA on Coursera 2024 dataset."""
    if df.empty:
        return

    print("\n" + "="*60)
    print("COURSERA 2024 DATASET ANALYSIS")
    print("="*60)

    # Convert relevant columns to numeric
    for col in ['rating', 'enrolled', 'num_review', 'Satisfaction Rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Total courses: {len(df)}")
    print(f"   Unique organizations: {df['Organization'].nunique()}")
    print(f"   Average rating: {df['rating'].mean():.2f}")
    print(f"   Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
    print(f"   Average satisfaction rate: {df['Satisfaction Rate'].mean():.2f}")

    # Enrollment stats
    if 'enrolled' in df.columns:
        print(f"   Average enrolled: {df['enrolled'].mean():.0f}")
        print(f"   Max enrolled: {df['enrolled'].max():.0f}")

    # Level distribution
    if 'Level' in df.columns:
        print(f"\n📊 Level Distribution:")
        level_counts = df['Level'].value_counts()
        print(level_counts)
        plt.figure(figsize=(10, 6))
        level_counts.plot(kind='bar')
        plt.title('Course Level Distribution')
        plt.xlabel('Level')
        plt.ylabel('Number of Courses')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("coursera2024_level_distribution.png")
        plt.close()

    # Top organizations
    print(f"\n🏢 Top 10 Organizations by Course Count:")
    top_orgs = df['Organization'].value_counts().head(10)
    print(top_orgs)

    # Skills analysis
    if 'Skills' in df.columns:
        print(f"\n💡 Skills Analysis:")
        all_skills = []
        for skills_str in df['Skills'].dropna():
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"   Total unique skills: {len(skill_counts)}")
            print(f"   Most common skills:")
            for skill, count in skill_counts.most_common(10):
                print(f"     {skill}: {count}")

    # Satisfaction Rate
    if 'Satisfaction Rate' in df.columns:
        print(f"\n😊 Satisfaction Rate:")
        print(df['Satisfaction Rate'].describe())

def main():
    """Main function to run the complete EDA."""
    print("🚀 Starting Exploratory Data Analysis...")
    
    # Load datasets
    datasets = load_datasets()
    
    # Perform EDA
    basic_dataset_info(datasets)
    missing_values_analysis(datasets)
    
    # Dataset-specific analysis
    coursera_eda(datasets['coursera'])
    naukri_eda(datasets['naukri'])
    nyc_jobs_eda(datasets['nyc_jobs'])
    courses_eda(datasets['courses'])
    coursera_2024_eda(datasets['coursera_2024'])
    
    # Cross-dataset analysis
    cross_dataset_analysis(datasets)
    
    # Generate insights
    generate_insights_report(datasets)
    
    print("\n✅ EDA Complete! Check the visualizations above for insights.")

if __name__ == "__main__":
    main() 
