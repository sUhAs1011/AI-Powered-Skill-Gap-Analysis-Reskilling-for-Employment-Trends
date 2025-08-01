# C3I

### Objective

- Develop a model that extracts skills from user resumes and identifies gaps between current capabilities and target job requirements.

- Create a deep learning-based recommendation system using DSSM that suggests relevant courses based on identified skill gaps and job requirements.

- Build a robust system that can extract skills from various resume formats (PDF, DOCX, images) using OCR and NLP techniques, filtering out irrelevant content.

- Develop an interactive Streamlit application that provides instant career guidance, skill analysis, and course recommendations with a user-friendly interface.


### Implementation

- eda_analysis: Analyzes raw job/course data, identifying missing values, performing dataset-specific insights, and conducting initial skill gap analysis via exact matching to inform preprocessing.

- data_processing: Cleans and standardizes raw data, handling missing values robustly and performing advanced skill normalization. It creates `combined_text` fields for embeddings and saves preprocessed files for later stages.

- populate_chromadb: Generates `all-MiniLM-L6-v2` embeddings for preprocessed jobs, courses, and individual skills, storing them in ChromaDB. It also creates initial job-to-course similarity mappings.

- model_training: Trains a Deep Structured Semantic Model (DSSM) using ChromaDB embeddings, generating positive/negative pairs for contrastive learning. It employs CosineEmbeddingLoss, EMA(Exponential Moving Averaage) for validation, and early stopping to optimize and save the best model.

- model_testing: A Streamlit web application that integrates the trained DSSM and ChromaDB. It analyzes user resumes, identifies skill gaps for desired jobs, and recommends relevant courses, leveraging both pre-computed mappings and DSSM insights.


