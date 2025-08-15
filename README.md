# Centre of Cognitive Computing and Computational Intelligence(C3I)

### Key Features 

- Utilized all-MiniLM-L6-v2 to generate and push refined job and course embeddings into ChromaDB for efficient semantic search.

- Employed a Deep Structured Semantic Model (DSSM) for training to learn enhanced semantic relationships.

- Developed a Streamlit web application as a user-friendly frontend interface, facilitating interactive skill gap analysis and course recommendations.

- Provided intelligent course suggestions directly addressing identified skill gaps relevant to a specific job position, leveraging both pre-computed mappings and the trained DSSM.

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

### Architecture Diagram
<img width="1581" height="685" alt="image" src="https://github.com/user-attachments/assets/41760cde-c598-4ba1-b64c-a1edd68e53cb" />

 ### Model Training Results 
 <img width="3541" height="1768" alt="training_curves" src="https://github.com/user-attachments/assets/f51a0967-f0ae-400a-bad7-41e2a03d9320" />


### Streamlit Interface
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/31d20d53-f02e-4599-8981-0e9c1c504087" />


### Course Recommendation
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ef0748d4-d270-4380-9396-cfd040b7b2a8" />

### Invalid Document Upload
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f8a1f6d8-f265-422a-9575-f169464b7717" />

### Invalid Job Title
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c9b3b9cd-f551-46eb-a657-273a918eebba" />







