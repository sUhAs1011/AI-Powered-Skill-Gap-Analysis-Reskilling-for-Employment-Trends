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

- `eda_analysis`: This script analyzes raw job and course data, identifying missing values and key characteristics. It performs cross-dataset skill analysis to find common skills and highlight gaps between job requirements and course offerings.

- `data_processing`: This script cleans and standardizes raw data, handling missing values and normalizing skill names to a canonical form. It also creates a combined text field for each job and course, which is essential for later embedding.
  
- `populate_chromadb`: Generates This script converts the preprocessed text into vector embeddings using `all-MiniLM-L6-v2` and populates a ChromaDB vector database with these embeddings. It also performs an initial job-to-course similarity mapping, saving the results to a JSON file.

- `model_training`: This script trains a `Deep Structured Semantic Model (DSSM)` to refine job-course similarity. It uses embeddings from ChromaDB and generates a dataset of positive and negative pairs for training. The training process incorporates an `Exponential Moving Average (EMA)` and early stopping to save the best-performing model.

- `model_testing`: This script is a Streamlit web application that acts as the user interface. It analyzes a user's resume, identifies skill gaps for a desired job, and recommends relevant courses by leveraging either the pre-computed mappings or the trained DSSM model for deeper semantic matching.

- `utils`: The utils.py script serves as a central toolkit for the project, containing reusable helper functions that standardize and preprocess text and skills. It sets up `Natural Language Processing (NLP)` components from nltk and spacy for tasks like lemmatization and stop word removal. The script's core functionality revolves around a large `SKILL_NORMALIZATION_MAP` that maps common skill variations to a single, canonical form, ensuring consistency across all job and course data. This script is used by both data_processing.py and testing.py to clean and normalize text, extract skills, and perform semantic comparisons, ensuring that data is consistently formatted before it's used for model training or user interaction. 

### Architecture Diagram
![WhatsApp Image 2025-08-25 at 17 08 29_c9deb4c6](https://github.com/user-attachments/assets/17cf07cb-a8ee-498a-a080-66e6e984af04)

 ### Model Training Results 
<img width="3552" height="1768" alt="training_curves" src="https://github.com/user-attachments/assets/be045287-41da-4112-9e65-afe22c77c6c9" />

<img width="2370" height="570" alt="metrics_table" src="https://github.com/user-attachments/assets/bc2bb0d4-f157-4423-94e6-c9f00f5f8dc7" />


<img width="751" height="249" alt="image" src="https://github.com/user-attachments/assets/25ef7862-cebd-44f3-8821-925d897031a6" />

<img width="867" height="290" alt="image" src="https://github.com/user-attachments/assets/36217c5e-c20a-4422-b30e-78a515bf21f0" />



### Streamlit Interface
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d3b9ffd3-2226-4016-8955-4e796a264877" />

### Course Recommendation
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ef0748d4-d270-4380-9396-cfd040b7b2a8" />

### Invalid Document Upload
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f8a1f6d8-f265-422a-9575-f169464b7717" />

### Invalid Job Title
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c9b3b9cd-f551-46eb-a657-273a918eebba" />







