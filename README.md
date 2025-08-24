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

- eda_analysis: This script analyzes raw job and course data, identifying missing values and key characteristics. It performs cross-dataset skill analysis to find common skills and highlight gaps between job requirements and course offerings.

- data_processing: This script cleans and standardizes raw data, handling missing values and normalizing skill names to a canonical form. It also creates a combined text field for each job and course, which is essential for later embedding.
  
- populate_chromadb: Generates This script converts the preprocessed text into vector embeddings using 'all-MiniLM-L6-v2' and populates a ChromaDB vector database with these embeddings. It also performs an initial job-to-course similarity mapping, saving the results to a JSON file.

- model_training: This script trains a Deep Structured Semantic Model (DSSM) to refine job-course similarity. It uses embeddings from ChromaDB and generates a dataset of positive and negative pairs for training. The training process incorporates an Exponential Moving Average (EMA) and early stopping to save the best-performing model.

- model_testing: This script is a Streamlit web application that acts as the user interface. It analyzes a user's resume, identifies skill gaps for a desired job, and recommends relevant courses by leveraging either the pre-computed mappings or the trained DSSM model for deeper semantic matching.

### Architecture Diagram
<img width="1581" height="685" alt="image" src="https://github.com/user-attachments/assets/41760cde-c598-4ba1-b64c-a1edd68e53cb" />

 ### Model Training Results 
<img width="3523" height="1768" alt="training_curves" src="https://github.com/user-attachments/assets/082c7879-5664-40b3-83a6-525d1789970f" />

<img width="1018" height="421" alt="image" src="https://github.com/user-attachments/assets/a1037ba7-e0d1-40d0-a2bb-f68680daf771" />

<img width="1062" height="264" alt="image" src="https://github.com/user-attachments/assets/497969e6-b205-4908-8259-aea7661b25aa" />



### Streamlit Interface
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d3b9ffd3-2226-4016-8955-4e796a264877" />

### Course Recommendation
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ef0748d4-d270-4380-9396-cfd040b7b2a8" />

### Invalid Document Upload
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f8a1f6d8-f265-422a-9575-f169464b7717" />

### Invalid Job Title
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c9b3b9cd-f551-46eb-a657-273a918eebba" />







