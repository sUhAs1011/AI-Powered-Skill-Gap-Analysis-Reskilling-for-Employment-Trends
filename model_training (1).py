import chromadb
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import random
import json
# --- Paths and Collection Names ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
JOB_DATASET_DIR = BASE_DIR / "job_dataset"

COLLECTIONS = [
    "job_skills_embeddings",
    "course_skills_embeddings", 
    "jobs_embeddings",
    "courses_embeddings"
]

# DSSM Configuration
DSSM_CONFIG = {
    'query_dim': 384,  # all-mpnet-base-v2 output dimension
    'doc_dim': 384,
    "hidden_dims": [256,128,64],  # You can tune these as needed
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'margin': 0.2  # Margin for triplet loss
}

def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.max_seq_length = 512
    model.eval()
    return model

class DSSMModel(nn.Module):
    """Deep Structured Semantic Model for job-course matching."""
    
    def __init__(self, query_dim, doc_dim, hidden_dims, dropout=0.1):
        super(DSSMModel, self).__init__()
        
        # Query tower (for job descriptions)
        self.query_tower = self._build_tower(query_dim, hidden_dims, dropout)
        
        # Document tower (for course descriptions)
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
        # Pass through respective towers
        query_features = self.query_tower(query_emb)
        doc_features = self.doc_tower(doc_emb)
        return query_features, doc_features

class TripletDataset(Dataset):
    """Dataset for triplet learning with job-course pairs."""
    
    def __init__(self, job_embeddings, course_embeddings, job_metadata, course_metadata, all_pairs):
        self.job_embeddings = job_embeddings
        self.course_embeddings = course_embeddings
        self.job_metadata = job_metadata
        self.course_metadata = course_metadata
        self.all_pairs = all_pairs
    
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        job_id, course_id, label = self.all_pairs[idx]
        
        job_emb = torch.tensor(self.job_embeddings[job_id], dtype=torch.float32)
        course_emb = torch.tensor(self.course_embeddings[course_id], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return job_emb, course_emb, label

def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))

def get_collection(client, name):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using various methods."""
    if not text or not isinstance(text, str):
        return []
    
    skills = []
    
    # Method 1: Look for skills in brackets or parentheses
    skill_patterns = [
        r'\[([^\]]+)\]',  # [skill1, skill2]
        r'\(([^)]+)\)',   # (skill1, skill2)
        r'"([^"]+)"',     # "skill1, skill2"
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if ',' in match:
                skills.extend([s.strip() for s in match.split(',') if s.strip()])
            else:
                skills.append(match.strip())
    
    # Method 2: Try to parse as Python list
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            skills.extend([str(s).strip() for s in parsed if s])
    except:
        pass
    
    # Method 3: Split by common delimiters
    delimiters = [';', ',', '|', '•', '-']
    for delimiter in delimiters:
        if delimiter in text:
            parts = text.split(delimiter)
            skills.extend([s.strip() for s in parts if s.strip() and len(s.strip()) > 2])
            break
    
    # Method 4: Extract individual words that look like skills
    words = re.findall(r'\b[A-Z][a-zA-Z\s&]+(?:\.js|\.py|\.net|\.com)?\b', text)
    skills.extend([w.strip() for w in words if len(w.strip()) > 2 and w.strip().lower() not in ['the', 'and', 'for', 'with', 'from']])
    
    return list(set(skills))  # Remove duplicates

def get_title_from_metadata(meta: Dict[str, Any]) -> str:
    """Extract title from metadata."""
    title_fields = ['title', 'job_title', 'jobtitle', 'business_title', 'course_title', 'name']
    
    for field in title_fields:
        if field in meta and meta[field]:
            return str(meta[field])
    
    # Fallback: look for any field with 'title' in the name
    for key, value in meta.items():
        if 'title' in key.lower() and value:
            return str(value)
    
    # Last resort: use the first non-empty string field
    for key, value in meta.items():
        if isinstance(value, str) and value.strip():
            return value.strip()
    
    return ""

def get_skills_from_metadata(meta: Dict[str, Any]) -> List[str]:
    """Extract skills from metadata."""
    skills_fields = ['skills', 'job_skills', 'preferred_skills', 'required_skills', 'technical_skills', 'skill_name', 'Canonical_Course_Skills']
    
    for field in skills_fields:
        if field in meta and meta[field]:
            skills = extract_skills_from_text(str(meta[field]))
            if skills:
                return skills
    
    # Fallback: look for any field with 'skill' in the name
    for key, value in meta.items():
        if 'skill' in key.lower() and value:
            skills = extract_skills_from_text(str(value))
            if skills:
                return skills
    
    return []

def extract_embeddings_from_chromadb(client, collection_name, limit=10000):
    """Extract embeddings and metadata from ChromaDB collection."""
    try:
        collection = client.get_or_create_collection(name=collection_name)
        results = collection.get(include=["embeddings", "metadatas"], limit=limit)
        
        embeddings = results.get('embeddings', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])
        
        print(f"  📊 Collection '{collection_name}': {len(embeddings)} embeddings, {len(metadatas)} metadata records")
        
        return {id_: emb for id_, emb in zip(ids, embeddings)}, metadatas, ids
    except Exception as e:
        print(f"Error extracting from {collection_name}: {e}")
        return {}, [], []

def read_csv_with_fallback(path):
    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    print(f"  Error loading {path}: {last_err}")
    return None

def check_available_data():
    """Check what data is available in the job_dataset directory."""
    print(f"🔍 Checking available data in {JOB_DATASET_DIR}...")
    
    if not JOB_DATASET_DIR.exists():
        print(f"❌ Job dataset directory not found: {JOB_DATASET_DIR}")
        return False
    
    # List available CSV files
    csv_files = list(JOB_DATASET_DIR.glob("*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files:")
    
    for csv_file in csv_files:
        try:
            # Try to read the first few lines to check the file
            df = read_csv_with_fallback(csv_file)
            if df is not None:
                df_head = df.head(5)
                print(f"  ✅ {csv_file.name}: {df_head.shape[0]} rows, {df_head.shape[1]} columns")
                print(f"     Columns: {list(df_head.columns)}")
            else:
                print(f"  ❌ {csv_file.name}: Could not read file with any encoding.")
        except Exception as e:
            print(f"  ❌ {csv_file.name}: Error reading file - {e}")
    
    return len(csv_files) > 0



def train_dssm_model(dssm_model, train_loader, val_loader, config):
    """Train the DSSM model with CosineEmbeddingLoss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dssm_model.to(device)
    
    optimizer = torch.optim.Adam(dssm_model.parameters(), lr=config['learning_rate'])
    # Use CosineEmbeddingLoss for matching tasks, as it's better for learning similarity
    criterion = nn.CosineEmbeddingLoss(margin=config.get('margin', 0.5))
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # --- Exponential Moving Average for Validation Loss ---
    ema_alpha = 0.1  # Smoothing factor (adjust as needed)
    ema_val_loss = None
    
    print(f"🚀 Starting DSSM training for {config['epochs']} epochs...")
    print(f"📊 Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"💻 Device: {device}")
    print("-" * 80)
    
    epochs_no_improve = 0  # Counter for early stopping
    for epoch in range(config['epochs']):
        # Training
        dssm_model.train()
        train_loss = 0.0
        
        print(f"\n📚 Epoch {epoch+1}/{config['epochs']} - Training Phase")
        print("-" * 50)
        
        for batch_idx, (job_emb, course_emb, labels) in enumerate(train_loader):
            job_emb, course_emb, labels = job_emb.to(device), course_emb.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass to get embeddings
            query_features, doc_features = dssm_model(job_emb, course_emb)
            
            # Create target tensor for CosineEmbeddingLoss: 1 for positive, -1 for negative
            target_labels = torch.where(labels > 0, 1.0, -1.0).to(device)
            
            loss = criterion(query_features, doc_features, target_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print batch progress every 50 batches
            if batch_idx > 0 and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:4d}/{len(train_loader):4d} | "
                      f"Current Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        dssm_model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        print(f"\n🔍 Epoch {epoch+1}/{config['epochs']} - Validation Phase")
        print("-" * 50)
        
        with torch.no_grad():
            for batch_idx, (job_emb, course_emb, labels) in enumerate(val_loader):
                job_emb, course_emb, labels = job_emb.to(device), course_emb.to(device), labels.to(device)
                
                query_features, doc_features = dssm_model(job_emb, course_emb)
                target_labels = torch.where(labels > 0, 1.0, -1.0).to(device)
                loss = criterion(query_features, doc_features, target_labels)
                val_loss += loss.item()
                
                # For metrics, calculate the cosine similarity of the output embeddings
                similarities = F.cosine_similarity(query_features, doc_features)
                val_predictions.extend(similarities.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # --- Exponential Moving Average Update ---
        if ema_val_loss is None:
            ema_val_loss = avg_val_loss
        else:
            ema_val_loss = ema_alpha * avg_val_loss + (1 - ema_alpha) * ema_val_loss
        
        # --- Find the best threshold for accuracy on the validation set ---
        val_predictions = np.array(val_predictions)
        val_labels_binary = (np.array(val_labels) > 0).astype(int)
        
        best_accuracy = 0
        best_threshold = 0.0
        # Iterate over a range of potential thresholds to find the best one
        for threshold in np.arange(-1.0, 1.0, 0.05):
            val_pred_binary = (val_predictions > threshold).astype(int)
            accuracy = np.mean(val_pred_binary == val_labels_binary)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch+1}/{config['epochs']} Summary:")
        print(f"  Training Loss:   {avg_train_loss:.6f}")
        print(f"  Validation Loss: {avg_val_loss:.6f}")
        print(f"  EMA Val Loss:    {ema_val_loss:.6f}")
        print(f"  Best Val Thresh: {best_threshold:.2f}")
        print(f"  Validation Acc:  {best_accuracy:.4f}")
        
        # Save best model (based on EMA validation loss)
        if ema_val_loss < best_val_loss:
            best_val_loss = ema_val_loss
            torch.save(dssm_model.state_dict(), BASE_DIR / "trained_model" / "dssm_best_model.pth")
            print(f"  ✅ New best model saved! (EMA Val Loss: {best_val_loss:.6f})")
            epochs_no_improve = 0  # Reset counter
        else:
            print(f"  ⏸  No improvement (Best EMA: {best_val_loss:.6f})")
            epochs_no_improve += 1

        # Early stopping check
        if epoch > 5 and epochs_no_improve >= 3:  # Check after a few epochs
            print(f"  ⚠  Early stopping triggered (No improvement in EMA Val Loss for {epochs_no_improve} epochs)")
            break
        
    # Print final training summary
    print(f"\n🎉 Training Complete!")
    print(f"📊 Final Results:")
    print(f"  Best Validation Loss: {best_val_loss:.6f}")
    print(f"  Final Training Loss:  {train_losses[-1]:.6f}")
    print(f"  Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"  Total Epochs Trained: {len(train_losses)}")
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }
    
    import json
    history_path = BASE_DIR / "trained_model" / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"📁 Training history saved to {history_path}")
    
    # Plot training curves if matplotlib is available
    try:
        plot_training_curves(train_losses, val_losses, save_path=BASE_DIR / "trained_model" / "training_curves.png")
        print(f"📊 Training curves saved to {BASE_DIR / 'trained_model' / 'training_curves.png'}")
    except ImportError:
        print("📊 Matplotlib not available - skipping training curve plot")
    
    return dssm_model

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Plot training and validation losses
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('DSSM Training and Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best validation loss
        best_epoch = val_losses.index(min(val_losses)) + 1
        best_loss = min(val_losses)
        plt.annotate(f'Best Val Loss: {best_loss:.6f}\nEpoch: {best_epoch}', 
                    xy=(best_epoch, best_loss), xytext=(best_epoch + 1, best_loss + 0.01),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def create_training_examples_from_metadata(metadatas: List[Dict[str, Any]]) -> List[InputExample]:
    """Create training examples from ChromaDB metadata."""
    train_examples = []
    
    for meta in metadatas:
        title = get_title_from_metadata(meta)
        skills = get_skills_from_metadata(meta)
        
        if not title or not skills:
            continue

        # Create positive examples: title paired with each skill
        for skill in skills:
            if skill and len(skill.strip()) > 2:
                train_examples.append(InputExample(
                    texts=[title, skill.strip()], 
                    label=1.0
                ))
        
        # Create negative examples: title paired with random skills from other entries
        # (This will be done in a separate pass to avoid duplicates)
    
    return train_examples

def create_negative_examples(metadatas: List[Dict[str, Any]], positive_examples: List[InputExample]) -> List[InputExample]:
    """Create negative training examples by pairing titles with unrelated skills."""
    negative_examples = []
    all_skills = []
    
    # Collect all skills
    for meta in metadatas:
        skills = get_skills_from_metadata(meta)
        all_skills.extend(skills)
    
    all_skills = list(set(all_skills))  # Remove duplicates
    
    # Create negative examples
    for meta in metadatas:
        title = get_title_from_metadata(meta)
        if not title:
            continue
        
        # Get skills for this entry
        entry_skills = set(get_skills_from_metadata(meta))
        
        # Pair title with skills from other entries (negative examples)
        for skill in all_skills:
            if skill not in entry_skills and len(skill.strip()) > 2:
                negative_examples.append(InputExample(
                    texts=[title, skill.strip()], 
                    label=0.0
                ))
                
                # Limit negative examples to avoid imbalance
                if len(negative_examples) >= len(positive_examples):
                    break
        
        if len(negative_examples) >= len(positive_examples):
            break
    
    return negative_examples

def normalize_dataset_name(fname):
    name = fname.lower().replace('.csv', '').replace('.json', '')
    name = re.sub(r'[ &\-]+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = name.strip('_')
    return name

def main():
    print("🚀 Starting DSSM model training using ChromaDB data...")
    
    # Create output directory
    model_save_path = BASE_DIR / "trained_model"
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Check available data first
    print("🔍 Checking available data...")
    data_available = check_available_data()
    
    if not data_available:
        print("❌ No data found. Please ensure your CSV files are in the job_dataset directory.")
        return
    
    client = get_chroma_client()
    base_model = get_embedding_model()
    
    # Extract embeddings from ChromaDB
    print("📊 Extracting embeddings from ChromaDB...")
    print(f"📁 ChromaDB path: {CHROMA_DATA_PATH}")
    print(f"📁 Job dataset path: {JOB_DATASET_DIR}")
    
    # Check if ChromaDB exists and has data
    if not CHROMA_DATA_PATH.exists():
        print(f"❌ ChromaDB directory not found at {CHROMA_DATA_PATH}")
        print("Please run populate_chromadb.py first to create embeddings.")
        return
    
    # Extract job embeddings
    job_embeddings, job_metadata, job_ids = extract_embeddings_from_chromadb(
        client, "jobs_embeddings", limit=100000
    )
    print(f"Extracted {len(job_embeddings)} job embeddings")
    print("Sample job_embedding keys:", list(job_embeddings.keys())[:20])
    # NOTE: Use normalize_dataset_name everywhere you use dataset_name for job_id or course_id
    
    # Extract course embeddings
    course_embeddings, course_metadata, course_ids = extract_embeddings_from_chromadb(
        client, "courses_embeddings", limit=100000
    )
    print(f"Extracted {len(course_embeddings)} course embeddings")
    print("Sample course_embedding keys:", list(course_embeddings.keys())[:10])
    
    if not job_embeddings or not course_embeddings:
        print("❌ No embeddings found. Please run populate_chromadb.py first.")
        return
    
    # --- Create positive pairs directly from embeddings using cosine similarity ---
    print("🔗 Creating positive pairs using cosine similarity between embeddings...")
    
    # Convert embeddings to numpy arrays for faster computation
    job_ids_list = list(job_embeddings.keys())
    course_ids_list = list(course_embeddings.keys())
    
    print(f"Computing similarities between {len(job_ids_list)} jobs and {len(course_ids_list)} courses...")
    
    # Sample a subset for faster computation (you can increase this)
    max_jobs = min(1000, len(job_ids_list))
    max_courses = min(2000, len(course_ids_list))
    
    sampled_job_ids = random.sample(job_ids_list, max_jobs)
    sampled_course_ids = random.sample(course_ids_list, max_courses)
    
    print(f"Using {max_jobs} jobs and {max_courses} courses for similarity computation...")
    
    # Create job and course embedding matrices
    job_emb_matrix = np.array([job_embeddings[job_id] for job_id in sampled_job_ids])
    course_emb_matrix = np.array([course_embeddings[course_id] for course_id in sampled_course_ids])
    
    # Compute cosine similarities
    similarities = np.dot(job_emb_matrix, course_emb_matrix.T)
    job_norms = np.linalg.norm(job_emb_matrix, axis=1, keepdims=True)
    course_norms = np.linalg.norm(course_emb_matrix, axis=1, keepdims=True)
    similarities = similarities / (job_norms * course_norms.T)
    
    # Find positive pairs (similarity > threshold)
    threshold = 0.3  # Adjust this threshold as needed
    positive_pairs = []
    
    for i, job_id in enumerate(sampled_job_ids):
        for j, course_id in enumerate(sampled_course_ids):
            sim_score = similarities[i, j]
            if sim_score > threshold:
                positive_pairs.append((job_id, course_id, sim_score))
    
    print(f"Created {len(positive_pairs)} positive pairs using cosine similarity (threshold: {threshold})")
    if len(positive_pairs) < 100:
        print("⚠  Warning: Very few positive pairs. Consider lowering similarity threshold.")

    # Filter positive pairs to only those with valid job_id and course_id
    job_ids_set = set(job_embeddings.keys())
    course_ids_set = set(course_embeddings.keys())

    def add_job_prefix(job_id):
        return job_id if job_id.startswith('job_') else f'job_{job_id}'

    def add_course_prefix(course_id):
        return course_id if course_id.startswith('course_') else f'course_{course_id}'

    filtered_positive_pairs = [
        (add_job_prefix(job_id), add_course_prefix(course_id), sim_score)
        for (job_id, course_id, sim_score) in positive_pairs
        if add_job_prefix(job_id) in job_ids_set and add_course_prefix(course_id) in course_ids_set
    ]
    print(f"Filtered positive pairs: {len(filtered_positive_pairs)} (from {len(positive_pairs)})")

    # --- Balance dataset: sample negatives efficiently ---
    positive_set = {(p[0], p[1]) for p in filtered_positive_pairs}
    negative_ratio = 2  # For every positive, use 2 negatives
    num_negatives_to_sample = negative_ratio * len(filtered_positive_pairs)
    
    negative_pairs = set()
    job_ids_list = list(job_ids_set)
    course_ids_list = list(course_ids_set)
    
    print(f"🔄 Sampling {num_negatives_to_sample} negative pairs efficiently...")
    max_attempts = num_negatives_to_sample * 5  # Prevent infinite loops
    attempts = 0
    
    while len(negative_pairs) < num_negatives_to_sample and attempts < max_attempts:
        job_id = random.choice(job_ids_list)
        course_id = random.choice(course_ids_list)
        
        if (job_id, course_id) not in positive_set:
            negative_pairs.add((job_id, course_id, 0.0))
        
        attempts += 1
    
    print(f"✅ Generated {len(negative_pairs)} unique negative pairs after {attempts} attempts")
    
    all_pairs = filtered_positive_pairs + list(negative_pairs)
    random.shuffle(all_pairs)
    print(f"Total pairs for training: {len(all_pairs)} (Positives: {len(filtered_positive_pairs)}, Negatives: {len(negative_pairs)})")

    # Create dataset
    print("📦 Creating training dataset...")
    dataset = TripletDataset(
        job_embeddings, course_embeddings, job_metadata, course_metadata,
        all_pairs
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=DSSM_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DSSM_CONFIG['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize DSSM model
    print("🏗  Initializing DSSM model...")
    dssm_model = DSSMModel(
        query_dim=DSSM_CONFIG['query_dim'],
        doc_dim=DSSM_CONFIG['doc_dim'],
        hidden_dims=DSSM_CONFIG['hidden_dims'],
        dropout=DSSM_CONFIG['dropout']
    )
    
    # Train the model
    print("🎯 Starting DSSM training...")
    trained_model = train_dssm_model(dssm_model, train_loader, val_loader, DSSM_CONFIG)
    
    # Save the final model
    final_model_path = model_save_path / "dssm_final_model.pth"
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"✅ DSSM model training complete and saved to {final_model_path}")
    
    # Also save the base sentence transformer model for comparison
    print("💾 Saving base sentence transformer model...")
    base_model_save_path = model_save_path / "all-MiniLM-L6-v2-finetuned"
    base_model_save_path.mkdir(parents=True, exist_ok=True)
    base_model.save(str(base_model_save_path))
    print(f"✅ Base model saved to {base_model_save_path}")
    
    print("🎉 Training complete! You can now use the DSSM model for job-course matching.")
    
    # Test the trained model
    print("🧪 Testing the trained DSSM model...")
    job_id_to_meta = {id_: meta for id_, meta in zip(job_ids, job_metadata)}
    course_id_to_meta = {id_: meta for id_, meta in zip(course_ids, course_metadata)}
    test_dssm_model(trained_model, job_embeddings, course_embeddings, job_id_to_meta, course_id_to_meta)

def test_dssm_model(dssm_model, job_embeddings, course_embeddings, job_id_to_meta, course_id_to_meta, num_tests=5):
    """Test the trained DSSM model with sample job-course pairs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dssm_model.to(device)
    dssm_model.eval()
    
    job_ids = list(job_embeddings.keys())
    course_ids = list(course_embeddings.keys())
    
    print(f"\n🔍 Testing DSSM model with {num_tests} random job-course pairs:")
    print("-" * 80)
    
    for i in range(num_tests):
        # Random job and course
        job_id = random.choice(job_ids)
        course_id = random.choice(course_ids)
        
        job_emb = torch.tensor(job_embeddings[job_id], dtype=torch.float32).unsqueeze(0).to(device)
        course_emb = torch.tensor(course_embeddings[course_id], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get the final embeddings from the model
            job_features, course_features = dssm_model(job_emb, course_emb)
            # Calculate the cosine similarity between the output embeddings
            similarity_score = F.cosine_similarity(job_features, course_features).item()
        
        # Get metadata for display using the pre-built mapping
        job_meta = job_id_to_meta.get(job_id, {})
        course_meta = course_id_to_meta.get(course_id, {})
        
        job_title = get_title_from_metadata(job_meta) or "Unknown Job"
        course_title = get_title_from_metadata(course_meta) or "Unknown Course"
        
        print(f"Test {i+1}:")
        print(f"  Job: {job_title}")
        print(f"  Course: {course_title}")
        print(f"  DSSM Similarity Score: {similarity_score:.4f}")
        print(f"  Match Quality: {'High' if similarity_score > 0.7 else 'Medium' if similarity_score > 0.4 else 'Low'}")
        print()
if __name__ == "__main__":
    main()
