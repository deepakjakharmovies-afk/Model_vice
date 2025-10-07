from fastapi import FastAPI
import spacy
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# ... other imports (FastAPI, spacy, json)
# Load the trained spaCy model once at startup
MODEL_PATH = "./output-v4/model-best" 
nlp = spacy.load(MODEL_PATH)
# api.py

app = FastAPI()

# ----------------- FIX IS HERE -----------------

# Define which origins are allowed to connect
origins = [
    "*", # Wildcard allows ANY origin (easiest for local testing)
    # If you want to be more specific:
    # "http://127.0.0.1:8000",
    # "http://10.0.2.2:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)

# -----------------------------------------------

# @app.post("/stream_info") # ... rest of your code

# Data model for the incoming streaming text
class StreamText(BaseModel):
    current_chunk: str # The new words just spoken
    
def extract_single_entity(text: str):
    """Processes a small chunk of text and returns a list of (entity_text, label)."""
    doc = nlp(text)
    
    # We want a list of tuples for simplicity in highlighting
    # Example: [("Deepak", "NAME"), ("Noida", "ADDRESS")]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

@app.post("/stream_info")
def process_stream_text(data: StreamText):
    """
    Receives a chunk of text and returns entities found in that chunk.
    This is designed to be called rapidly by the Flutter app.
    """
    chunk = data.current_chunk.strip()
    
    if not chunk:
        return {"entities": []}
        
    extracted_list = extract_single_entity(chunk)
    
    # Return a list of entities (text and label)
    return {"entities": extracted_list}

# Run the API with: python -m uvicorn api:app --reload