import spacy
import json
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split

# --- Configuration ---
TRAINING_DATA_PATH = "ner_dataset.json"
TRAIN_OUTPUT_PATH = "train.spacy"
DEV_OUTPUT_PATH = "dev.spacy"
TEST_SIZE = 0.2  # 20% for development/evaluation
RANDOM_STATE = 42

def create_doc_bin(data, nlp):
    """Converts the list of (text, entities) into a DocBin object."""
    doc_bin = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations.get("entities", []):
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is not None:
                ents.append(span)
        if ents:
            try:
                doc.ents = ents
                doc_bin.add(doc)
            except ValueError:
                # Catches residual token overlap errors
                pass 
    return doc_bin

# Load a blank model for tokenization (safe and stable)
nlp = spacy.blank("en") 

print(f"Loading and splitting data from {TRAINING_DATA_PATH}")
with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# Split data into training and development sets
train_data, dev_data = train_test_split(
    full_data, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE
)

# Convert and save the training and development data
train_db = create_doc_bin(train_data, nlp)
train_db.to_disk(TRAIN_OUTPUT_PATH)
dev_db = create_doc_bin(dev_data, nlp)
dev_db.to_disk(DEV_OUTPUT_PATH)

print(f"Data saved to {TRAIN_OUTPUT_PATH} and {DEV_OUTPUT_PATH}")