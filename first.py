import spacy
import json
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split

# --- Configuration ---
TRAINING_DATA_PATH = "train_data_complex.json"
TRAIN_OUTPUT_PATH = "train.spacy"
DEV_OUTPUT_PATH = "dev.spacy"
TEST_SIZE = 0.2  # 20% for development/evaluation
RANDOM_STATE = 42
BASE_MODEL = "en_core_web_lg"

def create_doc_bin(data, nlp):
    """Converts the list of (text, entities) into a DocBin object."""
    doc_bin = DocBin()
    
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        
        # Iterate over entities defined by character offsets
        for start, end, label in annotations.get("entities", []):
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
            else:
                # This ensures you catch any small errors in your character offsets
                print(f"Skipping entity: Label='{label}' Text='{text[start:end]}' in '{text}'")
        
        doc.ents = ents
        doc_bin.add(doc)
    return doc_bin

if __name__ == "__main__":
    print(f"Loading base model: {BASE_MODEL}")
    try:
        # Load the base model to get the tokenizer and vocabulary
        nlp = spacy.load(BASE_MODEL)
    except OSError:
        print(f"Model '{BASE_MODEL}' not found. Downloading...")
        spacy.cli.download(BASE_MODEL)
        nlp = spacy.load(BASE_MODEL)

    print(f"Loading training data from {TRAINING_DATA_PATH}")
    with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # Split data into training and development sets
    train_data, dev_data = train_test_split(
        full_data, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

    print(f"Total examples: {len(full_data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Development examples: {len(dev_data)}")

    # Convert and save the training data
    train_db = create_doc_bin(train_data, nlp)
    train_db.to_disk(TRAIN_OUTPUT_PATH)
    print(f"Saved training data to {TRAIN_OUTPUT_PATH}")

    # Convert and save the development data
    dev_db = create_doc_bin(dev_data, nlp)
    dev_db.to_disk(DEV_OUTPUT_PATH)
    print(f"Saved development data to {DEV_OUTPUT_PATH}")

    print("Data preparation complete. Ready for training.")