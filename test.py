import spacy

# The path where the final model (with both NER and EntityRuler) was saved
MODEL_PATH = "./output-v4/model-best"

# --- Test Sentences ---
test_texts = [
    # 1. Complex Address (Tests Statistical NER)
    	"full name is Soham Rathi age 45 son of Mr. Vivek  residing at H.No. 12/99, Main Road, Kaka Nagar, Pune-411005"
]

def test_model(model_path, texts):
    """Loads the model and prints the entities for a list of texts."""
    try:
        # Load the final hybrid model
        nlp = spacy.load(model_path)
        print(f"Model '{model_path}' loaded successfully. Testing entities...\n")
        
        for text in texts:
            doc = nlp(text)
            print("-" * 50)
            print(f"Text: {text}")
            
            if doc.ents:
                print("Extracted Entities:")
                for ent in doc.ents:
                    # Print the text, label, and the component that created the entity
                    # The source is added by the EntityRuler or the statistical NER
                    source = ent.ent_id_ if ent.ent_id_ else "Statistical NER"
                    print(f"  > {ent.text:<30} | Label: {ent.label_:<15} | Source: {source}")
            else:
                print("No entities extracted.")

    except OSError:
        print(f"ERROR: Could not find model at path: {model_path}")
        print("Ensure 'add_ruler.py' ran successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_model(MODEL_PATH, test_texts)