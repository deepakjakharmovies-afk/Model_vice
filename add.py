import spacy
import srsly
import os

# --- Configuration ---
# Trained model output from Step 2
TRAINED_MODEL_PATH = "./output/model-best"
# Your pattern file
PATTERNS_FILE = "patterns.json" 
# New directory for the final model
FINAL_MODEL_PATH = "./output-v2/model22" 
# Name of the ruler component in the pipeline
RULER_NAME = "entity_ruler"

print(f"Loading trained model from {TRAINED_MODEL_PATH}...")
# Load the statistical model you just trained
nlp = spacy.load(TRAINED_MODEL_PATH)

# Load patterns using srsly (same reader that was available in the error message)
print(f"Loading patterns from {PATTERNS_FILE}...")
patterns = list(srsly.read_json(PATTERNS_FILE))

# Create the EntityRuler component
ruler = nlp.add_pipe(
    RULER_NAME, 
    # Add the ruler before the NER model so its rules run first
    before="ner", 
    config={"overwrite_ents": True} # Overwrite statistical entities if a rule matches
)

# Add all patterns to the ruler
ruler.add_patterns(patterns)

# Save the final model with the ruler included
if not os.path.exists(FINAL_MODEL_PATH):
    os.makedirs(FINAL_MODEL_PATH)

nlp.to_disk(FINAL_MODEL_PATH)

print("\n✨ SUCCESS! ✨")
print(f"Final hybrid model saved to: {FINAL_MODEL_PATH}")
# print("You can now load your model using: spacy.load('./output/final-model-with-ruler')")
# === Custom scorer for ADDRESS weighting ===
# === Custom scorer for ADDRESS weighting ===
# === Custom scorer registration ===
from spacy import registry
from spacy.scorer import Scorer

@registry.scorers.register("spacy.custom_weighted_ner_scorer.v1")
def make_weighted_scorer():
    def weighted_scorer(examples, **kwargs):
        scorer = Scorer()
        results = scorer.score(examples)
        if "ents_per_type" in results and "ADDRESS" in results["ents_per_type"]:
            address_f = results["ents_per_type"]["ADDRESS"]["f"]
            weight = 2.0  # Increase to give ADDRESS more emphasis
            results["ents_f"] = (results["ents_f"] + weight * address_f) / (1 + weight)
        return results
    return weighted_scorer

# Optional: tell spaCy to actually use it automatically at runtime
from spacy.training.loop import train
import spacy

old_train = train
def custom_train(*args, **kwargs):
    kwargs["scorer"] = make_weighted_scorer()
    return old_train(*args, **kwargs)
spacy.training.loop.train = custom_train
