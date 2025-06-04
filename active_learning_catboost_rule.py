# Imports for interactive widgets
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
import json
import numpy as np

print("Loading data...")
# Load sentences data
with open("sentences_data_tagged_catboost.json", "r", encoding="utf-8") as f:
    sentences_data = json.load(f)

# Extract sentences
all_sentences = [item['sentence'] for item in sentences_data]
print(f"Loaded {len(all_sentences)} sentences")

print("Getting embeddings...")
# Get embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
all_embeddings = model.encode(all_sentences)
print("Embeddings complete")

print("Loading classifier...")
# Initialize and load trained classifier
clf = CatBoostClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=3,
    l2_leaf_reg=5,
    random_seed=42,
    early_stopping_rounds=10,
    verbose=False
)
clf.load_model('rule_extraction_model.cbm')

print("Getting initial predictions...")
# Get initial predictions
all_predictions = clf.predict(all_embeddings)
prediction_probs = clf.predict_proba(all_embeddings)

# Function to get indices of lowest confidence predictions
def get_low_confidence_indices(probs, n=10):
    # Get confidence scores (max probability for each prediction)
    confidences = np.max(probs, axis=1)
    # Get indices sorted by confidence
    return np.argsort(confidences)[:n]

# Function to get random indices
def get_random_indices(total_len, n=10, exclude_indices=None):
    exclude_set = set(exclude_indices) if exclude_indices is not None else set()
    available_indices = list(set(range(total_len)) - exclude_set)
    if len(available_indices) < n:
        return available_indices
    return np.random.choice(available_indices, size=n, replace=False)

iteration = 0
print("\nStarting active learning loop...")
while True:
    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    
    # Get 10 lowest confidence predictions
    low_conf_indices = get_low_confidence_indices(prediction_probs, n=10)

    print("Low confidence indices: ",low_conf_indices)
    
    # Get 10 random sentences (excluding low confidence ones)
    random_indices = get_random_indices(len(all_sentences), n=10, exclude_indices=low_conf_indices)
    
    print("\nLow confidence sentences to check:")
    print("----------------------------------")
    low_conf_labeled = 0
    for idx in low_conf_indices:
        conf_score = max(prediction_probs[idx])
        current_pred = "Rule" if all_predictions[idx] == 1 else "Non-rule"
        print(f"\nSentence {idx} (Confidence: {conf_score:.3f}, Current prediction: {current_pred}):")
        print(all_sentences[idx])
        print("Is this a rule? (y/n/q):", end=" ")
        label = input().lower()
        if label == 'q':
            exit()
        sentences_data[idx]['rule_tag'] = (label == 'y')
        low_conf_labeled += 1
    print(f"Labeled {low_conf_labeled} low confidence sentences")

    print("\nRandom sentences to check:")
    print("-------------------------")
    random_labeled = 0
    for idx in random_indices:
        conf_score = max(prediction_probs[idx])
        current_pred = "Rule" if all_predictions[idx] == 1 else "Non-rule"
        print(f"\nSentence {idx} (Confidence: {conf_score:.3f}, Current prediction: {current_pred}):")
        print(all_sentences[idx])
        print("Is this a rule? (y/n/q):", end=" ")
        label = input().lower()
        if label == 'q':
            exit()
        sentences_data[idx]['rule_tag'] = (label == 'y')
        random_labeled += 1
    print(f"Labeled {random_labeled} random sentences")
    
    
        
    print("\nRetraining model...")
    # Retrain model with updated labels
    train_labels = [item['rule_tag'] for item in sentences_data]
    clf.fit(all_embeddings, train_labels)
    
    print("Updating predictions...")
    # Update predictions and probabilities
    all_predictions = clf.predict(all_embeddings)
    prediction_probs = clf.predict_proba(all_embeddings)
    
    print("Saving data...")
    # Save updated data
    with open("sentences_data_tagged_catboost.json", "w", encoding="utf-8") as f:
        json.dump(sentences_data, f, ensure_ascii=False, indent=4)

    # Save the model
    clf.save_model(f'rule_extraction_model_iteration_{iteration}.cbm')

    print(f"Iteration {iteration} completed.")

    # Ask if user wants to continue
    print("\nContinue with next batch? (y/n):", end=" ")
    cont = input().lower()
    if cont != 'y':
        break
