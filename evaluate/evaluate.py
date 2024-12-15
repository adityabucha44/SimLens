from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def predict_top_k_labels(features, labels, test_features, k=5):
    predictions = []
    for test_feature in test_features:
        test_feature_2d = test_feature.reshape(1, -1)  # Shape (1, n_features)
        similarities = cosine_similarity(test_feature_2d, features)  # Similarity with all features
        top_k_indices = np.argsort(similarities[0])[-k:][::-1]  # Indices of top-k most similar
        predictions.append([labels[idx] for idx in top_k_indices])  # Collect top-k labels
    return predictions


# Evaluate model using top-k predictions with binary decisions
def evaluate_model_top_k_binary(features, labels, model_name="Model", k=5):
    print(f"Evaluating {model_name} with Top-{k} Neighbors...")

    # Generate top-k predictions for the dataset
    top_k_predictions = predict_top_k_labels(features, labels, features, k=k)

    # Prepare true labels and count how many are correct in top-k predictions
    y_true = labels
    binary_predictions = []
    true_counts = []  # Stores how many true labels are in top-k for each sample

    for i, top_k in enumerate(top_k_predictions):
        true_count = sum([1 for pred in top_k if pred == y_true[i]])  # Count correct labels in top-k
        true_counts.append(true_count)  # Save count

        # Binary decision: 1 if true_count > k/2, else 0
        binary_predictions.append(1 if true_count > k // 2 else 0)

    # Compute metrics
    y_binary_true = [1] * len(y_true)  # Ground truth binary (1 for all samples, as they are correct by definition)
    accuracy = accuracy_score(y_binary_true, binary_predictions)
    precision = precision_score(y_binary_true, binary_predictions, zero_division=1)
    recall = recall_score(y_binary_true, binary_predictions, zero_division=1)
    f1 = f1_score(y_binary_true, binary_predictions, zero_division=1)
    avg_true_count = sum(true_counts) / len(true_counts)  # Average number of true labels in top-k

    print(f"{model_name} Performance with Top-{k} Neighbors (Binary Decision):")
    print(f"Accuracy (Binary): {accuracy:.4f}")
    print(f"Precision (Binary): {precision:.4f}")
    print(f"Recall (Binary): {recall:.4f}")
    print(f"F1-Score (Binary): {f1:.4f}")
    print(f"Average True Predictions in Top-{k}: {avg_true_count:.4f}")
    print("\n****\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_true_count": avg_true_count,
    }
