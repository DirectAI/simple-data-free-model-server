import pandas as pd
import os
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

EXPERIMENT_NUMBER = 3
DEFAULT_RAW_FP = f"../.cache/output_v{EXPERIMENT_NUMBER}.csv"
OUTPUT_FP = f"metrics_v{EXPERIMENT_NUMBER}.txt"


def build_confusion_matrix(raw_data_fp=DEFAULT_RAW_FP):
    # Read the CSV file
    df = pd.read_csv(raw_data_fp)

    # Get unique labels
    labels = df["label"].unique()

    # Create confusion matrix
    y_true = df["label"]
    y_pred = df["pred"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm


def write_metrics_to_file(cm, raw_data_fp=DEFAULT_RAW_FP, output_fp=OUTPUT_FP):
    df = pd.read_csv(raw_data_fp)
    labels = df["label"].unique()

    y_true = df["label"]
    y_pred = df["pred"]

    # Calculate overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # Calculate precision, recall, and f1-score for each class
    precision = {}
    recall = {}
    f1_score = {}

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[label] = (
            2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
            if (precision[label] + recall[label]) > 0
            else 0
        )

    # Write metrics to file
    with open(output_fp, "w") as f:
        overall_precision = np.mean(list(precision.values()))
        overall_recall = np.mean(list(recall.values()))

        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Overall Precision: {overall_precision:.4f}\n")
        f.write(f"Overall Recall: {overall_recall:.4f}\n\n")
        f.write("Class-wise Precision, Recall, and F1-Score:\n")
        for label in labels:
            f.write(f"{label}:\n")
            f.write(f"  Precision: {precision[label]:.4f}\n")
            f.write(f"  Recall: {recall[label]:.4f}\n")
            f.write(f"  F1-Score: {f1_score[label]:.4f}\n\n")


def plot_cfm(cm):
    # Create a heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("confusion_matrix.jpg", dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix has been saved as 'confusion_matrix.jpg'")


# Function to get pairs with more than 10 confusions
def get_high_confusion_pairs(confusion_matrix, labels, threshold=10):
    pairs = []
    for i in range(len(labels)):
        for j in range(
            i + 1, len(labels)
        ):  # Start from i+1 to avoid duplicates and self-comparisons
            if confusion_matrix[i][j] > threshold or confusion_matrix[j][i] > threshold:
                pairs.append(
                    (
                        labels[i],
                        labels[j],
                        max(confusion_matrix[i][j], confusion_matrix[j][i]),
                    )
                )
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def print_high_confusion_results(cm, raw_data_fp):
    df = pd.read_csv(raw_data_fp)

    # Get unique labels
    labels = df["label"].unique()
    # Get pairs with high confusion
    cm = build_confusion_matrix()
    high_confusion_pairs = get_high_confusion_pairs(cm, labels)

    # Print the results
    print("\nPairs of classes with more than 10 confusions:")
    for pair in high_confusion_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]} confusions")

    # Optionally, save to a file
    with open("high_confusion_pairs.txt", "w") as f:
        f.write("Pairs of classes with more than 10 confusions:\n")
        for pair in high_confusion_pairs:
            f.write(f"{pair[0]} and {pair[1]}: {pair[2]} confusions\n")

    print("\nHigh confusion pairs have been saved to 'high_confusion_pairs.txt'")


def get_confused_images(class1, class2, df):
    confused_images = df[
        ((df["label"] == class1) & (df["pred"] == class2))
        | ((df["label"] == class2) & (df["pred"] == class1))
    ]
    return confused_images["path"].tolist()


def print_confused_images(class1, class2, df):
    confused_images = get_confused_images(class1, class2, df)
    print(f"\nConfused images between {class1} and {class2}:")
    for filename in confused_images:
        print(filename)


def copy_confused_images_to_review(
    class1, class2, df, review_dir="to_review", raw_data_fp=DEFAULT_RAW_FP
):
    confused_images = get_confused_images(class1, class2, df)

    # Create the review directory if it doesn't exist, and clear it if it does
    if os.path.exists(review_dir):
        shutil.rmtree(review_dir)
    os.makedirs(review_dir)

    print(f"\nCopying confused images between {class1} and {class2} to '{review_dir}':")
    for filename in confused_images:
        # Get the full path of the source file
        actual_label = filename.split("/")[0]
        src_path = os.path.join(
            os.path.dirname(raw_data_fp), "food101_validation_data", filename + ".jpg"
        )

        # Create the destination path
        dest_path = os.path.join(
            review_dir, os.path.basename(f"{filename}_{actual_label}.jpg")
        )

        # Copy the file
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {filename}")

    print(f"\nAll confused images have been copied to '{review_dir}'")


def class_specific_pipeline(raw_data_fp=DEFAULT_RAW_FP, kth_pair=0):
    df = pd.read_csv(raw_data_fp)

    # Get unique labels
    labels = df["label"].unique()
    # Get pairs with high confusion
    cm = build_confusion_matrix()
    write_metrics_to_file(cm)
    high_confusion_pairs = get_high_confusion_pairs(cm, labels)
    # Example usage for the top confused pair
    if high_confusion_pairs:
        top_confused_pair = high_confusion_pairs[kth_pair]
        class1, class2, _ = top_confused_pair
        copy_confused_images_to_review(class1, class2, df)


class_specific_pipeline(kth_pair=0)
