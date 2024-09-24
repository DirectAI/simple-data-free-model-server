import os
import shutil
import csv

# Paths
csv_file = "/directai_fastapi/.cache/fairface/black_fairface_train_pred.csv"  # Your CSV file path
train_folder = "/directai_fastapi/.cache/fairface/train"  # Path to the folder containing the images
correct_folder = (
    "/directai_fastapi/.cache/fairface/correct"  # Folder for correct predictions
)
incorrect_folder = (
    "/directai_fastapi/.cache/fairface/incorrect"  # Folder for incorrect predictions
)

# Create the correct and incorrect folders if they do not exist
os.makedirs(correct_folder, exist_ok=True)
os.makedirs(incorrect_folder, exist_ok=True)

# Read the CSV file using the csv module
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    # Skip the header row
    next(reader)

    # Iterate over each row in the CSV
    for row in reader:
        image_name = f"{row[0]}.jpg"  # row[0] is the path column
        image_path = os.path.join(train_folder, image_name)

        # Check if the image exists
        if os.path.exists(image_path):
            is_correct = float(row[3])  # row[3] is the is_correct column

            # Move to correct or incorrect folder based on is_correct value
            if is_correct == 1.0:
                shutil.copy(image_path, os.path.join(correct_folder, image_name))
            elif is_correct == 0.0:
                shutil.copy(image_path, os.path.join(incorrect_folder, image_name))
        else:
            print(f"Image {image_name} not found.")

print("Images have been moved to the correct folders.")
