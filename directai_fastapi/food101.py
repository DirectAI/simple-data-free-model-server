import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json


# Define the path where you want to save the test dataset
save_path = "food101_test_data"
os.makedirs(save_path, exist_ok=True)


# Function to save images to disk in subdirectories based on label
def save_images_to_disk(save_directory):
    # Load only the test (validation) dataset
    dataset_name = "food101"
    test_ds, info = tfds.load(
        dataset_name, split="validation", as_supervised=True, with_info=True
    )

    # Get the class names from the dataset info
    class_names = info.features["label"].names
    for i, (image, label) in enumerate(test_ds):
        # Convert the image to uint8 type and save it as a JPEG file
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # Get the label name
        label_name = class_names[label.numpy()]

        # Create subdirectory for the label if it doesn't exist
        label_dir = os.path.join(save_directory, label_name)
        os.makedirs(label_dir, exist_ok=True)

        # Save the image to the appropriate subdirectory
        image_path = os.path.join(label_dir, f"image_{i}.jpg")
        tf.io.write_file(image_path, tf.image.encode_jpeg(image))

        # Print progress every 100 images
        if i % 100 == 0:
            print(f"Saved {i} images to {label_dir}.")


label_names = [
    folder.replace("_", " ")
    for folder in os.listdir(save_path)
    if os.path.isdir(os.path.join(save_path, folder))
]
classifier_config = {
    "classifier_configs": [
        {
            "name": folderw,
            "examples_to_include": [folder.replace("_", " ")],
            "examples_to_exclude": [],
        }
        for folder in os.listdir(save_path)
        if os.path.isdir(os.path.join(save_path, folder))
    ]
}


with open("classifier_config.json", "w") as json_file:
    json.dump(classifier_config, json_file, indent=4)

print("classifier_config has been saved to classifier_config.json")


# Save the test dataset to disk
# save_images_to_disk(save_path)

print("Test dataset downloaded and saved to disk in subdirectories by label.")


folder_names = [
    folder.replace("_", " ")
    for folder in os.listdir(save_path)
    if os.path.isdir(os.path.join(save_path, folder))
]
print(folder_names)
