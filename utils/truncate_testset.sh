#!/bin/bash

# Define path to the train directory
train_dir="imagenet-100/train"

# Loop through each category directory in train
for category in "$train_dir"/*; do
  if [ -d "$category" ]; then
    # List all files, sort them, and keep the first 400 files
    keep_files=$(ls "$category" | sort | head -400)

    # Find all files in the directory and delete those not in the list of first 400
    for image in "$category"/*; do
      # Get the base name of the image file
      image_name=$(basename "$image")

      # Check if the image is in the keep_files list
      if ! echo "$keep_files" | grep -q "$image_name"; then
        # If not in the list, delete the image
        rm "$image"
      fi
    done
  fi
done

echo "Non-first 400 images deleted successfully."
