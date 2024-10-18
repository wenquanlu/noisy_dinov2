#!/bin/bash

# Define paths to the train and test directories
train_dir="imagenet-100/train"
test_dir="imagenet-100/test"

# Loop through each category directory in train
for category in $train_dir/*; do
  if [ -d "$category" ]; then
    # Move the last 100 images from train to test
    # `tail -100` gives the last 100 entries from the list of sorted files
    ls "$category" | tail -100 | while read image; do
      mv "$category/$image" "$test_dir"
    done
  fi
done

echo "Images moved successfully."
