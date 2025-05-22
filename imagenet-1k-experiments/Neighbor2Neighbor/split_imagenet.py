import os
from pathlib import Path
from tqdm import tqdm

def move_class(class_dir, dest):
    """
    Move a class directory to the destination while avoiding duplicates.
    """
    if not dest.exists():  # Only move if the destination doesn't already exist
        class_dir.rename(dest)

def split_imagenet_folder(input_folder, output_base_folder, parts):
    """
    Split an ImageNet-like dataset into specified parts while maintaining the folder structure.

    Parameters:
        input_folder (str): Path to the input ImageNet folder.
        output_base_folder (str): Base folder to save the split parts.
        parts (int): Number of parts to split into.
    """
    # Ensure the input folder exists
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder '{input_folder}' does not exist.")

    # Get train and val directories
    train_path = input_path / "train"
    val_path = input_path / "val"

    if not train_path.exists() or not val_path.exists():
        raise ValueError("Input folder must contain 'train' and 'val' subfolders.")

    # Get all class directories from train and val
    train_class_dirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    val_class_dirs = sorted([d for d in val_path.iterdir() if d.is_dir()])

    # Check that the classes can be evenly split
    num_classes = len(train_class_dirs)
    if num_classes != len(val_class_dirs):
        raise ValueError("Mismatch between number of classes in 'train' and 'val'.")

    if num_classes % parts != 0:
        raise ValueError(f"Number of classes ({num_classes}) is not evenly divisible by {parts}.")

    classes_per_part = num_classes // parts

    # Create output folders and distribute classes
    for part in range(1, parts + 1):
        part_folder = Path(output_base_folder) / f"imagenet-gauss100-part{part}"
        train_folder = part_folder / "train"
        val_folder = part_folder / "val"

        # Create train and val directories
        train_folder.mkdir(parents=True, exist_ok=True)
        val_folder.mkdir(parents=True, exist_ok=True)

        # Select classes for this part
        start_idx = (part - 1) * classes_per_part
        end_idx = part * classes_per_part
        part_train_classes = train_class_dirs[start_idx:end_idx]
        part_val_classes = val_class_dirs[start_idx:end_idx]

        # Move train classes
        for class_dir in tqdm(part_train_classes, desc=f"Moving train classes to Part {part}", unit="class"):
            move_class(class_dir, train_folder / class_dir.name)

        # Move val classes
        for class_dir in tqdm(part_val_classes, desc=f"Moving val classes to Part {part}", unit="class"):
            move_class(class_dir, val_folder / class_dir.name)

if __name__ == "__main__":
    input_folder = "imagenet-gauss100"  # Path to the input dataset
    output_base_folder = "./"  # Path to save the output parts
    parts = 4  # Number of parts to split into

    split_imagenet_folder(input_folder, output_base_folder, parts)
    print("Dataset split completed.")
