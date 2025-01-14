import torch

# Load the original state dictionary
state_dict = torch.load("output_shot1-200-denoised/model_0174999.rank_0.pth")['model']

# Initialize new top-level dictionaries
new_state_dict = {
    "denoised_teacher": {},
    "dino_loss_denoised": {},
    "ibot_patch_loss_denoised": {}
}

# Split keys into respective top-level dictionaries
for key, value in state_dict.items():
    if key.startswith("teacher"):
        new_key = key.replace("teacher.", "", 1)  # Remove the prefix 'teacher.'
        new_state_dict["denoised_teacher"][new_key] = value
    elif key.startswith("dino_loss"):
        new_key = key.replace("dino_loss.", "", 1)  # Remove the prefix 'dino_loss.'
        new_state_dict["dino_loss_denoised"][new_key] = value
    elif key.startswith("ibot_patch_loss"):
        new_key = key.replace("ibot_patch_loss.", "", 1)  # Remove the prefix 'ibot_patch_loss.'
        new_state_dict["ibot_patch_loss_denoised"][new_key] = value

# Verify the new top-level keys and their sub-keys
print("Top-level keys in new_state_dict:")
for top_key in new_state_dict.keys():
    print(f"{top_key}: {len(new_state_dict[top_key])} keys")

# Save the modified state dictionary
torch.save(new_state_dict, "output_shot1-200-denoised/denoised_teacher_0174999.pth")

print("New state dictionary saved successfully.")
