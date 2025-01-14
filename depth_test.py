from dinov2.hub.depthers import _make_dinov2_dpt_depther

import torch

depth_model = _make_dinov2_dpt_depther(arch_name="vit_small", pretrained=False, weights=None, depth_range=(1, 255))

state_dict = torch.load("output_clean-200/eval/training_249999/teacher_checkpoint.pth")["teacher"]

new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("backbone"):
        new_key = key.replace("backbone.", "", 1)  # Remove the prefix 'teacher.'
        new_state_dict[new_key] = value

depth_model.backbone.load_state_dict(new_state_dict)

#dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
