

import pickle
import matplotlib.pyplot as plt
import json
import re

def separate_noise_and_parameter(noise):
    """
    Separates the noise type and parameter from a noise string.

    Args:
        noise (str): A noise string in the format 'typeparameter', e.g., 'gauss50'.

    Returns:
        dict: A dictionary with 'type' and 'parameter' keys.
    """
    match = re.match(r"([a-zA-Z]+)([\d\.]+)", noise)
    if match:
        noise_type = match.group(1)
        parameter = match.group(2)
        return noise_type, parameter


f_shot1 = open("shot1_data.pkl", 'rb')
shot1 = pickle.load(f_shot1)

f_shot3 = open("shot3_data.pkl", 'rb')
shot3 = pickle.load(f_shot3)

f_shot10 = open("shot10_data.pkl", 'rb')
shot10 = pickle.load(f_shot10)

f_extra_shot1 = open("shot1_extra_data.pkl", 'rb')
shot1_extra = pickle.load(f_extra_shot1)
f_extra_shot3 = open("shot3_extra_data.pkl", 'rb')
shot3_extra = pickle.load(f_extra_shot3)
f_extra_shot10 = open("shot10_extra_data.pkl", 'rb')
shot10_extra = pickle.load(f_extra_shot10)

shot1.update(shot1_extra)
shot3.update(shot3_extra)
shot10.update(shot10_extra)
print(shot10)

epochs = [i * 5 for i in range(1, 41)]

def get_general_accuracies(output_dir, n):

    steps = [6249, 12499, 18749, 24999, 31249, 37499, 43749, 49999, 56249, 62499, 68749, 74999, 81249, 87499, 93749, 99999, 106249, 112499, 118749, 124999, 131249, 137499, 143749, 149999, 156249, 162499, 168749, 174999, 181249, 187499, 193749, 199999, 206249, 212499, 218749, 224999, 231249, 237499, 243749, 249999]

    accuracies = []
    for step in steps[:n]:
        max_acc = 0
        f = open(output_dir + "/eval/training_" + str(step) + "/linear/results_eval_linear.json", "r")
        lines = f.readlines()
        for line in lines:
            if line.startswith("iter") or len(line.strip()) == 0:
                continue
            else:
                result = json.loads(line)
                if result["best_classifier"]["accuracy"] > max_acc:
                    max_acc = result["best_classifier"]["accuracy"]
        accuracies.append(max_acc)
        f.close()
    return accuracies

clean = get_general_accuracies("output_clean-200", 40)

def draw_line_plot(dic, noise, clean):
    plt.figure(figsize=(10, 6))
    denoised = dic[f"{noise}_denoised"]
    noisy = dic[f"{noise}_noisy"]
    ours = dic[f"{noise}_ours"]
    noise_type, noise_param = separate_noise_and_parameter(noise)
    if noise_type == "gauss":
        title = f"Gaussian σ = {noise_param}"
    elif noise_type == "shot":
        title = f"Shot λ = {noise_param}"
    elif noise_type == "speckle":
        title = f"Speckle σ = {noise_param * 255}"

    plt.plot(epochs, clean, marker='o', label='Clean')
    plt.plot(epochs, denoised, marker='o', label='Denoised')
    plt.plot(epochs[-12:], ours, marker='o', label='Ours')
    plt.plot(epochs, noisy, marker='o', label='Noisy')
    plt.xlabel('Epochs')
    plt.ylabel('Linear Probing Accuracy')
    plt.title(title)
    #plt.grid(True)
    plt.legend()
    plt.savefig(f"draw/{noise}.jpg")

def gather_noise_acc(noise):
    dic = {}
    dic[f"{noise}_denoised"] = get_general_accuracies(f"output_{noise}-200-denoised", 40)
    dic[f"{noise}_noisy"] = get_general_accuracies(f"output_{noise}-200", 40)
    dic[f"{noise}_ours"] = get_general_accuracies(f"output_{noise}-resume-0-140-200-0-60-60", 12)
    dic[f"{noise}_130"] = get_general_accuracies(f"output_{noise}-resume-0-130-200-0-70-70", 14)
    dic[f"{noise}_120"] = get_general_accuracies(f"output_{noise}-resume-0-120-200-0-80-80", 16)
    return dic

gauss50 = gather_noise_acc("gauss50")
gauss100 = gather_noise_acc("gauss100")
gauss255 = gather_noise_acc("gauss255")
speckle4 = gather_noise_acc("speckle0.4")
speckle7 = gather_noise_acc("speckle0.7")
speckle10 = gather_noise_acc("speckle1.0")

# draw_line_plot(gauss50, "gauss50", clean)
# draw_line_plot(gauss100, "gauss100", clean)
# draw_line_plot(gauss255, "gauss255", clean)
# draw_line_plot(shot10, "shot10", clean)
# draw_line_plot(shot3, "shot3", clean)
# draw_line_plot(shot1, "shot1", clean)
# draw_line_plot(speckle4, "speckle0.4", clean)
# draw_line_plot(speckle7, "speckle0.7", clean)
# draw_line_plot(speckle10, "speckle1.0", clean)

def draw_grid_plot(noise_dicts, noise_labels, clean):
    """
    Draws a 3x3 grid of line plots for different noise types.

    Args:
        noise_dicts (list of dict): A list of dictionaries containing noise accuracies.
        noise_labels (list of str): A list of noise labels corresponding to the dictionaries.
        clean (list): Accuracy values for the clean dataset.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 3x3 grid for easier indexing

    for i, (dic, noise) in enumerate(zip(noise_dicts, noise_labels)):
        denoised = dic[f"{noise}_denoised"]
        noisy = dic[f"{noise}_noisy"]
        ours = dic[f"{noise}_ours"]
        ours_130 = dic[f"{noise}_130"]
        ours_120 = dic[f"{noise}_120"]
        noise_type, noise_param = separate_noise_and_parameter(noise)
        if noise_type == "gauss":
            title = f"Gaussian σ = {noise_param}"
        elif noise_type == "shot":
            title = f"Shot λ = {noise_param}"
        elif noise_type == "speckle":
            title = f"Speckle σ = {float(noise_param) * 255}"

        ax = axes[i]  # Get the specific subplot
        ax.plot(epochs, clean, marker='o', markersize=0, label='Clean')
        ax.plot(epochs, denoised, marker='o', markersize=0, label='Denoised')
        ax.plot(epochs[-12:], ours, marker='o', markersize=0, label='140 Epochs Restart')
        ax.plot(epochs, noisy, marker='o', markersize=0, label='Noisy')
        ax.plot(epochs[-14:], ours_130, marker='o', markersize=0, label='130 Epochs Restart')
        ax.plot(epochs[-16:], ours_120, marker='o', markersize=0, label='120 Epochs Restart')
        ax.set_xlabel('Epochs', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.grid(True)
        ax.legend(fontsize=12.5, framealpha=0)

    # Hide any unused subplots
    for j in range(len(noise_dicts), 9):
        fig.delaxes(axes[j])
    plt.subplots_adjust(hspace=0.3)
    #plt.tight_layout()
    plt.savefig("draw/noise_grid_appendix.png")
    plt.show()

# Gather all noise data
noise_dicts = [
    gauss50, gauss100, gauss255, shot10, shot3, shot1, speckle4, speckle7, speckle10
]
noise_labels = [
    "gauss50", "gauss100", "gauss255", "shot10", "shot3", "shot1", "speckle0.4", "speckle0.7", "speckle1.0"
]

# Call the function to generate the grid plot
draw_grid_plot(noise_dicts, noise_labels, clean)