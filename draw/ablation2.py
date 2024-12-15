import os
import json
import matplotlib.pyplot as plt

import numpy as np

noise_type = "gauss100"


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

def get_accuracies(output_dir, steps):

    accuracies = []
    for step in steps:
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
steps = [6249, 12499, 18749, 24999, 31249, 37499, 43749, 49999, 56249, 62499, 68749, 74999, 81249, 87499, 93749, 99999, 106249, 112499, 118749, 124999, 131249, 137499, 143749, 149999, 156249, 162499, 168749, 174999, 181249, 187499, 193749, 199999, 206249, 212499, 218749, 224999, 231249, 237499, 243749, 249999]

epochs = [i * 5 for i in range(1, 41)]

gauss50_200 = get_accuracies("output_{}-200".format(noise_type), steps=steps[-18:])
output_noisenoise = get_general_accuracies("output_gauss100-noisy-resume-0-140-200-0-60-60", 12)
output_gauss50_resume_0_140_200_0_60_60 = get_general_accuracies("output_{}-resume-0-140-200-0-60-60".format(noise_type), 12)
no_restart = get_accuracies(f"output_{noise_type}-resume-0-140-200-0-60-60-norestart", steps=steps[-12:])
plt.figure(figsize=(10, 6))

plt.plot(epochs[-12:], output_gauss50_resume_0_140_200_0_60_60, marker='o', label='Ours', color="#2ca02c")
plt.plot(epochs[-12:], no_restart, marker="o", label="Only Denoised Pretraining", color='#9467bd')
plt.plot(epochs[-12:], output_noisenoise, marker="o", label="Only Noisy Training with Restart", color='#8c564b')
plt.plot(epochs[-18:], gauss50_200, marker='o', label='Noisy', color='#d62728')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12.5)
plt.savefig("draw/noise2noise_accuracy_small.png".format(noise_type))