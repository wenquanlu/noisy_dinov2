import json
import numpy as np

noises = ["gauss_50", 
         "gauss_100", 
         "gauss_255", 
         "speckle_0.4", 
         "speckle_0.7", 
         "speckle_1.0", 
         "shot_10",
         "shot_3",
         "shot_1"
         ]

test_datasets = ["roxford5k", "rparis6k"]

steps = [0, 249999]

global_steps = [6249, 12499, 18749, 24999, 31249, 37499, 43749, 49999, 56249, 62499, 68749, 74999, 81249, 87499, 93749, 99999, 106249, 112499, 118749, 124999, 131249, 137499, 143749, 149999, 156249, 162499, 168749, 174999, 181249, 187499, 193749, 199999, 206249, 212499, 218749, 224999, 231249, 237499, 243749, 249999]

data_root = "/home//Workspace/revisitop/data"

log_file = "ic_result.txt"

def get_general_accuracies(output_dir, n):

    accuracies = []
    for step in global_steps[:n]:
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

with open("instance_recog.sh", 'w') as f:
    for noise in noises:
        noise_type = "".join(noise.split("_"))
        for test_dataset in test_datasets:
            for step in steps:
                if step == 249999:
                    noisy_step = step
                    denoised_step = step
                    ours_step = 74999
                else:
                    noisy_accuracies = get_general_accuracies(f"output_{noise_type}-200", 40)
                    noisy_step = global_steps[np.argmax(noisy_accuracies)]
                    denoised_accracies = get_general_accuracies(f"output_{noise_type}-200-denoised", 40)
                    denoised_step = global_steps[np.argmax(denoised_accracies)]
                    ours_accuracies = get_general_accuracies(f"output_{noise_type}-resume-0-140-200-0-60-60", 12)
                    ours_step = global_steps[np.argmax(ours_accuracies)]
                command_noisy = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset={test_dataset} --config-file=output_{noise_type}-200/config.yaml --pretrained-weights=output_{noise_type}-200/eval/training_{noisy_step}/teacher_checkpoint.pth --noise={noise} --log_file={log_file})"
                command_denoised = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset={test_dataset} --config-file=output_{noise_type}-200-denoised/config.yaml --pretrained-weights=output_{noise_type}-200-denoised/eval/training_{denoised_step}/teacher_checkpoint.pth --noise={noise} --log_file={log_file})"
                command_ours = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset={test_dataset} --config-file=output_{noise_type}-resume-0-140-200-0-60-60/config.yaml --pretrained-weights=output_{noise_type}-resume-0-140-200-0-60-60/eval/training_{ours_step}/teacher_checkpoint.pth --noise={noise} --log_file={log_file})"
                f.write(command_noisy)
                f.write("\n")
                f.write(command_denoised)
                f.write("\n")
                f.write(command_ours)
                f.write("\n")

    clean_step = global_steps[np.argmax(get_general_accuracies("output_clean-200", 40))]
    command_oxford_clean_best = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset=roxford5k --config-file=output_clean-200/config.yaml --pretrained-weights=output_clean-200/eval/training_{clean_step}/teacher_checkpoint.pth --noise=identity_0 --log_file={log_file})"
    command_oxford_clean_final = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset=roxford5k --config-file=output_clean-200/config.yaml --pretrained-weights=output_clean-200/eval/training_249999/teacher_checkpoint.pth --noise=identity_0 --log_file={log_file})"
    command_paris_clean_best = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset=rparis6k --config-file=output_clean-200/config.yaml --pretrained-weights=output_clean-200/eval/training_{clean_step}/teacher_checkpoint.pth --noise=identity_0 --log_file={log_file})"
    command_paris_clean_final = f"JOB_ID=$(python dinov2/eval/instance_recog.py --data_root={data_root} --test_dataset=rparis6k --config-file=output_clean-200/config.yaml --pretrained-weights=output_clean-200/eval/training_249999/teacher_checkpoint.pth --noise=identity_0 --log_file={log_file})"
    f.write(command_oxford_clean_best)
    f.write("\n")
    f.write(command_oxford_clean_final)
    f.write("\n")
    f.write(command_paris_clean_best)
    f.write("\n")
    f.write(command_paris_clean_final)
    f.write("\n")

