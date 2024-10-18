import os
f = open("LOC_synset_mapping.txt", "r")
f_w = open("imagenet-100/labels.txt", "w")

for line in f.readlines():
    label = line[10:].strip()
    class_id = line[:9].strip()
    class_subsets = os.listdir("imagenet-100/train")
    if class_id in class_subsets:
        f_w.write('{},"{}"\n'.format(class_id, label))
