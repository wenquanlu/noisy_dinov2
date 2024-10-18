import os
import shutil

created_classes = []
files = os.listdir("imagenet-100/test")
for file in files:
    class_id = file[:9]
    if class_id not in created_classes:
        os.mkdir("imagenet-100/test/" + class_id)
        created_classes.append(class_id)
    shutil.move("imagenet-100/test/"+file, "imagenet-100/test/"+class_id + "/")

