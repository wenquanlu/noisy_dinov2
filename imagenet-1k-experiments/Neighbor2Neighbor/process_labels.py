f_read = open("labels_preprocessed.txt", "r")
f_write = open("labels.txt", "w")
for line in f_read.readlines():
    class_id = line[:9].strip()
    class_name = line[10:].strip()
    f_write.write(f'{class_id},"{class_name}"\n')


