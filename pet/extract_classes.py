import os

files = os.listdir("../data/oxford-pet-dataset/images")

class_dict = {}

for f in files:
    c = ' '.join(f.split('_')[:-1])
    if not c in class_dict:
        class_dict[c] = len(class_dict)

print(class_dict)
