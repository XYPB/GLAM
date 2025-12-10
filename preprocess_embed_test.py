import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from .dataset.constants_val import *

print("Processing test set...")
print("Please ensure you have finish the pre-training caption generation step first by running the pre-training script.")


assert os.path.exists("data/train_captions_structural_raw.pickle"), "is False, Please run the pre-training script first"
train_path2captions = pickle.load(open('data/train_captions_structural_raw.pickle', 'rb'))


import re
filenames = []
path2label = {}
path2density = {}
label2path = {}
density2path = {}

density_desc2label = {
    'almost entirely fat': 1,
    'scattered fibroglandular densities': 2,
    'heterogeneously dense': 3,
    'extremely dense': 4,
    "normal male dense": 5
}

for p, sentences in tqdm(train_path2captions.items()):
    sent = sentences[0].lower().replace('-', '')
    birads = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
    for den_sent in density_desc2label.keys():
        if den_sent in sent:
            density_label = density_desc2label[den_sent]
            break
    if density_label not in density2path:
        density2path[density_label] = []
    density2path[density_label].append(p)
    path2density[p] = density_label
    path2label[p] = int(birads)
    if int(birads) not in label2path:
        label2path[int(birads)] = []
    label2path[int(birads)].append(p)
print(np.unique(list(path2label.values()), return_counts=True))
print(np.unique(list(path2density.values()), return_counts=True))

pickle.dump(path2label, open('data/Embed/train_path2label.pickle', 'wb'))
pickle.dump(path2density, open('data/Embed/train_path2density.pickle', 'wb'))


assert os.path.exists("data/valid_captions_structural_raw.pickle"), "is False, Please run the pre-training script first"
valid_path2captions = pickle.load(open('./data/valid_captions_structural_raw.pickle', 'rb'))
filenames = []
path2label = {}
path2density = {}
label2path = {}
density2path = {}

density_desc2label = {
    'almost entirely fat': 1,
    'scattered fibroglandular densities': 2,
    'heterogeneously dense': 3,
    'extremely dense': 4,
    "normal male dense": 5
}

for p, sentences in tqdm(valid_path2captions.items()):
    sent = sentences[0].lower().replace('-', '')
    birads = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
    for den_sent in density_desc2label.keys():
        if den_sent in sent:
            density_label = density_desc2label[den_sent]
            break
    if density_label not in density2path:
        density2path[density_label] = []
    density2path[density_label].append(p)
    path2density[p] = density_label
    path2label[p] = int(birads)
    if int(birads) not in label2path:
        label2path[int(birads)] = []
    label2path[int(birads)].append(p)
print(np.unique(list(path2label.values()), return_counts=True))
print(np.unique(list(path2density.values()), return_counts=True))

pickle.dump(path2label, open('data/Embed/valid_path2label.pickle', 'wb'))
pickle.dump(path2density, open('data/Embed/valid_path2density.pickle', 'wb'))


assert os.path.exists("data/test_captions_structural_raw.pickle"), "is False, Please run the pre-training script first"
test_path2captions = pickle.load(open('./data/test_captions_structural_raw.pickle', 'rb'))
filenames = []
path2label = {}
path2density = {}
label2path = {}
density2path = {}

density_desc2label = {
    'almost entirely fat': 1,
    'scattered fibroglandular densities': 2,
    'heterogeneously dense': 3,
    'extremely dense': 4,
    "normal male dense": 5
}

for p, sentences in tqdm(test_path2captions.items()):
    sent = sentences[0].lower().replace('-', '')
    birads = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
    for den_sent in density_desc2label.keys():
        if den_sent in sent:
            density_label = density_desc2label[den_sent]
            break
    if density_label not in density2path:
        density2path[density_label] = []
    density2path[density_label].append(p)
    path2density[p] = density_label
    path2label[p] = int(birads)
    if int(birads) not in label2path:
        label2path[int(birads)] = []
    label2path[int(birads)].append(p)
print(np.unique(list(path2label.values()), return_counts=True))
print(np.unique(list(path2density.values()), return_counts=True))

pickle.dump(path2label, open('data/Embed/test_path2label.pickle', 'wb'))
pickle.dump(path2density, open('data/Embed/test_path2density.pickle', 'wb'))


test_10pct_path = []
test_10pct_path2label = {}
random.seed(42)
for k in label2path.keys():
    size = int(len(label2path[k]) * 0.1)
    print(k, max(size, 200))
    sampled = random.sample(label2path[k], max(size, 200))
    for p in sampled:
        test_10pct_path2label[p] = path2label[p]
    test_10pct_path.extend(sampled)
pickle.dump(test_10pct_path, open('data/Embed/test_10pct_path.pickle', 'wb'))
pickle.dump(test_10pct_path2label, open('data/Embed/test_10pct_path2label.pickle', 'wb'))


test_10pct_path = []
test_10pct_path2density = {}
random.seed(42)
for k in density2path.keys():
    size = int(len(density2path[k]) * 0.1)
    print(k, max(size, 200))
    sampled = random.sample(density2path[k], max(size, 200))
    for p in sampled:
        test_10pct_path2density[p] = path2density[p]
    test_10pct_path.extend(sampled)
pickle.dump(test_10pct_path, open('data/Embed/test_10pct_path_density.pickle', 'wb'))
pickle.dump(test_10pct_path2density, open('data/Embed/test_10pct_path2density.pickle', 'wb'))