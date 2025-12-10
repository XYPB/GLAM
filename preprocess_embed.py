import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from .dataset.constants_val import *

df_anno = pd.read_csv("data/tables/EMBED_OpenData_clinical_reduced.csv")
df_anno_patho = pd.read_csv("data/tables/EMBED_OpenData_clinical.csv")
df_meta = pd.read_csv("data/tables/EMBED_OpenData_metadata_reduced.csv")
df_meta = df_meta.drop("Unnamed: 0", axis=1)


# Find inter-view/inter-side images
img_path2same_case = {}
img_path2same_side = {}
same_case_cnt = []
same_side_cnt = []
for i, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
    sid = row[EMBED_SID_COL]
    side = row[EMBED_SIDE_COL]
    cur_path = EMBED_PATH_TRANS_FUNC(row[EMBED_PATH_COL])

    same_study_df = df_meta[df_meta[EMBED_SID_COL] == sid]
    same_study_p = [
        EMBED_PATH_TRANS_FUNC(p) for p in same_study_df[EMBED_PATH_COL].tolist()
    ]
    assert cur_path in same_study_p
    same_study_p.remove(cur_path)
    img_path2same_case[cur_path] = same_study_p
    same_case_cnt.append(len(same_study_p))

    same_side_df = same_study_df[same_study_df[EMBED_SIDE_COL] == side]
    # print(same_side_df[EMBED_VIEW_COL], row[EMBED_VIEW_COL])
    same_side_p = [
        EMBED_PATH_TRANS_FUNC(p) for p in same_side_df[EMBED_PATH_COL].tolist()
    ]
    assert cur_path in same_side_p
    same_side_p.remove(cur_path)
    # print(cur_path, same_side_p)
    img_path2same_side[cur_path] = same_side_p
    same_side_cnt.append(len(same_side_p))

pickle.dump(img_path2same_case, open("data/tables/img_path2inter_side.pkl", "wb"))
pickle.dump(img_path2same_side, open("data/tables/img_path2inter_view.pkl", "wb"))


to_other_view = {}
for i, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
    sid = row[EMBED_SID_COL]
    side = row[EMBED_SIDE_COL]
    cur_path = EMBED_PATH_TRANS_FUNC(row[EMBED_PATH_COL])
    cur_view = row[EMBED_VIEW_COL]
    
    same_study_df = df_meta[df_meta[EMBED_SID_COL] == sid]
    same_side_df = same_study_df[same_study_df[EMBED_SIDE_COL] == side]
    different_view_df = same_side_df[same_side_df[EMBED_VIEW_COL] != cur_view]
    views = different_view_df[EMBED_VIEW_COL].to_list()
    same_side_p = [EMBED_PATH_TRANS_FUNC(p) for p in different_view_df[EMBED_PATH_COL].tolist()]

    to_other_view[cur_path] = same_side_p

pickle.dump(to_other_view, open('data/Embed/tables/img_path2other_view_ccmlo.pkl', 'wb'))


# Only consider 2D images
df_meta = df_meta[df_meta[EMBED_IMAGE_TYPE_COL] == "2D"]

unique_patients = list(set(df_meta["empi_anon"].tolist()))
unique_exams_cnt_all = Counter(df_meta["acc_anon"].tolist())
patient2exam_all = {
    patient: df_meta[df_meta["empi_anon"] == patient]["acc_anon"].tolist()
    for patient in unique_patients
}
exam2birads_all = {
    exam: df_anno[df_anno["acc_anon"] == exam]["asses"].tolist()
    for exam in unique_exams_cnt_all.keys()
}

exam2patho_all = {
    exam: (
        df_anno_patho[df_anno_patho["acc_anon"] == exam]["path_severity"].tolist(),
        df_anno_patho[df_anno_patho["acc_anon"] == exam]["side"].tolist(),
    )
    for exam in unique_exams_cnt_all.keys()
}
exam2patho_all = {k: [] if np.isnan(v[0][0]) else v for k, v in exam2patho_all.items()}

unique_patients = list(set(df_meta["empi_anon"].tolist()))
unique_exams_cnt = Counter(df_meta["acc_anon"].tolist())
unique_images = list(set(df_meta["anon_dicom_path"].tolist()))
patient2exam = {
    patient: df_meta[df_meta["empi_anon"] == patient]["acc_anon"].tolist()
    for patient in unique_patients
}
patient2img = {
    patient: df_meta[df_meta["empi_anon"] == patient]["anon_dicom_path"].tolist()
    for patient in unique_patients
}
exam2view = {
    exam: df_meta[df_meta["acc_anon"] == exam]["ViewPosition"].tolist()
    for exam in unique_exams_cnt.keys()
}
exam2side = {
    exam: df_meta[df_meta["acc_anon"] == exam]["ImageLateralityFinal"].tolist()
    for exam in unique_exams_cnt.keys()
}
exam2density = {
    exam: df_anno[df_anno["acc_anon"] == exam]["tissueden"].tolist()
    for exam in unique_exams_cnt.keys()
}
exam2birads = {
    exam: df_anno[df_anno["acc_anon"] == exam]["asses"].tolist()
    for exam in unique_exams_cnt.keys()
}

total_rows = len(unique_patients)
first_split = int(total_rows * 0.7)
second_split = first_split + int(total_rows * 0.2)
shuffle_patient = random.sample(unique_patients, len(unique_patients))
patient_train = shuffle_patient[:first_split]
patient_test = shuffle_patient[first_split:second_split]
patient_val = shuffle_patient[second_split:]

df_train = df_meta[df_meta["empi_anon"].isin(patient_train)]
df_test = df_meta[df_meta["empi_anon"].isin(patient_test)]
df_val = df_meta[df_meta["empi_anon"].isin(patient_val)]
print(len(df_train), len(df_test), len(df_val))

df_train.to_csv("data/tables/EMBED_OpenData_metadata_reduced_train.csv", index=False)
df_test.to_csv("data/tables/EMBED_OpenData_metadata_reduced_test.csv", index=False)
df_val.to_csv("data/tables/EMBED_OpenData_metadata_reduced_valid.csv", index=False)

