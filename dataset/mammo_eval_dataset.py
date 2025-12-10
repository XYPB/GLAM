import torch
import time
import warnings
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
from collections import Counter
from glob import glob
import torchvision.transforms as transforms
import random
import pydicom as dicom
from .transforms import OtsuCut
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from .constants_val import *
from .utils import get_imgs, read_from_dicom, get_tokenizer


class VinDr(torch.utils.data.Dataset):

    def __init__(
        self,
        split="train",
        transform=None,
        imsize=1024,
        data_pct=1.0,
        llm_type="gpt",
        pred_density=False,
        pred_mass=False,
        pred_calc=False,
        uniform_norm=False,
        max_words=64,
        structural_cap=False,
        natural_cap=False,
        simple_cap=False,
        raw_caption=False,
        load_jpg=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.df = pd.read_csv(VINDR_CSV_DIR)
        self.data_path = VINDR_IMAGE_DIR
        self.transform = transform
        self.imsize = imsize
        self.uniform_norm = uniform_norm
        self.pred_density = pred_density
        self.pred_mass = pred_mass
        self.pred_calc = pred_calc
        self.llm_type = llm_type
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.natural_cap = natural_cap
        self.simple_cap = simple_cap
        self.raw_caption = raw_caption
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.load_jpg = load_jpg
        self.n_classes = 5 if not self.pred_density else 4
        if split == "test":
            self.df = self.df[self.df["split"] == "test"]
        else:
            self.df = self.df[self.df["split"] == "training"]
        self.findings_df = pd.read_csv(VINDR_DET_CSV_DIR)

        if data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=data_pct)

        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.labels = []
        self.path2label = {}
        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            if self.pred_density:
                label = entry["breast_density"].split(" ")[-1]
                label = VINDR_DENSITY_LETTER2DIGIT[label]
            elif self.pred_mass or self.pred_calc:
                image_id = entry["image_id"]
                findings = self.findings_df[self.findings_df["image_id"] == image_id][
                    "finding_categories"
                ]
                findings_list = findings.to_list()
                findings_str = " ".join(findings_list)
                if self.pred_mass:
                    label = 2 if "Mass" in findings_str else 1
                else:
                    label = 2 if "Suspicious Calcification" in findings_str else 1
            else:
                # BIRADS 1 ~ 5
                label = int(entry["breast_birads"].split(" ")[-1])
            sid = entry["study_id"]
            imid = entry["image_id"]
            dicom_path = os.path.join(self.data_path, f"{sid}/{imid}.dicom")
            self.filenames.append(dicom_path)
            self.labels.append(label - 1)
            self.path2label[dicom_path] = label - 1
        print("### Sampled split distribution: ", Counter(self.labels))

    def __len__(self):
        return len(self.df)

    def get_caption(self, series_sents):
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        tokens["masked_ids"] = tokens["input_ids"]

        return tokens, x_len

    def get_zeroshot_caption(self):
        base_captions = ""
        zero_shot_caps = []
        zero_shot_caps_len = []
        if self.pred_density:
            for density, density_desc in EMBED_DENSITY_DESC.items():
                if density == 5:
                    continue
                if self.structural_cap:
                    density_desc = EMBED_DENSITY_DESC[density]
                    captions = (
                        base_captions
                        + EMBED_DENSITY
                        + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                            "{{DENSITY}}", density_desc
                        )
                    )
                    if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                        captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                elif self.natural_cap:
                    density_desc = EMBED_DENSITY_DESC[density]
                    captions = base_captions + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                        "{{DENSITY}}", density_desc
                    )
                    if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                        captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                else:
                    captions = (
                        base_captions
                        + BREAST_BASE_CAPTION
                        + BREAST_DENSITY_CAPTION
                        + str(density)
                        + ": "
                        + density_desc
                        + "."
                    )
                # Update caption type if using raw style caption
                if self.raw_caption:
                    captions = captions.replace(".", " " + self.tokenizer.sep_token)
                    captions = captions.replace(";", " " + self.tokenizer.sep_token)
                    if self.llm_type != "bert":
                        captions = (
                            self.tokenizer.bos_token
                            + " "
                            + captions
                            + " "
                            + self.tokenizer.eos_token
                        )
                else:
                    captions = captions.replace("\n", " ").lower()
                cap, cap_len = self.get_caption([captions])
                zero_shot_caps.append(cap)
                zero_shot_caps_len.append(cap_len)
        else:
            for digits in range(0, 5):
                asses = VINDR_BIRADS_DIGIT2LETTER[digits]
                birads_desc = EMBED_BIRADS_DESC[asses]
                # VinDr only consider BIRADS 1 ~ 5
                if asses in ["A", "K"]:
                    continue
                birads = EMBED_LETTER_TO_BIRADS[asses]
                # build density caption following training format
                if self.structural_cap:
                    # findings
                    mass_info = EMBED_MASS_CAPTION[asses]
                    captions = (
                        base_captions
                        + EMBED_FINDINGS
                        + EMBED_FINDS_CAPTION
                        + mass_info
                        + " "
                    )
                    # impression
                    impression_desc = EMBED_BIRADS_DESC[asses]
                    captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                        "{{BIRADS}}", str(birads)
                    ).replace("{{BIRADS_DESC}}", impression_desc)
                    # overall assesment
                    captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                elif self.natural_cap:
                    # findings
                    mass_info = EMBED_MASS_CAPTION[asses]
                    captions = base_captions + EMBED_FINDS_CAPTION + mass_info + " "
                    # impression
                    impression_desc = EMBED_BIRADS_DESC[asses]
                    captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                        "{{BIRADS}}", str(birads)
                    ).replace("{{BIRADS_DESC}}", impression_desc)
                else:
                    captions = (
                        base_captions
                        + BREAST_BASE_CAPTION
                        + BREAST_BIRADS_CAPTION
                        + str(birads)
                        + ": "
                        + birads_desc
                        + "."
                    )
                # Update caption type if using raw style caption
                if self.raw_caption:
                    captions = captions.replace(".", " " + self.tokenizer.sep_token)
                    captions = captions.replace(";", " " + self.tokenizer.sep_token)
                    if self.llm_type != "bert":
                        captions = (
                            self.tokenizer.bos_token
                            + " "
                            + captions
                            + " "
                            + self.tokenizer.eos_token
                        )
                else:
                    captions = captions.replace("\n", " ").lower()
                cap, cap_len = self.get_caption([captions])
                zero_shot_caps.append(cap)
                zero_shot_caps_len.append(cap_len)

        stacked_caps = {}
        for cap in zero_shot_caps:
            for k, v in cap.items():
                if k not in stacked_caps:
                    stacked_caps[k] = v
                else:
                    stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
        zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
        self.zero_shot_caps = stacked_caps
        self.zero_shot_caps_len = zero_shot_caps_len

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        sid = entry["study_id"]
        imid = entry["image_id"]
        view = entry["laterality"] + entry["view_position"]
        label = self.labels[idx]
        dicom_path = os.path.join(self.data_path, f"{sid}/{imid}.dicom")

        if self.load_jpg:
            img_path = dicom_path.replace("vindr-1.0.0", "vindr-1.0.0-resized-1024")
            img_path = img_path.replace(".dicom", "_resized.png")
            assert os.path.exists(img_path)
            img = get_imgs(img_path, scale=self.imsize, transform=self.transform)
        else:
            assert os.path.exists(dicom_path)
            img = read_from_dicom(dicom_path, transform=self.transform)
        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1
        if self.zero_shot_caps is None:
            self.get_zeroshot_caption()

        return (
            img,
            self.zero_shot_caps,
            self.zero_shot_caps_len,
            dicom_path,
            one_hot_labels,
            self.zero_shot_caps,
            one_hot_labels,
            img,
        )


class RSNAMammo(torch.utils.data.Dataset):

    def __init__(
        self,
        split="train",
        transform=None,
        data_pct=1.0,
        llm_type="gpt",
        max_words=72,
        img_size=1024,
        structural_cap=False,
        natural_cap=False,
        simple_cap=False,
        balanced_test=False,
        raw_caption=False,
        less_train_neg=0,
        *args,
        **kwargs,
    ):
        if split == "test":
            split = "valid"
        assert split in ["train", "valid"]
        self.transform = transform
        self.llm_type = llm_type
        self.img_size = img_size
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.natural_cap = natural_cap
        self.simple_cap = simple_cap
        self.raw_caption = raw_caption
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.n_classes = 2

        if split == "train":
            self.df = pd.read_csv(RSNA_MAMMO_TRAIN_CSV)
        else:
            if balanced_test:
                warnings.warn(
                    "Balanced test set is not supported for RSNA Mammography dataset, move to original test set"
                )
                self.df = pd.read_csv(RSNA_MAMMO_TEST_CSV)
                # self.df = pd.read_csv(RSNA_MAMMO_BALANCE_TEST_CSV)
            else:
                self.df = pd.read_csv(RSNA_MAMMO_TEST_CSV)

        if data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=data_pct)
        if less_train_neg > 0:
            df_neg = self.df[self.df["cancer"] == 0]
            df_pos = self.df[self.df["cancer"] == 1]
            df_neg = df_neg.sample(frac=less_train_neg)
            self.df = pd.concat([df_neg, df_pos])

        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.path2label = {}
        self.labels = []
        missing_cnt = 0
        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            label = entry["cancer"]
            pid, iid = entry["patient_id"], entry["image_id"]
            path = os.path.join(RSNA_MAMMO_JPEG_DIR, f"{pid}/{iid}_resized.jpg")
            if not os.path.exists(path):
                missing_cnt += 1
            self.labels.append(label)
            self.filenames.append(path)
            self.path2label[path] = label
        print("### Sampled split distribution: ", Counter(self.labels))

    def __len__(self):
        return len(self.df)

    def get_caption(self, series_sents):
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        tokens["masked_ids"] = tokens["input_ids"]

        return tokens, x_len

    def get_zeroshot_caption(self):
        base_captions = ""
        zero_shot_caps = []
        zero_shot_caps_len = []
        for label, canncer_desc in RSNA_MAMMO_CANCER_DESC.items():
            # build density caption following training format
            birads, birads_desc = RSNA_MAMMO_BIRADS_DESC[label]
            if self.structural_cap:
                # findings
                captions = (
                    base_captions
                    + EMBED_FINDINGS
                    + EMBED_FINDS_CAPTION
                    + canncer_desc
                    + " "
                )
                # impression
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", birads
                ).replace("{{BIRADS_DESC}}", birads_desc)
                # overall assesment
                captions += EMBED_ASSESSMENT + birads_desc
            elif self.natural_cap:
                # findings
                captions = base_captions + EMBED_FINDS_CAPTION + canncer_desc + " "
                # impression
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", birads_desc)
            else:
                captions = (
                    base_captions
                    + BREAST_BASE_CAPTION
                    + BREAST_BIRADS_CAPTION
                    + str(birads)
                    + ": "
                    + birads_desc
                    + "."
                )
            # Update caption type if using raw style caption
            if self.raw_caption:
                captions = captions.replace(".", " " + self.tokenizer.sep_token)
                captions = captions.replace(";", " " + self.tokenizer.sep_token)
                if self.llm_type != "bert":
                    captions = (
                        self.tokenizer.bos_token
                        + " "
                        + captions
                        + " "
                        + self.tokenizer.eos_token
                    )
            else:
                captions = captions.replace("\n", " ").lower()
            cap, cap_len = self.get_caption([captions])
            zero_shot_caps.append(cap)
            zero_shot_caps_len.append(cap_len)

        stacked_caps = {}
        for cap in zero_shot_caps:
            for k, v in cap.items():
                if k not in stacked_caps:
                    stacked_caps[k] = v
                else:
                    stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
        zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
        self.zero_shot_caps = stacked_caps
        self.zero_shot_caps_len = zero_shot_caps_len

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        label = self.labels[idx]
        path = self.filenames[idx]

        img = get_imgs(path, scale=self.img_size, transform=self.transform)
        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1
        if self.zero_shot_caps is None:
            self.get_zeroshot_caption()

        return (
            img,
            self.zero_shot_caps,
            self.zero_shot_caps_len,
            path,
            one_hot_labels,
            self.zero_shot_caps,
            one_hot_labels,
            img,
        )


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            OtsuCut(),
            transforms.Resize((512, 512)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1550), (0.1521)),
        ]
    )
    dataset = VinDr(
        "data\csv\breast-level_annotations.csv",
        transform=transform,
        binary=False,
        test=False,
    )

    img, label = dataset.__getitem__(10)
    print(torch.mean(img), label)
    plt.imsave("./tmp/vindr_img.jpg", img.squeeze().numpy())
