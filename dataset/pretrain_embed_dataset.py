import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch
import cv2
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from .constants_val import *
from .utils import get_imgs, get_tokenizer, read_from_dicom, check_element_type
from tqdm import tqdm
from copy import deepcopy
import random
from memory_profiler import profile
from .transforms import DataTransforms, DataFixedTransforms, TextTransform, otsu_mask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class EmbedPretrainingDataset(data.Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        data_pct=1.0,
        imsize=256,
        max_words=72,
        llm_type="gpt",
        simple_cap=False,
        cls_prompt=False,
        train_sub_set=False,
        structural_cap=False,
        natural_cap=False,
        instance_test_cap=False,
        slip=False,
        inter_view=False,
        inter_side=False,
        balanced_test=False,
        prob_diff_dcm=0.5,
        load_jpg=False,
        pred_density=False,
        raw_caption=False,
        mask_ratio=0,
        mask_meta=-1,
        aug_orig_img=False,
        test_data_pct=1.0,
        bootstrap_test=False,
        aug_text=False,
        heavy_aug=False,
        screen_only=False,
        aligned_mlo=False,
        fixed_view=False,
        **kwargs,
    ):
        super().__init__()
        if not os.path.exists(EMBED_DATA_DIR):
            raise RuntimeError(f"{EMBED_DATA_DIR} does not exist!")

        self.llm_type = llm_type
        self.transform = transform
        self.data_pct = data_pct
        self.imsize = imsize
        self.split = split
        self.cls_prompt = cls_prompt
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.instance_test_cap = instance_test_cap
        self.inter_view = inter_view
        self.inter_side = inter_side
        self.balanced_test = balanced_test
        self.slip = slip
        self.prob_diff_dcm = prob_diff_dcm
        self.pred_density = pred_density
        self.raw_caption = raw_caption
        self.load_jpg = load_jpg
        self.mask_ratio = mask_ratio
        self.mask_meta = mask_meta
        self.aug_orig_img = aug_orig_img
        self.test_data_pct = test_data_pct
        self.bootstrap_test = bootstrap_test
        self.screen_only = screen_only
        self.aligned_mlo = aligned_mlo
        self.fixed_view = fixed_view
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        if split == "train":
            self.df = pd.read_csv(EMBED_TRAIN_META_CSV)
        elif split == "valid":
            self.df = pd.read_csv(EMBED_VALID_META_CSV)
        elif split == "test":
            self.df = pd.read_csv(EMBED_TEST_META_CSV)
            self.cls_prompt = True
        else:
            raise ValueError(f"split {split} not supported")
        self.df_anno = pd.read_csv(EMBED_ANNO_CSV_REDUCED)
        self.df_anno_full = pd.read_csv(EMBED_ANNO_CSV)
        df_legends = pd.read_csv(EMBED_LEGENDS_CSV)

        self.massshape_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "massshape"
            ].iterrows()
        }
        self.massdensity_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "massdens"
            ].iterrows()
        }
        self.calcfind_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "calcfind"
            ].iterrows()
        }
        self.calcdistri_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "calcdistri"
            ].iterrows()
        }

        # Only use 2D mammograms for now
        self.df = self.df[self.df[EMBED_IMAGE_TYPE_COL].isin(["2D"])]
        self.df[EMBED_PATH_COL] = self.df[EMBED_PATH_COL].apply(EMBED_PATH_TRANS_FUNC)

        # Only use screening images if screen_only is True
        if screen_only:
            screen_idx = self.df[EMBED_PROCEDURE_COL].apply(
                lambda x: x.lower().find("screen") > 0
            )
            self.df = self.df[screen_idx]
            # Clean up the magnification view and none CC/MLO view
            self.df = self.df[self.df["spot_mag"] != 0]
            self.df = self.df[self.df[EMBED_VIEW_COL].isin(["CC", "MLO"])]
        if fixed_view:
            # ensure always load extra view images
            self.prob_diff_dcm = -1
            self.path2view = {
                row[EMBED_PATH_COL]: row[EMBED_VIEW_COL]
                for _, row in self.df.iterrows()
            }

        if self.structural_cap or self.natural_cap:
            self.max_words = max(144, self.max_words)

        if self.pred_density:
            if split == "train":
                density_file = EMBED_TRAIN_PATH2DENSITY
            elif split == "valid":
                density_file = EMBED_VALID_PATH2DENSITY
            elif split == "test":
                density_file = EMBED_TEST_PATH2DENSITY
            else:
                raise ValueError(f"split {split} not supported")
            assert os.path.exists(density_file)
            self.path2density = pickle.load(open(density_file, "rb"))

        if self.balanced_test:
            if self.pred_density:
                df_path = EMBED_10PCT_DEN_TEST_PATH
                assert os.path.exists(df_path)
                print("### Using balanced test set with 10% test examples...")
                # Note this also contains the density label
                self.balanced_test_path = pickle.load(open(df_path, "rb"))
            else:
                df_path = EMBED_10PCT_TEST_PATH
                assert os.path.exists(df_path)
                print("### Using balanced test set with 10% test examples...")
                self.balanced_test_path = pickle.load(open(df_path, "rb"))
            if self.test_data_pct != 1.0:
                print(f"### Using {self.test_data_pct * 100}% of the test data...")
                sampled_path = random.sample(
                    list(self.balanced_test_path.keys()),
                    int(self.test_data_pct * len(self.balanced_test_path)),
                )
                sampled_path2sent = {
                    k: self.balanced_test_path[k] for k in sampled_path
                }
                self.balanced_test_path = sampled_path2sent
            if self.bootstrap_test:
                print("### Using bootstrap test set...")
                self.balanced_test_path = {
                    k: self.balanced_test_path[k]
                    for k in random.choices(
                        list(self.balanced_test_path.keys()),
                        k=len(self.balanced_test_path),
                    )
                }
        else:
            self.balanced_test_path = None

        self.tokenizer = get_tokenizer(llm_type)

        self.filenames, self.path2sent, self.path2label = self.load_text_data(split)

        if self.inter_view or self.inter_side:
            if self.inter_view:
                if self.screen_only:
                    print(
                        "### Using extra images from other-view (same side, CC/MLO only) for training..."
                    )
                    _ext_img_map = pickle.load(open(EMBED_OTHER_VIEW_MAP, "rb"))
                else:
                    print(
                        "### Using extra images from inter-view (same side) for training..."
                    )
                    _ext_img_map = pickle.load(open(EMBED_INTER_VIEW_MAP, "rb"))
            elif self.inter_side:
                print(
                    "### Using extra images from inter-side (same study) for training..."
                )
                _ext_img_map = pickle.load(open(EMBED_INTER_SIDE_MAP, "rb"))

            # expand user home directory
            self.ext_img_map = {}
            for k, v in _ext_img_map.items():
                self.ext_img_map[os.path.expanduser(k)] = [
                    os.path.expanduser(p) for p in v
                ]
            if self.screen_only:
                # ensure all the ext images keys are in current filenames
                cur_filenames = set(self.filenames)
                ext_filenames = set(self.ext_img_map.keys())
                to_remove = ext_filenames - cur_filenames
                for k in to_remove:
                    del self.ext_img_map[k]
                # ensure all the images in the filenames are in the ext images
                to_add = cur_filenames - ext_filenames
                for k in to_add:
                    self.ext_img_map[k] = []
                if self.fixed_view:
                    # ensure all the images in the filenames are in the ext images
                    cur_filenames = set(self.filenames)
                    ext_filenames = set(self.path2view.keys())
                    self.filenames = list(cur_filenames.intersection(ext_filenames))
                # _view_cnt =
        else:
            self.ext_img_map = None

        self.orig_crop_transform = DataFixedTransforms(True, self.imsize, self.imsize)
        if aug_text:
            if heavy_aug:
                self.text_transform = TextTransform(
                    is_train=(split == "train"),
                    bos_token=self.tokenizer.bos_token,
                    eos_token=self.tokenizer.eos_token,
                    stop_token=self.tokenizer.sep_token,
                    remove_stop_word_prob=0.5,
                    synonym_replacement_prob=0.3,
                    random_swap_prob=0.3,
                    random_deletion_prob=0.3,
                    random_sent_swap_prob=0.3,
                    random_back_translation_prob=0.5,
                )
            else:
                self.text_transform = TextTransform(
                    is_train=(split == "train"),
                    bos_token=self.tokenizer.bos_token,
                    eos_token=self.tokenizer.eos_token,
                    stop_token=self.tokenizer.sep_token,
                )
        else:
            self.text_transform = None

    def load_text_data(self, split):
        base_filename = f"{split}_captions.pickle"
        if self.llm_type != "gpt":
            base_filename = base_filename.replace(".pickle", f"_{self.llm_type}.pickle")
        if self.structural_cap:
            base_filename = base_filename.replace(".pickle", "_structural.pickle")
        elif self.simple_cap:
            base_filename = base_filename.replace(".pickle", "_simple.pickle")
        elif self.natural_cap:
            base_filename = base_filename.replace(".pickle", "_natural.pickle")
        if self.raw_caption:
            base_filename = base_filename.replace(".pickle", "_raw.pickle")
        if self.screen_only:
            base_filename = base_filename.replace(".pickle", "_screen.pickle")
        filepath = os.path.join(EMBED_DATA_DIR, base_filename)

        if not os.path.isfile(filepath):
            print(f"### Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            st = time.time()
            print(f"### Loading captions from {filepath}...")
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
            print(f"### Loaded captions in {time.time() - st:.2} seconds")

        # Some of the paths in the dataframe are not in the captions
        filenames = []
        path2label = {}
        if self.data_pct != 1.0 and split == "train":
            print(f"### Using {self.data_pct * 100}% of the data...")
            sampled_path = random.sample(
                list(path2sent.keys()), int(self.data_pct * len(path2sent))
            )
            sampled_path2sent = {k: path2sent[k] for k in sampled_path}
            path2sent = sampled_path2sent

        print("### extract label from captions...")
        for p, sentences in tqdm(path2sent.items()):
            # Only use the test image from balanced test set during test time
            if (
                self.split == "test"
                and self.balanced_test
                and p not in self.balanced_test_path.keys()
            ):
                continue
            # Extract BI-RAS label from the last sentence
            # print(sentences)
            if self.raw_caption:
                sent = sentences[0].lower().replace("-", "")
            elif self.structural_cap:
                sent = sentences[-2]
            elif self.natural_cap or self.simple_cap or self.structural_cap:
                sent = sentences[-1]
            else:
                sent = sentences[-1].lower().replace("-", "")
            sent = sent.replace("bi rads", "birads")
            assert "birads" in sent
            if self.structural_cap or self.natural_cap:
                birads = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
            else:
                birads = re.findall(r"\bbirads\s\bscore\s(\d+)", sent)[0]
            # skip birads 3 - 6 considering only screening image with 0, 1, 2
            if self.screen_only and int(birads) > 2:
                continue
            if self.pred_density:
                if p not in self.path2density.keys():
                    print(f"### {p} not in density map")
                    continue
                # Ignore male images
                label = self.path2density[p] - 1
                if label == 4:
                    continue
                path2label[p] = label
                filenames.append(p)
            else:
                path2label[p] = int(birads)
                filenames.append(p)
        print(np.unique(list(path2label.values()), return_counts=True))
        return filenames, path2sent, path2label

    def _create_captions_(self, row, meta_only=False, mask_meta=-1):
        target_side = row[EMBED_SIDE_COL]
        anno_row = self.df_anno[self.df_anno[EMBED_SID_COL] == row[EMBED_SID_COL]]
        anno_full_row = self.df_anno_full[
            self.df_anno_full[EMBED_SID_COL] == row[EMBED_SID_COL]
        ]
        # Pick the correct side
        if target_side in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == target_side]
            anno_full_row = anno_full_row[
                anno_full_row[EMBED_FINDING_SIDE_COL] == target_side
            ]
        elif "B" in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            # Pick biliteral result otherwise
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == "B"]
            anno_full_row = anno_full_row[anno_full_row[EMBED_FINDING_SIDE_COL] == "B"]
        try:
            # pick the case with highest BI-RADS
            all_asses = anno_row[EMBED_BIRADS_COL].to_list()
            all_birads = [
                EMBED_LETTER_TO_BIRADS[a]
                for a in all_asses
                if check_element_type(a, EMBED_LETTER_TO_BIRADS.keys())
            ]
            # If all screening image, prefer 0 case
            if np.max(all_birads) <= 2 and np.min(all_birads) == 0:
                idx = np.argmin(all_birads)
            else:
                idx = np.argmax(all_birads)
            anno_row = anno_row.iloc[idx]
            anno_full_row = anno_full_row.iloc[idx]
        except:
            anno_row = anno_row.iloc[0]
            anno_full_row = anno_full_row.iloc[0]
        # use the first annotation

        label_cnt = 0
        # if critical information is missing
        missing_info = False

        if self.structural_cap:
            captions = ""
            procedure = row[EMBED_PROCEDURE_COL]
            if check_element_type(procedure):
                captions += EMBED_PROCEDURE + procedure
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions += EMBED_REASON + reason
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = (
                "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            )
            # Check element type
            age = str(int(age)) if check_element_type(age) else "unknown"
            race = race if check_element_type(race) else "unknown"
            ethnic = ethnic if check_element_type(ethnic) else "unknown"
            # Mask meta information
            if mask_meta > 0:
                age = "unknown" if random.random() < mask_meta else age
                race = "unknown" if random.random() < mask_meta else race
                ethnic = "unknown" if random.random() < mask_meta else ethnic
            # Replace the caption with information
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race)
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic)
            patient_cap = patient_cap.replace("{{AGE}}", age)
            captions += EMBED_PATIENT + patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            # Check element type
            image_type = image_type if check_element_type(image_type) else "unknown"
            side = (
                EMBED_SIDES_DESC[side]
                if check_element_type(side, EMBED_SIDES_DESC.keys())
                else "unknown"
            )
            view = view if check_element_type(view) else "unknown"
            # Mask meta information
            if mask_meta > 0:
                image_type = "unknown" if random.random() < mask_meta else image_type
                side = "unknown" if random.random() < mask_meta else side
                view = "unknown" if random.random() < mask_meta else view
            # Replace the caption with information
            image_cap = EMBED_IMAGE_INFO_CAPTION
            image_cap = image_cap.replace("{{IMAGE_TYPE}}", image_type)
            image_cap = image_cap.replace("{{SIDE}}", side)
            image_cap = image_cap.replace("{{VIEW}}", view)
            captions += EMBED_IMAGE + image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_DENSITY + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                    "{{DENSITY}}", density_desc
                )
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            calc_find = False
            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(
                    shape_code, self.massshape_dict.keys()
                ) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace(
                        "{{SHAPE}}", self.massshape_dict[shape_code]
                    ).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDINGS + EMBED_FINDS_CAPTION + mass_info + " "

                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(
                    calc_find_code, self.calcfind_dict.keys()
                ) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace(
                        "{{SHAPE}}", self.calcfind_dict[calc_find_code]
                    ).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "
                    calc_find = True

                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", impression_desc)

                captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                label_cnt += 1

                assert "{{" not in captions
                # dev
                # if calc_find:
                #     print(captions)
            else:
                missing_info = True
        elif self.natural_cap:
            captions = EMBED_NATURE_BASE_CAPTION

            procedure = row[EMBED_PROCEDURE_COL]
            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions = captions.replace("{{REASON}}", reason)
            else:
                captions = captions.replace("{{REASON}}", "")

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = (
                "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            )
            # Check element type
            age = str(int(age)) if check_element_type(age) else "unknown"
            race = race if check_element_type(race) else "unknown"
            ethnic = ethnic if check_element_type(ethnic) else "unknown"
            # Mask meta information
            if mask_meta > 0:
                age = "unknown" if random.random() < mask_meta else age
                race = "unknown" if random.random() < mask_meta else race
                ethnic = "unknown" if random.random() < mask_meta else ethnic
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race)
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic)
            patient_cap = patient_cap.replace("{{AGE}}", age)
            captions += patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            # Check element type
            image_type = image_type if check_element_type(image_type) else "unknown"
            side = (
                EMBED_SIDES_DESC[side]
                if check_element_type(side, EMBED_SIDES_DESC.keys())
                else "unknown"
            )
            view = view if check_element_type(view) else "unknown"
            # Mask meta information
            if mask_meta > 0:
                image_type = "unknown" if random.random() < mask_meta else image_type
                side = "unknown" if random.random() < mask_meta else side
                view = "unknown" if random.random() < mask_meta else view
            image_cap = EMBED_NATURE_IMAGE_CAPTION
            image_cap = image_cap.replace("{{SIDE}}", side)
            image_cap = image_cap.replace("{{VIEW}}", view)
            captions += image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_BREAST_COMPOSITION_CAPTION.replace(
                    "{{DENSITY}}", density_desc
                )
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(
                    shape_code, self.massshape_dict.keys()
                ) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace(
                        "{{SHAPE}}", self.massshape_dict[shape_code]
                    ).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDS_CAPTION + mass_info + " "

                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(
                    calc_find_code, self.calcfind_dict.keys()
                ) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace(
                        "{{SHAPE}}", self.calcfind_dict[calc_find_code]
                    ).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "

                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", impression_desc)

                assert "{{" not in captions
            else:
                missing_info = True
        else:
            # Start with base caption
            captions = BREAST_BASE_CAPTION

            if not self.simple_cap:
                # provide extra side, view, density information
                side = row[EMBED_SIDE_COL]
                if check_element_type(side, EMBED_SIDES_DESC.keys()):
                    captions += BREAST_SIDE_CAPTION + EMBED_SIDES_DESC[side]
                    captions += " "
                    label_cnt += 1

                view = row[EMBED_VIEW_COL]
                if check_element_type(view):
                    captions += BREAST_VIEW_CAPTION + view
                    captions += " "
                    label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += (
                    BREAST_DENSITY_CAPTION
                    + str(int(density))
                    + ":"
                    + density_desc
                    + "."
                )
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                asses_desc = EMBED_BIRADS_DESC[asses]
                birads = EMBED_LETTER_TO_BIRADS[asses]
                captions += BREAST_BIRADS_CAPTION + str(birads) + ":" + asses_desc + "."
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

        return captions, label_cnt, missing_info

    def create_path_2_sent_mapping(self):
        sent_lens = []
        path2sent = {}
        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # Find annotations for this image
            # Can be more than 1 annotations
            captions, label_cnt, missing_info = self._create_captions_(row)

            # Skip the image if there is no label
            if label_cnt == 0 or missing_info:
                continue

            # use space instead of newline
            captions = captions.replace("\n", " ")

            if self.raw_caption:
                sent_lens.append(len(captions))
                # replace period with sep_token
                # Every sep_token is a new sentence, use for localized loss
                captions = captions.replace(".", " " + self.tokenizer.sep_token)
                captions = captions.replace(";", " " + self.tokenizer.sep_token)

                # add bos/eos tokens
                if self.llm_type != "bert":
                    captions = (
                        self.tokenizer.bos_token
                        + " "
                        + captions
                        + " "
                        + self.tokenizer.eos_token
                    )
                path2sent[row[EMBED_PATH_COL]] = [
                    captions,
                ]
            else:
                # split sentences
                splitter = re.compile("[0-9]+\.")
                captions = splitter.split(captions)
                captions = [point.split(".") for point in captions]
                captions = [sent for point in captions for sent in point]

                cnt = 0
                study_sent = []
                # create tokens from captions
                for cap in captions:
                    if len(cap) == 0:
                        continue

                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) <= 0:
                        continue

                    # filter tokens for current sentence
                    included_tokens = []
                    for t in tokens:
                        t = t.encode("ascii", "ignore").decode("ascii")
                        if len(t) > 0:
                            included_tokens.append(t)

                    if len(included_tokens) > 0:
                        study_sent.append(" ".join(included_tokens))

                    cnt += len(included_tokens)

                if cnt >= 1:
                    sent_lens.append(cnt)
                    path2sent[row[EMBED_PATH_COL]] = study_sent

        sent_lens = np.array(sent_lens)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}] {len(sent_lens)}"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def random_mask(self, tokens, mask_ratio=0.1):
        # Unused
        return tokens
        masked_tokens = deepcopy(tokens)
        length = max(1, masked_tokens.shape[1] - 5)
        for i in range(1, length):
            if masked_tokens[0][i] == self.tokenizer.eos_token_id:
                break

            prob = random.random()
            if prob < mask_ratio:
                masked_tokens[0][i] = self.tokenizer.mask_token_id
        return tokens

    def get_caption(self, path, series_sents=None, mask_ratio=0.0):
        if series_sents is None:
            series_sents = self.path2sent[path]

        if random.random() < mask_ratio:
            target_row = self.df[self.df[EMBED_PATH_COL] == path].iloc[0]
            captions = self._create_captions_(target_row, mask_meta=self.mask_meta)[0]
            captions = captions.replace(".", " " + self.tokenizer.sep_token)
            captions = captions.replace(";", " " + self.tokenizer.sep_token)
            # add bos/eos tokens
            captions = (
                self.tokenizer.bos_token
                + " "
                + captions
                + " "
                + self.tokenizer.eos_token
            )
            series_sents = [
                captions,
            ]
        if self.text_transform is not None:
            series_sents = [
                self.text_transform(series_sents[0]),
            ]

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
        masked_ids = self.random_mask(tokens["input_ids"])
        tokens["masked_ids"] = masked_ids

        return tokens, x_len

    def get_birads_one_hot_label(self, index, get_full=False):
        num_classes = 3 if self.screen_only else len(EMBED_LETTER_TO_BIRADS)
        multi_hot_label = torch.zeros(num_classes)
        key = self.filenames[index]
        asses = self.path2label[key]
        multi_hot_label[asses] = 1
        return multi_hot_label

    def get_density_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_DENSITY_DESC) - 1)
        key = self.filenames[index]
        density = self.path2label[key]
        multi_hot_label[density] = 1
        return multi_hot_label

    def get_bin_mask(self, index, orig_key, tensor_img, ext=False):
        imgs_np = tensor_img.permute(1, 2, 0).numpy()
        # Normalize back to 0-255
        if imgs_np.max() - imgs_np.min() == 0:
            import PIL.Image as Image

            tmp_img = cv2.imread(orig_key, 0)
            tmp_img = Image.fromarray(tmp_img).convert("RGB")
            tmp_img = self.orig_crop_transform(tmp_img)
            # print('\n', tmp_img.max(), tmp_img.min(), '\n')
            assert tmp_img.shape[0] == tensor_img.shape[0]
            assert tmp_img.shape[1] == tensor_img.shape[1]
            if tmp_img.max() - tmp_img.min() == 0:
                print(
                    f"### Warning: Image {index}: {orig_key} : {int(ext)} has no variance"
                )
                imgs_np_cv2 = cv2.normalize(
                    imgs_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                print(imgs_np_cv2.min(), imgs_np_cv2.max())
                print(imgs_np.min(), imgs_np.max())
                print(tensor_img.min(), tensor_img.max())
                cv2.imwrite(f"tmp/preprocess/{index}_transformed_img.jpg", imgs_np_cv2)
                return torch.ones_like(tensor_img)[0, :, :], tensor_img
            tensor_img = tmp_img
            imgs_np = tensor_img.permute(1, 2, 0).numpy()
        # imgs_np = (imgs_np - imgs_np.min()) / (imgs_np.max() - imgs_np.min()) * 255
        imgs_np = cv2.normalize(
            imgs_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        imgs_np = imgs_np.astype(np.uint8)
        bin_mask = (cv2.cvtColor(imgs_np, cv2.COLOR_RGB2GRAY) > 5).astype(
            np.uint8
        ) * 255
        # blur the mask
        bin_mask = cv2.GaussianBlur(bin_mask, (5, 5), 2.0)
        # Boolean mask
        bin_mask = bin_mask > 0
        # if ext:
        #     cv2.imwrite(f'tmp/preprocess/{index}_transformed_img_ext.jpg', imgs_np)
        #     cv2.imwrite(f'tmp/preprocess/{index}_mask_ext.jpg', (bin_mask).astype(np.uint8) * 255)
        # else:
        #     cv2.imwrite(f'tmp/preprocess/{index}_transformed_img.jpg', imgs_np)
        #     cv2.imwrite(f'tmp/preprocess/{index}_mask.jpg', (bin_mask).astype(np.uint8) * 255)
        return (
            torch.tensor(bin_mask, dtype=tensor_img.dtype, device=tensor_img.device),
            tensor_img,
        )

    def __cls_getitem__(self, index):
        key = self.filenames[index]
        if self.pred_density:
            one_hot_label = self.get_density_one_hot_label(index)
        else:
            one_hot_label = self.get_birads_one_hot_label(index)
        if self.zero_shot_caps is None or self.instance_test_cap:
            zero_shot_caps = []
            zero_shot_caps_len = []
            # get base caption
            if self.instance_test_cap:
                target_row = self.df[self.df[EMBED_PATH_COL] == key].iloc[0]
                base_captions = self._create_captions_(target_row, meta_only=True)[0]
            else:
                base_captions = ""
            # get zero-shot captions based on classes
            if self.pred_density:
                for density, density_desc in EMBED_DENSITY_DESC.items():
                    if density == 5:
                        continue
                    # build density caption following training format
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
                        captions = (
                            base_captions
                            + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                                "{{DENSITY}}", density_desc
                            )
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
                    cap, cap_len = self.get_caption(None, [captions])
                    zero_shot_caps.append(cap)
                    zero_shot_caps_len.append(cap_len)
            else:
                target_birads = (
                    EMBED_SCREEN_BIRADS_DESC if self.screen_only else EMBED_BIRADS_DESC
                )
                for asses, birads_desc in target_birads.items():
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
                        impression_desc = target_birads[asses]
                        captions += (
                            EMBED_IMPRESSIONS
                            + EMBED_IMPRESSION_CAPTION.replace(
                                "{{BIRADS}}", str(birads)
                            ).replace("{{BIRADS_DESC}}", impression_desc)
                        )
                        # overall assesment
                        captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                    elif self.natural_cap:
                        # findings
                        mass_info = EMBED_MASS_CAPTION[asses]
                        captions = base_captions + EMBED_FINDS_CAPTION + mass_info + " "
                        # impression
                        impression_desc = target_birads[asses]
                        captions += (
                            EMBED_IMPRESSIONS
                            + EMBED_IMPRESSION_CAPTION.replace(
                                "{{BIRADS}}", str(birads)
                            ).replace("{{BIRADS_DESC}}", impression_desc)
                        )
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
                    cap, cap_len = self.get_caption(None, [captions])
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
        if self.load_jpg:
            key = GET_JPEG_PATH_FUNC(key)
            if self.aligned_mlo:
                aligned_key = GET_ALIGNED_MLO_FUNC(key)
                if os.path.exists(aligned_key):
                    key = aligned_key
            imgs, orig_img = get_imgs(
                key, self.imsize, self.transform, return_orig_img=True
            )
        else:
            imgs, orig_img = read_from_dicom(key, self.imsize, self.transform, True)
        bin_mask, imgs = self.get_bin_mask(index, key, imgs)
        if self.aug_orig_img:
            orig_img = self.transform(orig_img)
        else:
            orig_img = self.orig_crop_transform(orig_img)
        return (
            imgs,
            self.zero_shot_caps,
            self.zero_shot_caps_len,
            key,
            one_hot_label,
            self.zero_shot_caps,
            one_hot_label,
            orig_img,
        )

    def __getitem__(self, index):
        if self.cls_prompt:
            return self.__cls_getitem__(index)
        key = self.filenames[index]
        orig_key = key
        caps, cap_len = self.get_caption(key, mask_ratio=self.mask_ratio)
        # get the same caption with a different text augmentation
        if self.text_transform != None:
            ex_caps, _ = self.get_caption(key, mask_ratio=self.mask_ratio)
        else:
            ex_caps = caps
        if self.load_jpg:
            key = GET_JPEG_PATH_FUNC(key)
            if self.aligned_mlo:
                aligned_key = GET_ALIGNED_MLO_FUNC(key)
                if os.path.exists(aligned_key):
                    key = aligned_key
            imgs, orig_img = get_imgs(
                key, self.imsize, self.transform, return_orig_img=True
            )
        else:
            imgs, orig_img = read_from_dicom(key, self.imsize, self.transform, True)
        bin_mask, imgs = self.get_bin_mask(index, key, imgs)

        if self.slip:
            # Following slip add extra image with different augmentation
            use_other_dcm = random.random() > self.prob_diff_dcm
            if (self.inter_side or self.inter_view) and use_other_dcm:
                # ensure ext img is always within the current dataset
                ext_img_keys = [
                    ext_p
                    for ext_p in self.ext_img_map[orig_key]
                    if ext_p in self.filenames
                ]
                # update the key after cleaning
                self.ext_img_map[orig_key] = ext_img_keys
                if len(ext_img_keys) == 0:
                    ext_imgs = self.transform(orig_img)
                else:
                    ext_img_key = random.choice(ext_img_keys)
                    if self.load_jpg:
                        ext_img_key = GET_JPEG_PATH_FUNC(ext_img_key)
                        if self.aligned_mlo:
                            aligned_ext_img_key = GET_ALIGNED_MLO_FUNC(ext_img_key)
                            if os.path.exists(aligned_ext_img_key):
                                ext_img_key = aligned_ext_img_key
                        ext_imgs = get_imgs(ext_img_key, self.imsize, self.transform)
                    else:
                        ext_imgs = read_from_dicom(
                            ext_img_key, self.imsize, self.transform
                        )
            else:
                ext_imgs = self.transform(orig_img)
            ext_bin_mask, ext_imgs = self.get_bin_mask(index, key, ext_imgs, ext=True)
            if self.fixed_view:
                # The first image is always CC view an the other one is always MLO view
                cur_view = self.path2view[orig_key]
                if cur_view == "CC":
                    imgs = [imgs, ext_imgs]
                else:
                    imgs = [ext_imgs, imgs]
            else:
                imgs = [imgs, ext_imgs]
        if self.pred_density:
            one_hot_label = self.get_density_one_hot_label(index)
        else:
            one_hot_label = self.get_birads_one_hot_label(index)
        orig_img = self.orig_crop_transform(orig_img)

        # No need to unpaired text
        return imgs, caps, cap_len, key, one_hot_label, ex_caps, one_hot_label, orig_img


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention, masked_ids, labels = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    up_ids, up_labels, up_attention = [], [], []
    ext_imgs = []
    orig_imgs = []
    path = []
    tokens_type_id_exist = False
    bin_masks = []
    ext_bin_masks = []
    eval_mode = False
    for b in batch:
        img, cap, cap_l, p, label, up_cap, up_label, orig_img = b
        if isinstance(img, list):
            img, ext_img = img
            ext_imgs.append(ext_img)
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        up_ids.append(up_cap["input_ids"])
        if "token_type_ids" in cap:
            tokens.append(cap["token_type_ids"])
            tokens_type_id_exist = True
        labels.append(label)
        up_labels.append(up_label)
        attention.append(cap["attention_mask"])
        up_attention.append(up_cap["attention_mask"])
        masked_ids.append(cap["masked_ids"])
        path.append(p)
        if isinstance(orig_img, list):
            if len(orig_img) == 2:
                bin_masks.append(orig_img[0])
                ext_bin_masks.append(orig_img[1])
            else:
                bin_masks.append(orig_img[0])
        else:
            orig_imgs.append(orig_img)

    # Pad label to the same length
    max_len = max([len(l) for l in labels])
    for i, l in enumerate(labels):
        if len(l) < max_len:
            labels[i] = torch.nn.functional.pad(l, (0, max_len - len(l)))
    for i, l in enumerate(up_labels):
        if len(l) < max_len:
            up_labels[i] = torch.nn.functional.pad(l, (0, max_len - len(l)))

    # stack
    imgs = torch.stack(imgs)
    ext_imgs = torch.stack(ext_imgs) if len(ext_imgs) > 0 else None
    orig_imgs = torch.stack(orig_imgs) if len(orig_imgs) > 0 else None
    # keep the batch dim
    ids = torch.stack(ids).squeeze(1)
    up_ids = torch.stack(up_ids).squeeze(1)
    if tokens_type_id_exist:
        tokens = torch.stack(tokens).squeeze(1)
    labels = torch.stack(labels).squeeze(1)
    up_labels = torch.stack(up_labels).squeeze(1)
    attention = torch.stack(attention).squeeze(1)
    up_attention = torch.stack(up_attention).squeeze(1)
    masked_ids = torch.stack(masked_ids).squeeze(1)
    bin_masks = torch.stack(bin_masks) if len(bin_masks) > 0 else None
    ext_bin_masks = torch.stack(ext_bin_masks) if len(ext_bin_masks) > 0 else None

    # sort and add to dictionary
    sorted_cap_indices = torch.arange(len(cap_len))
    try:
        sorted_cap_lens = torch.tensor(cap_len)
    except TypeError:
        sorted_cap_lens = torch.stack(cap_len, 0)

    path = np.array(path)
    if len(path) != 1:
        path = path[sorted_cap_indices]
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices] if tokens_type_id_exist else None,
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "masked_ids": masked_ids[sorted_cap_indices],
        "multi_hot_label": labels[sorted_cap_indices],
        "up_caption_ids": up_ids[sorted_cap_indices],
        "up_multi_hot_label": up_labels[sorted_cap_indices],
        "up_attention_mask": up_attention[sorted_cap_indices],
        "orig_imgs": orig_imgs[sorted_cap_indices] if orig_imgs is not None else None,
        "bin_masks": bin_masks[sorted_cap_indices] if bin_masks is not None else None,
        "ext_bin_masks": (
            ext_bin_masks[sorted_cap_indices] if ext_bin_masks is not None else None
        ),
    }
    if ext_imgs is not None:
        return_dict["ext_imgs"] = ext_imgs
    return return_dict
