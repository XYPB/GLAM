import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from .sampler import BalanceSampler


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        collate_fn,
        transforms,
        data_pct,
        batch_size,
        num_workers,
        img_size=224,
        crop_size=224,
        raw_caption=False,
        llm_type="gpt",
        train_split="train",
        valid_split="valid",
        test_split="test",
        train_sub_set=False,
        structural_cap=False,
        simple_cap=False,
        natural_cap=False,
        instance_test_cap=False,
        inter_view=False,
        inter_side=False,
        balanced_test=False,
        slip=False,
        balance_training=False,
        pred_density=False,
        load_jpg=False,
        mask_ratio=0.0,
        mask_meta=-1.0,
        aug_orig_img=False,
        balance_ratio=-1,
        less_train_neg=0,
        test_data_pct=1.0,
        bootstrap_test=False,
        aug_text=False,
        heavy_aug=False,
        pred_only=False,
        max_words=144,
        extra_cap=None,
        prob_diff_dcm=0.5,
        extract_train=False,
        screen_only=False,
        aligned_mlo=False,
        align_orientation=False,
        remove_text=False,
        fixed_view=False,
        pred_mass=False,
        pred_calc=False,
    ):
        super().__init__()

        self.dataset = dataset

        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.crop_size = crop_size
        self.llm_type = llm_type
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.train_sub_set = train_sub_set
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.instance_test_cap = instance_test_cap
        self.inter_view = inter_view
        self.inter_side = inter_side
        self.balanced_test = balanced_test
        self.slip = slip
        self.balance_training = balance_training
        self.pred_density = pred_density
        self.raw_caption = raw_caption
        self.load_jpg = load_jpg
        self.mask_ratio = mask_ratio
        self.mask_meta = mask_meta
        self.aug_orig_img = aug_orig_img
        self.balance_ratio = balance_ratio
        self.less_train_neg = less_train_neg
        self.test_data_pct = test_data_pct
        self.bootstrap_test = bootstrap_test
        self.aug_text = aug_text
        self.heavy_aug = heavy_aug
        self.pred_only = pred_only
        self.max_words = max_words
        self.prob_diff_dcm = prob_diff_dcm
        self.extra_cap = extra_cap
        self.extract_train = extract_train
        self.screen_only = screen_only
        self.aligned_mlo = aligned_mlo
        self.align_orientation = align_orientation
        self.remove_text = remove_text
        self.fixed_view = fixed_view
        self.pred_mass = pred_mass
        self.pred_calc = pred_calc

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(
                True,
                self.img_size,
                self.crop_size,
                self.align_orientation,
                self.remove_text,
            )
        else:
            transform = None

        dataset_kwargs = {
            "split": self.train_split,
            "transform": transform,
            "data_pct": self.data_pct,
            "llm_type": self.llm_type,
            "simple_cap": self.simple_cap,
            "train_sub_set": self.train_sub_set,
            "structural_cap": self.structural_cap,
            "natural_cap": self.natural_cap,
            "instance_test_cap": self.instance_test_cap,
            "inter_side": self.inter_side,
            "inter_view": self.inter_view,
            "balanced_test": self.balanced_test,
            "slip": self.slip,
            "pred_density": self.pred_density,
            "imsize": self.img_size,
            "raw_caption": self.raw_caption,
            "load_jpg": self.load_jpg,
            "mask_ratio": self.mask_ratio,
            "mask_meta": self.mask_meta,
            "aug_orig_img": self.aug_orig_img,
            "less_train_neg": self.less_train_neg,
            "aug_text": self.aug_text,
            "heavy_aug": self.heavy_aug,
            "pred_only": self.pred_only,
            "max_words": self.max_words,
            "prob_diff_dcm": self.prob_diff_dcm,
            "extra_cap": self.extra_cap,
            "screen_only": self.screen_only,
            "aligned_mlo": self.aligned_mlo,
            "fixed_view": self.fixed_view,
            "pred_mass": self.pred_mass,
            "pred_calc": self.pred_calc,
        }

        dataset = self.dataset(**dataset_kwargs)

        if self.balance_training:
            if self.balance_ratio > 0:
                sampler = BalanceSampler(
                    np.array(dataset.labels), ratio=self.balance_ratio
                )
            else:
                num_samples = len(dataset)
                _, class_counts = np.unique(
                    list(dataset.path2label.values()), return_counts=True
                )
                class_weights = 1.0 / class_counts
                weights = []
                for idx in range(num_samples):
                    lb = dataset.path2label[dataset.filenames[idx]]
                    weights.append(class_weights[lb])

                sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(
                False,
                self.img_size,
                self.crop_size,
                self.align_orientation,
                self.remove_text,
            )
        else:
            transform = None

        dataset_kwargs = {
            "split": self.valid_split,
            "transform": transform,
            "data_pct": self.data_pct,
            "llm_type": self.llm_type,
            "simple_cap": self.simple_cap,
            "train_sub_set": self.train_sub_set,
            "structural_cap": self.structural_cap,
            "natural_cap": self.natural_cap,
            "instance_test_cap": self.instance_test_cap,
            "inter_side": self.inter_side,
            "inter_view": self.inter_view,
            "balanced_test": self.balanced_test,
            "slip": self.slip,
            "pred_density": self.pred_density,
            "imsize": self.img_size,
            "raw_caption": self.raw_caption,
            "load_jpg": self.load_jpg,
            "aug_orig_img": self.aug_orig_img,
            "pred_only": self.pred_only,
            "max_words": self.max_words,
            "prob_diff_dcm": self.prob_diff_dcm,
            "screen_only": self.screen_only,
            "aligned_mlo": self.aligned_mlo,
            "fixed_view": self.fixed_view,
            "pred_mass": self.pred_mass,
            "pred_calc": self.pred_calc,
        }

        dataset = self.dataset(**dataset_kwargs)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(
                False,
                self.img_size,
                self.crop_size,
                self.align_orientation,
                self.remove_text,
            )
        else:
            transform = None

        dataset_kwargs = {
            "split": self.test_split,
            "transform": transform,
            "data_pct": self.data_pct,
            "llm_type": self.llm_type,
            "simple_cap": self.simple_cap,
            "train_sub_set": self.train_sub_set,
            "structural_cap": self.structural_cap,
            "natural_cap": self.natural_cap,
            "instance_test_cap": self.instance_test_cap,
            "inter_side": self.inter_side,
            "inter_view": self.inter_view,
            "balanced_test": self.balanced_test,
            "slip": self.slip,
            "pred_density": self.pred_density,
            "imsize": self.img_size,
            "raw_caption": self.raw_caption,
            "load_jpg": self.load_jpg,
            "aug_orig_img": self.aug_orig_img,
            "test_data_pct": self.test_data_pct,
            "bootstrap_test": self.bootstrap_test,
            "pred_only": self.pred_only,
            "max_words": self.max_words,
            "prob_diff_dcm": self.prob_diff_dcm,
            "extract_train": self.extract_train,
            "screen_only": self.screen_only,
            "aligned_mlo": self.aligned_mlo,
            "fixed_view": self.fixed_view,
            "pred_mass": self.pred_mass,
            "pred_calc": self.pred_calc,
        }

        dataset = self.dataset(**dataset_kwargs)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


# if __name__=="__main__":
