# Geometry-Guided Local Alignment for Multi-View Visual Language Pre-Training in Mammography

#### By *[Yuexi Du](https://xypb.github.io/), Lihui Chen, and [Nicha C. Dvornek](https://www.hellonicha.com/)* from IPAG Yale University.

[![License: Apache](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](./LICENSE) [![arXiv:2509.10344](https://img.shields.io/badge/arXiv-2409.18119-B31B1B.svg)](https://arxiv.org/abs/2509.10344)


![teaser](assets/teaser.png)

This is the official implementation of paper **GLAM**: *"Geometry-Guided Local Alignment for Multi-View Visual Language Pre-Training in Mammography"* (accepted by **MICCAI 2025**)


## Table of Contents
- [News](#news)
- [Abstract:](#abstract)
- [Reproducibility](#reproducibility)
- [Environment:](#environment)
- [Environment:](#environment-1)
- [Dataset:](#dataset)
    - [A. EMBED](#a-embed)
    - [Data Split](#data-split)
    - [B. RSNA-Mammo dataset](#b-rsna-mammo-dataset)
    - [C. VinDr-Mammo dataset](#c-vindr-mammo-dataset)
    - [D. Define your data folder](#d-define-your-data-folder)
- [Pre-trained Checkpoint](#pre-trained-checkpoint)
- [Pre-training:](#pre-training)
- [Zero-shot Evaluation](#zero-shot-evaluation)
- [Linear-Probing and Full Fine-tuning](#linear-probing-and-full-fine-tuning)
- [Reference](#reference)


## News

- [Dec 2025] The code for GLAM is officially released!


## Abstract:

> Mammography screening is an essential tool for early detection of breast cancer. The speed and accuracy of mammography interpretation have the potential to be improved with deep learning methods. However, the development of a foundation visual language model (VLM) is hindered by limited data and domain differences between natural and medical images. Existing mammography VLMs, adapted from natural images, often ignore domain-specific characteristics, such as multi-view relationships in mammography. Unlike radiologists who analyze both views together to process ipsilateral correspondence, current methods treat them as independent images or do not properly model the multi-view correspondence learning, losing critical geometric context and resulting in suboptimal prediction. We propose GLAM: Global and Local Alignment for Multi-view mammography for VLM pretraining using geometry guidance. By leveraging the prior knowledge about the multi-view imaging process of mammograms, our model learns local cross-view alignments and fine-grained local features through joint global and local, visual-visual, and visual-language contrastive learning. Pretrained on EMBED, one of the largest open mammography datasets, our model outperforms baselines across multiple datasets under different settings.


## Reproducibility

### Environment:

### Environment:

We first prepare the environment with required packages, we use PyTorch 2.1.2 with CUDA 11.8 and pytorch-lightning 2.1+ for development as evaluation. We also use `xformers` for more efficient training and testing. You may install the environment with the following steps:

```bash
conda env create -f environment.yml
# (Required) Manually install cosine annealing with warmup
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```

### Dataset:

#### A. EMBED

We pretrained our model with [Emory Breast Imaging Dataset (EMBED)](https://github.com/Emory-HITI/EMBED_Open_Data) from Emory University, which is one of the current largest 2D Mammography datasets. The dataset requires application to access, which can be done by filling out this [form](https://forms.gle/HwGMM6vdv3w32TKF9). We use both screening and diagnostic images for pre-training.

Download the EMBED dataset at [here](https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#resources)

We pre-process and re-size the original DICOM images using `scripts/resize_embed.py`, which resizes the long side of the original DICOM image to 1024. This will speed up training by a lot and save your local disk space. For more detailed settings, please refer to our paper.

##### Data Split

Unfortunately, we cannot share the data split for the EMBED dataset publicly as access to this dataset needs approval. However, you can create your own data split following the same settings mentioned in the paper: 70%/10%/20% for training/validation/testing. You can also generate a similar split using `preprocess_embed.py`.

For downstream fine-tuning and final evaluation, please run `preprocess_embed_test.py` to get the corresponding data split.

#### B. RSNA-Mammo dataset

We use the RSNA-Mammo dataset from the RSNA breast cancer detection challenge for out-of-distribution evaluation, which is a binary classification dataset for breast cancer.

Download the RSNA-Mammo at [here](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview).

You need run `scripts/resize_rsna.py` to resize the image and speed-up loading.


#### C. VinDr-Mammo dataset

We use VinDr-Mammo dataset from the VinDr.ai as another out-of-domain evaluation. which is for BI-RADS and density classification.

Download the VinDr-Mammo at [here](https://vindr.ai/datasets/mammo).

Similarly, you can pre-process the data with `scripts/resize_vindr.py`.


#### D. Define your data folder

Before you proceed, you need to define the directory for all your datasets. You can change this at [here](https://github.com/XYPB/MaMA/blob/aefc7750f23b0d163feade8732e957c4a7552480/dataset/constants_val.py#L5), replace `<path-to-your-data-folder>` with your own path.

Besides, you also need to use your own Huggingface API token to access and download pretrained encoders. You need to search `<replace-with-your-hf-api-token>` within the repo, and replace it with your own API tokens.


### Pre-trained Checkpoint

**Note: Unfortunately, we are not allowed to share the pre-trained model weight due to the EMBED dataset [policy](https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/EMBED_license.md). You may apply for access to the data and then train the model, following the instructions below.**

**NOTE**: You may encounter a potential error when using gradient checkpoint with LLMs implemented by Huggingface. To solve this, you need to add `use_reentrant=True` to the `gradient_checkpoint` function in the source code. You may also refer to [this issue](https://github.com/huggingface/transformers/issues/28536).

### Pre-training:

We use `wandb` to log our experiment results, so you may want to configure your `wandb` first before reproduce the results.

You may also reproduce the full pre-training process as follows:
```bash
./scripts/pretrain.sh
```

### Zero-shot Evaluation

To reproduce the zero-shot evaluation, run:
```bash
scripts/zs_eval.sh
```

You need to replace the `PRETRAINED_MODEL` with your pre-trained model path from the last step.

### Linear-Probing and Full Fine-tuning

To train the models under linear probing settings, run
```bash
./scripts/lp_training.sh
```

To train the model under full fine-tune settings, run
```bash
./scripts/fft_training.sh
```

Similarly, you need to replace the `PRETRAINED_MODEL` with your pre-trained model path from the last step

To evaluate the fine-tuned models, you can replace `--pretrained_encoder` parameter with `--pretrained_model` and attach the path to the fine-tuned model and add `--eval` argument.


## Reference


```
@InProceedings{DuYue_GeometryGuided_MICCAI2025,
    author = { Du, Yuexi AND Chen, Lihui AND Dvornek, Nicha C.},
    title = { { Geometry-Guided Local Alignment for Multi-View Visual Language Pre-Training in Mammography } },
    booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
    year = {2025},
    publisher = {Springer Nature Switzerland},
    volume = {LNCS 15965},
    month = {September},
    page = {299 -- 310}
}
```