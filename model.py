import os
import random
import copy
from argparse import ArgumentParser

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassConfusionMatrix
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.distributed as dist

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from lightning_fabric.strategies import FSDPStrategy
from backbones.encoder_bert import BertEncoder
from backbones.encoder_dino import DinoEncoder, AttentionalPooler
from transformers import Adafactor
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
)
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
import torch._dynamo

# from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from backbones.cls_loss import BinaryCrossEntropyPosSmoothOnly
from backbones.asl import ASLSingleLabel
from backbones.utils import ModelEMA
from dataset.utils import get_specificity_with_sensitivity, pfbeta
from dataset.constants_val import (
    RSNA_POS_CLASS_WEIGHT,
    EMBED_SCREEN_BIRADS_WEIGHT,
    EMBED_SCREEN_DENSITY_WEIGHT,
    EMBED_ALL_BIRADS_WEIGHT,
    EMBED_ALL_DENSITY_WEIGHT,
    VIN_DR_DENSITY_WEIGHT,
    VIN_DR_BIRADS_WEIGHT,
    VIN_DR_MASS_WEIGHT,
    VIN_DR_CALC_WEIGHT,
)

torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHEXPERT_BASE_CAPTION = "this is a chest x ray of a patient with "


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

os.environ["WANDB_START_METHOD"] = "thread"


class GLAM(LightningModule):

    def __init__(
        self,
        img_encoder: str = "dinov2_vitb14_reg",
        freeze_llm: bool = False,
        emb_dim: int = 128,
        softmax_temperature: float = 0.07,
        learning_rate: float = 2e-5,
        momentum: float = 0.9,
        weight_decay: float = 0.05,
        batch_size: int = 144,
        num_workers: int = 8,
        num_heads: int = 1,
        lamb: float = 0.75,
        epsilon: float = 0.05,
        agg_tokens: bool = False,
        grad_ckpt: bool = False,
        img_cls_ft: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.embed:
            self.hparams.num_classes = 4 if self.hparams.pred_density else 7
            if self.hparams.screen_only and not self.hparams.pred_density:
                self.hparams.num_classes = 3
        elif self.hparams.vindr:
            if self.hparams.pred_mass or self.hparams.pred_calc:
                self.hparams.num_classes = 1 if self.hparams.weighted_binary else 2
            elif self.hparams.pred_density:
                self.hparams.num_classes = 4
            else:
                self.hparams.num_classes = 5
        elif self.hparams.rsna_mammo:
            if self.hparams.weighted_binary:
                self.hparams.num_classes = 1
            else:
                self.hparams.num_classes = 2

        if self.hparams.num_classes == 1:
            self.confmat = MulticlassConfusionMatrix(self.hparams.num_classes + 1)
        else:
            self.confmat = MulticlassConfusionMatrix(self.hparams.num_classes)
        self.all_scores_train = None
        self.all_labels_train = None
        self.all_scores_val = None
        self.all_labels_val = None
        self.all_scores = None
        self.all_labels = None
        self.all_vis_feats = None

        # init encoders
        downsample_factor = 14
        self.img_encoder_q = DinoEncoder(
            model_name=img_encoder,
            output_dim=self.hparams.emb_dim,
            linear_proj=self.hparams.linear_proj,
            freeze_vit=self.hparams.freeze_vit,
            vit_grad_ckpt=self.hparams.vit_grad_ckpt,
            img_size=self.hparams.crop_size,
        )

        # Randomize the visual transformer
        if self.hparams.random_vit:
            self.img_encoder_q.model.init_weights()

        # Create a text encoder
        init_text_encoder = not self.hparams.img_cls_ft
        if init_text_encoder:
            if self.hparams.llm_type == "bert":
                self.text_encoder_q = BertEncoder(
                    output_dim=self.hparams.emb_dim,
                    freeze_llm=self.hparams.freeze_llm,
                    linear_proj=self.hparams.linear_proj,
                    agg_tokens=self.hparams.agg_tokens,
                    grad_ckpt=self.hparams.grad_ckpt,
                )
            else:
                raise NotImplementedError

        # Load pre-trained vit parameter
        if self.hparams.pretrained_encoder != None:
            print(
                "\n### Loading pretrained model from {}\n".format(
                    self.hparams.pretrained_encoder
                )
            )
            state_dict = torch.load(self.hparams.pretrained_encoder, map_location="cpu")
            if "fp32" not in self.hparams.pretrained_encoder:
                state_dict = state_dict["state_dict"]
            img_encoder_state_dict = {
                k.replace("img_encoder_q.", ""): v
                for k, v in state_dict.items()
                if k.startswith("img_encoder_q")
            }
            missing, unexpected = self.img_encoder_q.load_state_dict(
                img_encoder_state_dict, strict=False
            )
            print("### Missing keys: ", missing)
            print("### Unexpected keys: ", unexpected)
            if init_text_encoder:
                text_encoder_state_dict = {
                    k.replace("text_encoder_q.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("text_encoder_q")
                }
                self.text_encoder_q.load_state_dict(text_encoder_state_dict)

        # create a global classifier
        if self.hparams.img_cls_ft:
            self.img_encoder_q.global_embed = nn.Linear(
                self.img_encoder_q.feature_dim, self.hparams.num_classes
            )
            self.img_encoder_q.global_embed.weight.requires_grad = True
            self.img_encoder_q.global_embed.bias.requires_grad = True

        # Create patch contrast components
        if self.hparams.patch_contrast:
            # Alternate mega_patch_stride to control the overlap between each mega patch
            if self.hparams.attn_pooler:
                # each view use a separate attentional pooler
                hw = self.hparams.crop_size // downsample_factor
                num_patches = (hw // self.hparams.mega_patch_size) ** 2
                self.merge_patch1 = AttentionalPooler(
                    self.img_encoder_q.feature_dim,  # Use the same size query
                    self.img_encoder_q.feature_dim,
                    n_queries=num_patches,
                )
                self.merge_patch2 = AttentionalPooler(
                    self.img_encoder_q.feature_dim,  # Use the same size query
                    self.img_encoder_q.feature_dim,
                    n_queries=num_patches,
                )
                if self.hparams.late_loss > 0:
                    for param in self.merge_patch1.parameters():
                        param.requires_grad = False
                    for param in self.merge_patch2.parameters():
                        param.requires_grad = False
            else:
                self.merge_patch = nn.AvgPool2d(
                    self.hparams.mega_patch_size, self.hparams.mega_patch_stride
                )

            self.patch_ln = nn.LayerNorm(self.img_encoder_q.feature_dim)
            self.view_attn1 = nn.MultiheadAttention(
                self.img_encoder_q.feature_dim, self.hparams.num_heads, batch_first=True
            )
            self.view_attn2 = nn.MultiheadAttention(
                self.img_encoder_q.feature_dim, self.hparams.num_heads, batch_first=True
            )
            self.patch_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / self.hparams.softmax_temperature)
            )
            if self.hparams.late_loss > 0:
                for param in self.view_attn1.parameters():
                    param.requires_grad = False
                for param in self.view_attn2.parameters():
                    param.requires_grad = False

        # Initialize the learnable logit scale
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.hparams.softmax_temperature)
        )

        self.zero_shot_text_feats = None

        self.img_path = []

        # Create extra slip training components
        if self.hparams.slip:
            self.simclr_proj = SimCLRProjectionHead(
                self.img_encoder_q.feature_dim,
                self.img_encoder_q.feature_dim,
                self.hparams.emb_dim,
            )
            self.simclr_loss = NTXentLoss(gather_distributed=(self.hparams.devices > 1))

        if self.hparams.asl:
            if self.hparams.vindr:
                self.asl_loss_func = ASLSingleLabel(
                    gamma_neg=self.hparams.gamma_neg,
                    gamma_pos=self.hparams.gamma_pos,
                )
            elif self.hparams.rsna_mammo:
                if self.hparams.asl_weighted:
                    # TODO
                    raise NotImplementedError
                else:
                    self.asl_loss_func = ASLSingleLabel(
                        gamma_neg=self.hparams.gamma_neg,
                        gamma_pos=self.hparams.gamma_pos,
                    )
            else:
                self.asl_loss_func = ASLSingleLabel(
                    gamma_neg=self.hparams.gamma_neg,
                    gamma_pos=self.hparams.gamma_pos,
                )

        # Freeze unused parameters:
        if self.hparams.pool_feat:
            self.img_encoder_q.model.norm.weight.requires_grad = False
            self.img_encoder_q.model.norm.bias.requires_grad = False

        # Only use EMA for vision encoder
        if self.hparams.ema:
            print("### Using EMA for vision encoder")
            assert self.hparams.img_cls_ft
            self.ema_img_encoder = ModelEMA(
                self.img_encoder_q, decay=self.hparams.ema_decay
            )

    def set_ema_encoder(self):
        # load ema model parameter to model
        print("### Loading EMA model parameters for inference...")
        self.img_encoder_q.load_state_dict(self.ema_img_encoder.module.state_dict())

    def get_data_keys(self, split="train"):
        keys = ["imgs", "caption_ids", "attention_mask", "multi_hot_label"]
        return keys

    # @profile
    def forward(self, batch, batch_idx, split="train"):
        """Forward step of our method"""
        ex_loss_dict = {}
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        loss_c = 0.0
        bz = img_emb_q.size(0)
        # Forward of query text encoder
        try:
            report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_feat = (
                self.text_encoder_q(
                    batch[cap_key],
                    batch[attn_key],
                    token_type=batch.get("token_type_ids", None),
                )
            )
        except Exception as e:
            print(batch[cap_key].shape)
            print(batch["path"])
            raise e
        if self.hparams.pool_txt_feat:
            report_feat_q_full = word_feat_q_full.mean(dim=1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        ########### image-text contrastive loss ################

        labels = torch.arange(bz).to(report_emb_q.device).long()
        scores = img_emb_q.mm(report_emb_q.t())
        scores *= self.logit_scale.exp()
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_c = loss0 + loss1

        # following slip, we add SimCLR projection results
        ########### image-image contrastive loss ################
        ex_patch_feat_q = None
        if self.hparams.slip and self.global_step >= self.hparams.late_loss:
            ext_feat_s1, ex_patch_feat_q, ext_full1 = self.img_encoder_q(
                batch["ext_imgs"]
            )
            if self.hparams.pool_feat:
                ext_feat_s1 = ext_full1.mean(dim=1)
            else:
                ext_feat_s2 = img_feat_q
            ext_emb_s1 = self.simclr_proj(ext_feat_s1)
            ext_emb_s2 = self.simclr_proj(ext_feat_s2)
            simclr_loss = self.simclr_loss(ext_emb_s1, ext_emb_s2)
            ex_loss_dict["simclr_loss"] = simclr_loss.item()
            loss_c += self.hparams.slip_loss_lambda * simclr_loss

            ########### symmetric clip loss ################
            if self.hparams.symmetric_clip:
                labels = torch.arange(bz).to(report_emb_q.device).long()
                ext_emb_q = self.img_encoder_q.global_embed(ext_feat_s1)
                ext_emb_q = F.normalize(ext_emb_q, dim=-1)
                ext_scores = ext_emb_q.mm(report_emb_q.t())
                ext_scores *= self.logit_scale.exp()
                ext_scores1 = ext_scores.transpose(0, 1)
                ext_loss0 = F.cross_entropy(ext_scores, labels)
                ext_loss1 = F.cross_entropy(ext_scores1, labels)
                loss_c += 1.0 * (ext_loss0 + ext_loss1)

        ########### patch contrastive loss ################
        if self.hparams.patch_contrast and self.global_step >= self.hparams.late_loss:
            if ex_patch_feat_q is None:
                _, ex_patch_feat_q, _ = self.img_encoder_q(batch["ext_imgs"])
            # Find patch correspondence
            # since image is already rotated, masked with OtsuCut, and resized, patchs are already aligned
            bsz, HW, C = patch_feat_q.shape
            if self.hparams.attn_pooler:
                # input is N x HW x C
                attn_mask = None
                ext_attn_mask = None
                mega_patch_emb = self.merge_patch1(
                    patch_feat_q, attn_mask=attn_mask
                )  # N x hw x C
                ex_mega_patch_emb = self.merge_patch2(
                    ex_patch_feat_q, attn_mask=ext_attn_mask
                )  # N x hw x C
                s = int(mega_patch_emb.shape[1] ** 0.5)
                # transpose from N x hw x C to N x C x h x w
                mega_patch_emb = mega_patch_emb.permute(0, 2, 1).reshape(bsz, C, s, s)
                ex_mega_patch_emb = ex_mega_patch_emb.permute(0, 2, 1).reshape(
                    bsz, C, s, s
                )
            else:
                s = int(HW**0.5)
                # transpose from N x HW x C to N x C x H x W
                patch_emb = (
                    patch_feat_q.permute(0, 2, 1).contiguous().reshape(bsz, C, s, s)
                )
                ex_patch_emb = (
                    ex_patch_feat_q.permute(0, 2, 1).contiguous().reshape(bsz, C, s, s)
                )
                # Current Patch are of size 16x16/14x14
                # Merge patches to get larger mega-patch
                # N x C x h x w
                mega_patch_emb = self.merge_patch(patch_emb)
                ex_mega_patch_emb = self.merge_patch(ex_patch_emb)
            # update number of patches in each spatial dimension
            s = mega_patch_emb.shape[2]

            view1_attn_mask = None
            view2_attn_mask = None

            if self.hparams.all_patch_contrast:
                mega_patch_emb = mega_patch_emb.permute(0, 2, 3, 1).reshape(
                    bsz, s * s, C
                )
                ex_mega_patch_emb = ex_mega_patch_emb.permute(0, 2, 3, 1).reshape(
                    bsz, s * s, C
                )
                mega_patch_emb = self.patch_ln(mega_patch_emb)  # N x hw x C
                ex_mega_patch_emb = self.patch_ln(ex_mega_patch_emb)  # N x hw x C
            else:
                # Each column is an individual sample
                # move width to the batch dimension
                # move the channel dimension to the last
                mega_patch_emb = mega_patch_emb.permute(0, 3, 2, 1).reshape(
                    bsz * s, s, C
                )
                ex_mega_patch_emb = ex_mega_patch_emb.permute(0, 3, 2, 1).reshape(
                    bsz * s, s, C
                )
                mega_patch_emb = self.patch_ln(mega_patch_emb)  # Nw x h x C
                ex_mega_patch_emb = self.patch_ln(ex_mega_patch_emb)  # Nw x h x C
            # Attentional weighted sum of view2 patches given view1 patches
            # Attention weighted of shape: Nw x h x h
            # This means the weight between each patch in view1 to a column of patches in view2
            view1_to_view2, attn_weight_12 = self.view_attn1(
                mega_patch_emb,
                ex_mega_patch_emb,
                ex_mega_patch_emb,
                attn_mask=view1_attn_mask,
                average_attn_weights=self.hparams.average_attn_weights,
            )
            # Repeat for view2
            view2_to_view1, attn_weight_21 = self.view_attn2(
                ex_mega_patch_emb,
                mega_patch_emb,
                mega_patch_emb,
                attn_mask=view2_attn_mask,
                average_attn_weights=self.hparams.average_attn_weights,
            )
            # Normalize for similarity computation
            # Nw x h x C
            view1_to_view2 = F.normalize(view1_to_view2, dim=-1)
            mega_patch_emb = F.normalize(mega_patch_emb, dim=-1)
            # Use all the patches from different position as the negative samples
            if not self.hparams.all_patch_contrast:
                view1_to_view2 = view1_to_view2.reshape(bsz, s * s, -1)  # N x hw x C
                mega_patch_emb = mega_patch_emb.reshape(bsz, s * s, -1)  # N x hw x C
            view1_sim = torch.bmm(mega_patch_emb, view1_to_view2.transpose(1, 2))
            view1_sim *= self.patch_scale.exp()  # N x hw x hw
            view1_label = (
                torch.arange(view1_sim.shape[1])
                .to(mega_patch_emb.device)
                .long()
                .repeat(view1_sim.shape[0])
            )
            view1_sim1 = view1_sim.reshape(bsz * s * s, -1).contiguous()
            view1_sim2 = (
                view1_sim.permute(0, 2, 1).reshape(bsz * s * s, -1).contiguous()
            )
            loss_view11 = F.cross_entropy(view1_sim1, view1_label)
            loss_view12 = F.cross_entropy(view1_sim2, view1_label)
            loss_view1 = 0.5 * (loss_view11 + loss_view12)

            # Normalize for similarity computation
            view2_to_view1 = F.normalize(view2_to_view1, dim=-1)
            ex_mega_patch_emb = F.normalize(ex_mega_patch_emb, dim=-1)
            # Use all the patches from different position as the negative samples
            if not self.hparams.all_patch_contrast:
                view2_to_view1 = view2_to_view1.reshape(bsz, s * s, -1)
                ex_mega_patch_emb = ex_mega_patch_emb.reshape(bsz, s * s, -1)
            view2_sim = torch.bmm(ex_mega_patch_emb, view2_to_view1.transpose(1, 2))
            view2_sim *= self.patch_scale.exp()  # N x hw x hw
            view2_label = (
                torch.arange(view2_sim.shape[1])
                .to(ex_mega_patch_emb.device)
                .long()
                .repeat(view2_sim.shape[0])
            )
            view2_sim1 = view2_sim.reshape(bsz * s * s, -1).contiguous()
            view2_sim2 = (
                view2_sim.permute(0, 2, 1).reshape(bsz * s * s, -1).contiguous()
            )
            loss_view21 = F.cross_entropy(view2_sim1, view2_label)
            loss_view22 = F.cross_entropy(view2_sim2, view2_label)
            loss_view2 = 0.5 * (loss_view21 + loss_view22)
            loss_view = 0.5 * (loss_view1 + loss_view2)
            ex_loss_dict["patch_contrast_loss"] = loss_view.item()

            loss_c += loss_view

            if self.hparams.same_pos_contrast:
                # contrast patches at the same position across the batch
                view1_to_view2 = view1_to_view2.permute(1, 0, 2)  # hw x N x C
                view2_to_view1 = view2_to_view1.permute(1, 0, 2)  # hw x N x C
                mega_patch_emb = mega_patch_emb.permute(1, 0, 2)  # hw x N x C
                ex_mega_patch_emb = ex_mega_patch_emb.permute(1, 0, 2)  # hw x N x C

                pos1_sim = torch.bmm(
                    mega_patch_emb, view1_to_view2.transpose(1, 2)
                )  # hw x N x N
                pos1_sim *= self.patch_scale.exp()
                pos1_label = (
                    torch.arange(pos1_sim.shape[1])
                    .to(mega_patch_emb.device)
                    .long()
                    .repeat(pos1_sim.shape[0])
                )
                pos1_sim1 = pos1_sim.reshape(s * s * bsz, -1).contiguous()
                pos1_sim2 = (
                    pos1_sim.permute(0, 2, 1).reshape(s * s * bsz, -1).contiguous()
                )
                loss_pos11 = F.cross_entropy(pos1_sim1, pos1_label)
                loss_pos12 = F.cross_entropy(pos1_sim2, pos1_label)
                loss_pos1 = 0.5 * (loss_pos11 + loss_pos12)

                pos2_sim = torch.bmm(
                    ex_mega_patch_emb, view2_to_view1.transpose(1, 2)
                )  # hw x N x N
                pos2_sim *= self.patch_scale.exp()
                pos2_label = (
                    torch.arange(pos2_sim.shape[1])
                    .to(ex_mega_patch_emb.device)
                    .long()
                    .repeat(pos2_sim.shape[0])
                )
                pos2_sim1 = pos2_sim.reshape(s * s * bsz, -1).contiguous()
                pos2_sim2 = (
                    pos2_sim.permute(0, 2, 1).reshape(s * s * bsz, -1).contiguous()
                )
                loss_pos21 = F.cross_entropy(pos2_sim1, pos2_label)
                loss_pos22 = F.cross_entropy(pos2_sim2, pos2_label)
                loss_pos2 = 0.5 * (loss_pos21 + loss_pos22)
                loss_pos = 0.5 * (loss_pos1 + loss_pos2)
                ex_loss_dict["patch_pos_contrast_loss"] = loss_pos.item()
                loss_c += loss_pos

        # compute retrieval accuracy
        # Re create labels
        labels = torch.arange(bz).to(img_emb_q.device).long()
        i2t_acc1, i2t_acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.0
        acc5 = (i2t_acc5 + t2i_acc5) / 2.0

        return loss_c, acc1, acc5, ex_loss_dict

    def zero_shot_inference(self, batch, batch_idx, split="test"):
        """Inference with zero shot setting"""
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        with torch.no_grad():
            # Forward of query image encoder
            img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
            # Following FLIP, use the average of patch features w/o layer norm
            if self.hparams.pool_feat:
                img_feat_q = img_full.mean(dim=1)
            # Use classification token instead of averaged patch tokens
            img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
            img_emb_q = F.normalize(img_emb_q, dim=-1)

            # Forward of query text encoder
            # Forward for each individual image
            bsz = img_emb_q.size(0)  # N x C
            batch_scores = []
            if batch[cap_key].shape[0] == 1:
                raise ValueError
            if not self.hparams.instance_test_cap:
                fixed_caption_ids = batch[cap_key][0]  # CLS x S, get rid of batch dim
                fixed_attention_mask = batch[attn_key][0]

            for idx in range(bsz):
                if self.hparams.instance_test_cap:
                    fixed_caption_ids = batch[cap_key][idx]
                    fixed_attention_mask = batch[attn_key][idx]
                if self.zero_shot_text_feats is None or self.hparams.instance_test_cap:
                    token_type = batch.get("token_type_ids", None)
                    token_type = None if token_type is None else token_type[idx]
                    (
                        report_feat_q_full,
                        word_feat_q_full,
                        word_attn_q_full,
                        sents_full,
                    ) = self.text_encoder_q(
                        fixed_caption_ids,
                        fixed_attention_mask,
                        token_type=token_type,
                    )
                    report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
                    report_emb_q = F.normalize(report_emb_q, dim=-1)

                    self.zero_shot_text_feats = report_emb_q  # CLS x C

                scores = img_emb_q[idx : idx + 1].mm(
                    self.zero_shot_text_feats.t()
                )  # 1 x CLS
                scores *= self.logit_scale.exp()
                batch_scores.append(scores.squeeze(0))
            scores = torch.stack(batch_scores, dim=0)  # N x CLS

            if self.hparams.extract_feat:
                assert self.hparams.devices == 1
                # nomalized image features
                all_feats = img_emb_q.detach().to(torch.float32).cpu().numpy()
                if self.all_vis_feats is None:
                    self.all_vis_feats = all_feats
                else:
                    self.all_vis_feats = np.concatenate(
                        [self.all_vis_feats, all_feats], axis=0
                    )
                if not self.hparams.pred_only:
                    for i in range(len(batch["path"])):
                        self.img_path.append(batch["path"][i])

            # skip all evaluations
            if self.hparams.pred_only:
                assert split == "test"
                assert self.hparams.devices == 1
                all_scores = scores.detach().to(torch.float32).cpu()
                for i in range(len(batch["path"])):
                    self.img_path.append(batch["path"][i])
                all_scores = all_scores.detach().to(torch.float32)
                if self.hparams.multi_label:
                    all_scores = torch.sigmoid(all_scores).cpu().numpy()
                else:
                    all_scores = torch.softmax(all_scores, dim=-1).cpu().numpy()
                if self.all_scores is None:
                    self.all_scores = all_scores
                else:
                    self.all_scores = np.concatenate(
                        [self.all_scores, all_scores], axis=0
                    )
                return 0, 0, 0

            ########### image-text zero-shot cls loss ################
            labels = batch[label_key].to(scores.device)  # N x CLS

            # Image to text classification loss
            if self.hparams.multi_label:
                loss0 = F.binary_cross_entropy_with_logits(
                    scores, labels.to(torch.float32)
                )

                preds = F.sigmoid(scores)
                i2t_acc1 = accuracy_score(
                    labels.to(torch.float32).detach().cpu().numpy().flatten(),
                    (preds > 0.5).to(torch.float32).detach().cpu().numpy().flatten(),
                )
            else:
                loss0 = F.cross_entropy(scores, labels.argmax(dim=-1))
                # compute retrieval accuracy
                i2t_acc1 = self.precision_at_k(
                    scores, labels.argmax(dim=-1), top_k=(1,)
                )[0]

            labels = labels.float().detach().cpu().numpy()
            if self.hparams.multi_label:
                scores = torch.sigmoid(scores.float().detach()).cpu().numpy()
            else:
                scores = torch.softmax(scores.float().detach(), dim=1).cpu().numpy()
            # auc = roc_auc_score(labels, scores)
            auc = 0.0
            # report = classification_report(np.argmax(labels, axis=-1), np.argmax(scores, axis=-1),
            #                                output_dict=True, zero_division=0)

            if split == "test":
                if self.hparams.devices > 1:
                    score_list = [
                        torch.zeros_like(torch.tensor(scores))
                        for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(score_list, torch.tensor(scores))
                    all_scores = torch.cat(score_list, dim=0)
                    label_list = [
                        torch.zeros_like(torch.tensor(labels))
                        for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(label_list, torch.tensor(labels))
                    all_labels = torch.cat(label_list, dim=0)
                else:
                    all_scores = torch.tensor(scores)
                    all_labels = torch.tensor(labels)
                self.confmat.update(
                    torch.argmax(all_scores, dim=-1), all_labels.argmax(dim=-1)
                )
                all_scores = all_scores.detach().to(torch.float32)
                all_scores = all_scores.cpu().numpy()
                all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
                if self.all_scores is None:
                    self.all_scores = all_scores
                else:
                    self.all_scores = np.concatenate(
                        [self.all_scores, all_scores], axis=0
                    )
                if self.all_labels is None:
                    self.all_labels = all_labels
                else:
                    self.all_labels = np.concatenate(
                        [self.all_labels, all_labels], axis=0
                    )
                for i in range(len(batch["path"])):
                    self.img_path.append(batch["path"][i])

            if self.hparams.retrieval:
                self.dataset_scores.append(scores)
                self.dataset_labels.append(labels)

        return loss0, i2t_acc1, auc

    def visual_forward(self, batch, batch_idx, split="train"):
        """Forward step of our method"""
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)

        # skip all evaluations
        if self.hparams.pred_only:
            assert split == "test"
            assert self.hparams.devices == 1
            all_img_emb_qs = img_emb_q
            for i in range(len(batch["path"])):
                self.img_path.append(batch["path"][i])
            all_img_emb_qs = all_img_emb_qs.detach().to(torch.float32)
            if self.hparams.multi_label or self.hparams.weighted_binary:
                all_img_emb_qs = torch.sigmoid(all_img_emb_qs).cpu().numpy()
            else:
                all_img_emb_qs = torch.softmax(all_img_emb_qs, dim=-1).cpu().numpy()
            if self.all_scores is None:
                self.all_scores = all_img_emb_qs
            else:
                self.all_scores = np.concatenate(
                    [self.all_scores, all_img_emb_qs], axis=0
                )

            return 0, 0, 0
        ########### Classification loss ################
        labels = batch[label_key].to(img_emb_q.device)  # N x CLS

        # Image classification loss
        if self.hparams.multi_label:
            reduce = not self.hparams.focal_loss
            if self.hparams.asl:
                loss0 = self.asl_loss_func(img_emb_q, labels.to(torch.float32))
            else:
                loss0 = F.binary_cross_entropy_with_logits(
                    img_emb_q, labels.to(torch.float32), reduce=reduce
                )
            if self.hparams.focal_loss:
                pt = torch.exp(-loss0)
                F_loss = (
                    self.hparams.focal_alpha
                    * (1 - pt) ** self.hparams.focal_gamma
                    * loss0
                )
                loss0 = F_loss.mean()
        elif self.hparams.pos_smooth_only:
            loss_fn = BinaryCrossEntropyPosSmoothOnly(
                smoothing=self.hparams.label_smoothing,
            )
            loss0 = loss_fn(img_emb_q, labels.to(torch.float32))
        else:
            if self.hparams.asl:
                loss0 = self.asl_loss_func(img_emb_q, labels.argmax(dim=-1))
            elif self.hparams.weighted_binary:
                if self.hparams.rsna_mammo:
                    pos_weight = RSNA_POS_CLASS_WEIGHT
                elif self.hparams.vindr and self.hparams.pred_mass:
                    pos_weight = VIN_DR_MASS_WEIGHT
                elif self.hparams.vindr and self.hparams.pred_calc:
                    pos_weight = VIN_DR_CALC_WEIGHT
                else:
                    raise NotImplementedError
                loss0 = F.binary_cross_entropy_with_logits(
                    img_emb_q.squeeze(),
                    labels.argmax(dim=-1).to(torch.float32),
                    pos_weight=torch.tensor(pos_weight),
                    reduction="mean",
                )
            else:
                if self.hparams.weighted:
                    if self.hparams.vindr:
                        if self.hparams.pred_density:
                            weight = VIN_DR_DENSITY_WEIGHT
                        else:
                            weight = VIN_DR_BIRADS_WEIGHT
                    elif self.hparams.screen_only:
                        if self.hparams.pred_density:
                            weight = EMBED_SCREEN_DENSITY_WEIGHT
                        else:
                            weight = EMBED_SCREEN_BIRADS_WEIGHT
                    else:
                        if self.hparams.pred_density:
                            weight = EMBED_ALL_DENSITY_WEIGHT
                        else:
                            weight = EMBED_ALL_BIRADS_WEIGHT
                    loss0 = F.cross_entropy(
                        img_emb_q,
                        labels.argmax(dim=-1),
                        weight=torch.tensor(weight).to(img_emb_q.device),
                    )
                else:
                    loss0 = F.cross_entropy(
                        img_emb_q,
                        labels.argmax(dim=-1),
                        label_smoothing=self.hparams.label_smoothing,
                    )

        # compute retrieval accuracy
        if self.hparams.multi_label:
            preds = (F.sigmoid(img_emb_q) > 0.5).to(int)
            i2t_acc1 = accuracy_score(
                labels.to(torch.float32).detach().cpu().numpy().flatten(),
                preds.to(torch.float32).detach().cpu().numpy().flatten(),
            )
            # Hack acc@5 to log mAP
            i2t_acc5 = 0.0
        elif self.hparams.weighted_binary:
            i2t_acc1 = accuracy_score(
                labels.argmax(dim=-1).detach().cpu().numpy(),
                (F.sigmoid(img_emb_q).squeeze() > 0.5).to(int).detach().cpu().numpy(),
            )
            i2t_acc5 = 0.0
        else:
            i2t_acc1, i2t_acc5 = self.precision_at_k(
                img_emb_q, labels.argmax(dim=-1), top_k=(1, 2)
            )

        if split == "test":
            if self.hparams.devices > 1:
                img_emb_q_list = [
                    torch.zeros_like(img_emb_q) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(img_emb_q_list, img_emb_q)
                all_img_emb_qs = torch.cat(img_emb_q_list, dim=0)
                label_list = [
                    torch.zeros_like(labels) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(label_list, labels)
                all_labels = torch.cat(label_list, dim=0)
            else:
                all_img_emb_qs = img_emb_q
                all_labels = labels
            if not self.hparams.multi_label:
                if self.hparams.num_classes == 1:
                    pred = torch.sigmoid(all_img_emb_qs).squeeze() > 0.5
                else:
                    pred = all_img_emb_qs.argmax(dim=-1)
                self.confmat.update(pred, all_labels.argmax(dim=-1))
            all_img_emb_qs = all_img_emb_qs.detach().to(torch.float32)
            if self.hparams.multi_label or self.hparams.weighted_binary:
                all_img_emb_qs = torch.sigmoid(all_img_emb_qs).cpu().numpy()
            else:
                all_img_emb_qs = torch.softmax(all_img_emb_qs, dim=-1).cpu().numpy()
            all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
            if split == "train":
                if self.all_scores_train is None:
                    self.all_scores_train = all_img_emb_qs
                else:
                    self.all_scores_train = np.concatenate(
                        [self.all_scores_train, all_img_emb_qs], axis=0
                    )
                if self.all_labels_train is None:
                    self.all_labels_train = all_labels
                else:
                    self.all_labels_train = np.concatenate(
                        [self.all_labels_train, all_labels], axis=0
                    )
            elif split == "val":
                if self.all_scores_val is None:
                    self.all_scores_val = all_img_emb_qs
                else:
                    self.all_scores_val = np.concatenate(
                        [self.all_scores_val, all_img_emb_qs], axis=0
                    )
                if self.all_labels_val is None:
                    self.all_labels_val = all_labels
                else:
                    self.all_labels_val = np.concatenate(
                        [self.all_labels_val, all_labels], axis=0
                    )
            else:
                if self.all_scores is None:
                    self.all_scores = all_img_emb_qs
                else:
                    self.all_scores = np.concatenate(
                        [self.all_scores, all_img_emb_qs], axis=0
                    )
                if self.all_labels is None:
                    self.all_labels = all_labels
                else:
                    self.all_labels = np.concatenate(
                        [self.all_labels, all_labels], axis=0
                    )
            for i in range(len(batch["path"])):
                self.img_path.append(batch["path"][i])

        return loss0, i2t_acc1, i2t_acc5

    def training_step(self, batch, batch_idx):
        # unlock params after late loss starting step
        if (
            self.hparams.late_loss > 0
            and self.global_step == self.hparams.late_loss
            and not self.hparams.img_cls_ft
        ):
            if self.hparams.patch_contrast:
                print("\n#### Unlocking patch contrast layer params...\n")
                if self.hparams.attn_pooler:
                    for param in self.merge_patch1.parameters():
                        param.requires_grad = True
                    for param in self.merge_patch2.parameters():
                        param.requires_grad = True
                for param in self.view_attn1.parameters():
                    param.requires_grad = True
                for param in self.view_attn2.parameters():
                    param.requires_grad = True

        ex_loss_dict = None
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(batch, batch_idx, "train")
        else:
            loss_c, acc1, acc5, ex_loss_dict = self(batch, batch_idx, "train")
        loss = loss_c

        # Update the EMA model
        if self.hparams.ema and self.hparams.img_cls_ft:
            # print("Update EMA model...")
            self.ema_img_encoder.update(self.img_encoder_q)

        log = {
            "train_loss": loss,
            "train_loss_c": loss_c,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        if ex_loss_dict is not None:
            for k, v in ex_loss_dict.items():
                log[f"train_{k}"] = v
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        ex_loss_dict = None
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(batch, batch_idx, "val")
        else:
            loss_c, acc1, acc5, ex_loss_dict = self(batch, batch_idx, "val")
        loss = loss_c

        log = {
            "val_loss": loss,
            "val_loss_c": loss_c,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        if ex_loss_dict is not None:
            for k, v in ex_loss_dict.items():
                log[f"train_{k}"] = v
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss

    def test_step(self, batch, batch_idx):

        ex_loss_dict = None
        if self.hparams.img_cls_ft:
            loss_c, acc1, auc = self.visual_forward(batch, batch_idx, "test")
        else:
            loss_c, acc1, auc = self.zero_shot_inference(batch, batch_idx, "test")
        loss = loss_c

        log = {
            "test_loss": loss,
            "test_loss_c": loss_c,
            "test_acc1": acc1,
            "test_auc": auc,
        }
        if ex_loss_dict is not None:
            for k, v in ex_loss_dict.items():
                log[f"train_{k}"] = v
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        if (
            self.hparams.multi_label
            and self.all_scores_train is not None
            and self.all_labels_train is not None
        ):
            all_scores = self.all_scores_train
            all_preds = (self.all_scores_train > 0.5).astype(int)
            acc = 100 * accuracy_score(
                self.all_labels_train.flatten(), all_preds.flatten()
            )
            class_ap = []
            for i in range(all_scores.shape[-1]):
                ap = 100 * average_precision_score(
                    self.all_labels_train[:, i], all_scores[:, i]
                )
                class_ap.append(ap)
            mAP = np.mean(class_ap)

            class_auc = []
            try:
                for i in range(all_preds.shape[-1]):
                    auc = 100 * roc_auc_score(
                        self.all_labels_train[:, i], self.all_scores_train[:, i]
                    )
                    class_auc.append(auc)
                mAUC = np.mean(class_auc)
            except ValueError:
                mAUC = 0.0

            class_f1 = []
            for i in range(all_preds.shape[-1]):
                f1 = 100 * f1_score(self.all_labels_train[:, i], all_preds[:, i])
                class_f1.append(f1)
            mF1 = np.mean(class_f1)
            print("### Train Multi-label Acc: {:.4f}".format(acc))
            print("\n### Train Multi-label mAP: {:.4f}".format(mAP))
            print("### Class AP: ", [f"{ap:.4f}" for ap in class_ap])
            print("\n### Train Multi-label mAUC: {:.4f}".format(mAUC))
            print("### Class AUC: ", [f"{auc:.4f}" for auc in class_auc])
            print("\n### Train Multi-label mF1: {:.4f}".format(mF1))
            print("### Class F1: ", [f"{f1:.4f}" for f1 in class_f1])
            self.log_dict(
                {"train_mAP": mAP, "train_mAUC": mAUC, "train_mF1": mF1},
                on_epoch=True,
                sync_dist=True,
                logger=True,
            )
        self.all_scores_train = None
        self.all_labels_train = None

    def on_validation_epoch_end(self) -> None:
        if (
            self.hparams.multi_label
            and self.all_scores_val is not None
            and self.all_labels_val is not None
        ):
            all_scores = self.all_scores_val
            all_preds = (self.all_scores_val > 0.5).astype(int)
            acc = 100 * accuracy_score(
                self.all_labels_val.flatten(), all_preds.flatten()
            )
            class_ap = []
            for i in range(all_scores.shape[-1]):
                ap = 100 * average_precision_score(
                    self.all_labels_val[:, i], all_scores[:, i]
                )
                class_ap.append(ap)
            mAP = np.mean(class_ap)

            class_auc = []
            try:
                for i in range(all_preds.shape[-1]):
                    auc = 100 * roc_auc_score(
                        self.all_labels_val[:, i], self.all_scores_val[:, i]
                    )
                    class_auc.append(auc)
                mAUC = np.mean(class_auc)
            except ValueError:
                mAUC = 0.0

            class_f1 = []
            for i in range(all_preds.shape[-1]):
                f1 = 100 * f1_score(self.all_labels_val[:, i], all_preds[:, i])
                class_f1.append(f1)
            mF1 = np.mean(class_f1)
            print("### Validation Multi-label Acc: {:.4f}".format(acc))
            print("\n### Validation Multi-label mAP: {:.4f}".format(mAP))
            print("### Class AP: ", [f"{ap:.4f}" for ap in class_ap])
            print("\n### Validation Multi-label mAUC: {:.4f}".format(mAUC))
            print("### Class AUC: ", [f"{auc:.4f}" for auc in class_auc])
            print("\n### Validation Multi-label mF1: {:.4f}".format(mF1))
            print("### Class F1: ", [f"{f1:.4f}" for f1 in class_f1])
            self.log_dict(
                {"val_mAP": mAP, "val_mAUC": mAUC, "val_mF1": mF1},
                on_epoch=True,
                sync_dist=True,
                logger=True,
            )
        self.all_scores_val = None
        self.all_labels_val = None

    def on_test_epoch_end(self):

        if self.hparams.extract_feat:
            dest_dir = os.path.dirname(self.hparams.pretrained_model)
            vis_feat_dest = os.path.join(dest_dir, "vis_feats.npy")
            path_dest = vis_feat_dest.replace(".npy", "_path.pickle")
            print("### Extracted features saved to ", vis_feat_dest)
            np.save(vis_feat_dest, self.all_vis_feats)
            import pickle

            with open(path_dest, "wb") as fp:
                pickle.dump(self.img_path, fp)
        if self.hparams.pred_only:
            # No label, skip evaluation
            all_scores = self.all_scores.squeeze()
            assert len(all_scores) == len(self.img_path)
            pred_dict = {p: s for p, s in zip(self.img_path, all_scores)}
            if self.hparams.save_prediction:
                # Save predictions to disk
                dest_dir = os.path.dirname(self.hparams.pretrained_model)
                save_path = os.path.join(dest_dir, "path2predictions.pickle")
                import pickle

                print("### Prediction saved to ", save_path)
                with open(save_path, "wb") as fp:
                    pickle.dump(pred_dict, fp)
                return

        conf_matrix = self.confmat.compute().cpu().numpy()
        print("\n\n### Confusion Matrix:\n", conf_matrix)
        if self.hparams.rsna_mammo:
            tn = conf_matrix[0, 0]
            tp = conf_matrix[1, 1]
            fn = conf_matrix[1, 0]
            fp = conf_matrix[0, 1]
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            f1 = 2 * tp / (2 * tp + fp + fn)
            print("\n### Sensitivity: {:.4f}".format(100 * sensitivity))
            print("### Specificity: {:.4f}".format(100 * specificity))
            print("### PPV: {:.4f}".format(100 * ppv))
            print("### NPV: {:.4f}".format(100 * npv))
            print("### F1: {:.4f}".format(100 * f1))
        cls_cnt = np.sum(conf_matrix, axis=1)
        cls_hit = np.diag(conf_matrix)
        cls_acc = cls_hit / cls_cnt
        print("\n### Class Accuracy: ", [f"{100 * acc:.4f}" for acc in cls_acc])
        # Calculate the accuracy using the accumulated predictions and targets
        idx_label = np.argmax(self.all_labels, -1)
        if self.hparams.num_classes > 1:
            idx_pred = np.argmax(self.all_scores, -1)
        else:
            idx_pred = (self.all_scores > 0.5).astype(int)
        if self.hparams.multi_label:
            all_scores = self.all_scores
            all_preds = (self.all_scores > 0.5).astype(int)
            acc = 100 * accuracy_score(self.all_labels.flatten(), all_preds.flatten())
            class_ap = []
            for i in range(all_scores.shape[-1]):
                ap = 100 * average_precision_score(
                    self.all_labels[:, i], all_scores[:, i]
                )
                class_ap.append(ap)
            mAP = np.mean(class_ap)

            class_auc = []
            try:
                for i in range(all_preds.shape[-1]):
                    auc = 100 * roc_auc_score(
                        self.all_labels[:, i], self.all_scores[:, i]
                    )
                    class_auc.append(auc)
                mAUC = np.mean(class_auc)
            except ValueError:
                mAUC = 0.0

            class_f1 = []
            for i in range(all_preds.shape[-1]):
                f1 = 100 * f1_score(self.all_labels[:, i], all_preds[:, i])
                class_f1.append(f1)
            mF1 = np.mean(class_f1)

            print("### Class AP: ", [f"{ap:.4f}" for ap in class_ap])
            print("### Multi-label mAP: {:.4f}".format(mAP))
            print("\n### Multi-label mAUC: {:.4f}".format(mAUC))
            print("### Class AUC: ", [f"{auc:.4f}" for auc in class_auc])
            print("\n### Multi-label mF1: {:.4f}".format(mF1))
            print("### Class F1: ", [f"{f1:.4f}" for f1 in class_f1])
            ba = 0.0
        else:
            acc = 100 * accuracy_score(idx_label, idx_pred)
            # f1 = 100 * f1_score(idx_label, idx_pred)
            ba = 100 * balanced_accuracy_score(idx_label, idx_pred)
        try:
            if self.hparams.rsna_mammo:
                if self.hparams.num_classes == 2:
                    all_scores = self.all_scores[:, 1]
                else:
                    all_scores = self.all_scores.squeeze()
                auc = 100 * roc_auc_score(idx_label, all_scores)
                spec_80 = 100 * get_specificity_with_sensitivity(
                    idx_label, all_scores, 0.8
                )
                if self.hparams.pf1_threshold:
                    # This is then hard F1
                    best_pF1 = -1
                    best_threshold = 0
                    for threshold in np.linspace(0.01, 0.99, 99):
                        pred = all_scores > threshold
                        if len(np.unique(pred)) == 1:
                            continue
                        cur_pF1 = 100 * pfbeta(idx_label, pred)
                        if cur_pF1 > best_pF1:
                            best_pF1 = cur_pF1
                            best_threshold = threshold
                    print(f"Find best threshold {best_threshold}")
                    pF1 = best_pF1
                else:
                    pF1 = 100 * pfbeta(idx_label, all_scores)
                    pF1 = float(pF1)
            else:
                auc = 100 * roc_auc_score(idx_label, self.all_scores, multi_class="ovr")
                spec_80 = 0.0
                pF1 = 0.0
        except Exception as e:
            print("### Warning: AUC calculation failed with error:", e)
            raise e
            auc = 0
            spec_80 = 0.0
            pF1 = 0.0

        if self.hparams.save_prediction:
            # Save predictions to disk
            save_dir = self.hparams.pretrained_model.replace("last.ckpt", "predictions")
            if self.hparams.pred_density:
                save_dir = save_dir.replace("predictions", "predictions_density")
            if self.hparams.vindr:
                save_dir = save_dir.replace("predictions", "predictions_vindr")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "labels.npy"), self.all_labels)
            np.save(os.path.join(save_dir, "scores.npy"), self.all_scores)
            path_dest = os.path.join(save_dir, "path.pickle")
            with open(path_dest, "wb") as fp:
                import pickle

                pickle.dump(self.img_path, fp)

        print("### Accuracy: {:.4f}".format(acc))
        print("### Balanced Accuracy: {:.4f}".format(ba))
        print("### AUC: {:.4f}".format(auc))
        print("### pF1: {:.4f}".format(pF1))

        # Reset metrics for the next test run
        self.confmat.reset()
        self.all_scores = None
        self.all_labels = None

        # TODO: calculate retrieval performance
        if self.hparams.retrieval:
            retrieval_k = [1, 2, 5, 10]
            self.dataset_scores = np.concatenate(self.dataset_scores, axis=0)
            self.dataset_labels = np.concatenate(self.dataset_labels, axis=0)
            assert self.dataset_scores.shape[0] == self.dataset_labels.shape[0]

            num_classes = self.dataset_labels.shape[1]
            precisions = {k: [] for k in retrieval_k}
            for idx in range(num_classes):
                cls_scores = self.dataset_scores[:, idx]
                # sort classes wise score in descending order
                retrieved_idx = np.argsort(cls_scores)[::-1]
                for k in retrieval_k:
                    precisions[k].append(
                        self.dataset_labels[retrieved_idx[:k], idx].mean()
                    )

            print("Retrieval performance:")
            for k in retrieval_k:
                print("Top-{}: {:.4f}".format(k, 100 * np.mean(precisions[k])))

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        """Compute the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = (
                    correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                )
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def multi_label_precision(
        output: torch.Tensor, target: torch.Tensor, threshold=0.5
    ):
        """Compute the accuracy over the k top predictions for the specified values"""
        with torch.no_grad():
            # Applying threshold to prediction probabilities
            preds = output > threshold

            # Correct output are only those where prediction and label are equal
            correct_preds = (preds == target).float()

            # Compute accuracy across all target
            accuracy = 100 * correct_preds.sum() / (len(target) * target.size(1))

            return accuracy

    def configure_optimizers(self):
        parameters = self.parameters()
        if self.hparams.adafactor:
            optimizer = Adafactor(
                parameters,
                self.hparams.learning_rate,
                beta1=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                relative_step=False,
                scale_parameter=False,
            )
        elif self.hparams.sgd:
            optimizer = torch.optim.SGD(
                parameters,
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                self.hparams.learning_rate,
                betas=(self.hparams.momentum, 0.999),
                weight_decay=self.hparams.weight_decay,
            )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.hparams.max_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=self.hparams.min_lr,
            warmup_steps=self.hparams.warm_up,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model args
        parser.add_argument("--emb_dim", type=int, default=128, help="128, 256, 512")
        parser.add_argument("--linear_proj", action="store_true")
        parser.add_argument("--pool_feat", action="store_true")
        parser.add_argument("--pool_txt_feat", action="store_true")
        ### Visual Model args
        parser.add_argument("--img_encoder", type=str, default="dinov2_vitb14_reg")
        parser.add_argument("--freeze_vit", action="store_true")
        parser.add_argument("--slip", action="store_true")
        parser.add_argument("--symmetric_clip", action="store_true")
        parser.add_argument("--slip_loss_lambda", type=float, default=1.0)
        parser.add_argument("--random_vit", action="store_true")
        parser.add_argument("--vit_grad_ckpt", action="store_true")
        ### LLM args
        parser.add_argument("--freeze_llm", action="store_true")
        parser.add_argument("--num_freeze_blocks", type=int, default=0)
        parser.add_argument("--masked_lm_ratio", type=float, default=0)

        # Training args
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--find_max_bsz", action="store_true")
        parser.add_argument("--max_epochs", type=int, default=50)  # Unused
        parser.add_argument("--max_steps", type=int, default=40000)
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--img_cls_ft", action="store_true")
        parser.add_argument("--num_classes", type=int, default=1000)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--devices", type=int, default=4)
        parser.add_argument("--strategy", type=str, default="ddp")
        parser.add_argument("--accelerator", type=str, default="gpu")
        parser.add_argument("--precision", type=str, default="32")
        parser.add_argument("--dev", action="store_true")
        parser.add_argument("--grad_ckpt", action="store_true")
        parser.add_argument("--warm_up", type=int, default=16000)
        parser.add_argument("--balance_training", action="store_true")
        parser.add_argument("--balance_ratio", type=int, default=-1)
        parser.add_argument("--multi_label", action="store_true")
        parser.add_argument("--late_loss", type=int, default=-1)
        parser.add_argument("--weighted_binary", action="store_true")
        parser.add_argument("--weighted", action="store_true")
        parser.add_argument("--patch_contrast", action="store_true")
        parser.add_argument("--all_patch_contrast", action="store_true")
        parser.add_argument("--attn_pooler", action="store_true")
        parser.add_argument("--same_pos_contrast", action="store_true")
        parser.add_argument("--mega_patch_size", type=int, default=8)
        parser.add_argument("--mega_patch_stride", type=int, default=8)
        ### Hyperparameters
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--min_lr", type=float, default=1e-8)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        ### Optimizer
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--sgd", action="store_true")
        ### Pretrained args
        parser.add_argument("--pretrained_encoder", type=str, default=None)
        ### EMA model update
        parser.add_argument("--ema", action="store_true")
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--ema_no_load", action="store_true")

        # Data args
        parser.add_argument("--agg_tokens", action="store_true")
        parser.add_argument("--train_sub_set", action="store_true")
        parser.add_argument("--data_pct", type=float, default=1.0)
        parser.add_argument("--train_split", type=str, default="train")
        parser.add_argument("--valid_split", type=str, default="valid")
        parser.add_argument("--load_jpg", action="store_true")
        parser.add_argument("--img_size", type=int, default=224)
        parser.add_argument("--crop_size", type=int, default=224)
        parser.add_argument("--aug_orig_img", action="store_true")
        parser.add_argument("--max_words", type=int, default=144)
        parser.add_argument("--prob_diff_dcm", type=float, default=0.5)
        parser.add_argument("--screen_only", action="store_true")
        parser.add_argument("--aligned_mlo", action="store_true")
        parser.add_argument("--fixed_view", action="store_true")
        parser.add_argument("--align_orientation", action="store_true")
        parser.add_argument("--remove_text", action="store_true")
        ### EMBED test set args
        parser.add_argument("--balanced_test", action="store_true")
        parser.add_argument("--pred_density", action="store_true")
        parser.add_argument("--pred_mass", action="store_true")
        parser.add_argument("--pred_calc", action="store_true")
        # Caption args
        parser.add_argument("--structural_cap", action="store_true")
        parser.add_argument("--simple_cap", action="store_true")
        parser.add_argument("--natural_cap", action="store_true")
        parser.add_argument("--raw_caption", action="store_true")
        parser.add_argument("--aug_text", action="store_true")
        parser.add_argument("--heavy_aug", action="store_true")
        parser.add_argument("--mask_ratio", type=float, default=0.0)
        parser.add_argument("--mask_meta", type=float, default=-1.0)
        parser.add_argument("--extra_cap", type=str, default=None)
        # EMBED multi-images args
        parser.add_argument("--inter_view", action="store_true")
        parser.add_argument("--inter_side", action="store_true")
        parser.add_argument("--ext_img_prob", type=float, default=0.5)
        # Fine-tuning args
        parser.add_argument("--label_smoothing", type=float, default=0)
        parser.add_argument("--less_train_neg", type=float, default=0)
        parser.add_argument("--pos_smooth_only", action="store_true")
        parser.add_argument("--focal_loss", action="store_true")
        parser.add_argument("--focal_alpha", type=float, default=1)
        parser.add_argument("--focal_gamma", type=float, default=2)
        parser.add_argument("--asl", action="store_true")
        parser.add_argument("--asl_weighted", action="store_true")
        parser.add_argument("--gamma_neg", type=float, default=4)
        parser.add_argument("--gamma_pos", type=float, default=1)
        parser.add_argument("--asl_clip", type=float, default=0.05)
        parser.add_argument("--save_last_k", type=int, default=1)
        # Inference args
        parser.add_argument("--instance_test_cap", action="store_true")
        parser.add_argument("--retrieval", action="store_true")
        parser.add_argument(
            "--average_attn_weights", action="store_false", default=True
        )
        parser.add_argument("--pf1_threshold", action="store_true")
        parser.add_argument("--rsna_trans", action="store_true")
        parser.add_argument("--test_data_pct", type=float, default=1.0)
        parser.add_argument("--bootstrap_test", action="store_true")
        parser.add_argument("--save_prediction", action="store_true")
        parser.add_argument("--extract_feat", action="store_true")
        parser.add_argument("--extract_train", action="store_true")
        parser.add_argument("--pred_only", action="store_true")

        parser.add_argument("--use_flash_attention", action="store_true")

        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPStrategy, FSDPStrategy))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""

        return trainer.max_steps
