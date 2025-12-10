import datetime
import os
from argparse import ArgumentParser

# from torch.utils.tensorboard import SummaryWriter

import torch
from dateutil import tz

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from dataset.data_module import DataModule
from dataset.pretrain_embed_dataset import (
    multimodal_collate_fn,
    EmbedPretrainingDataset,
)
from dataset.mammo_eval_dataset import VinDr, RSNAMammo
from dataset.transforms import (
    Moco2Transform,
    SimCLRTransform,
    SimCLRFixedTransform,
    DetectionDataTransforms,
    RSNAMammoTransform,
    ChestXRayTransform,
)

from model import GLAM

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

os.environ["WANDB_START_METHOD"] = "thread"


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def train(args, model, datamodule):
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension += f"_{args.experiment_name}"
    ckpt_dir = os.path.join(BASE_DIR, f"logs/ckpts/GLAM/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = "val_loss"
    mode = "min"
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.save_last_k > 1:
        callbacks.append(
            ModelCheckpoint(
                monitor="epoch",
                dirpath=ckpt_dir,
                save_last=True,
                mode="max",
                save_top_k=args.save_last_k,
            )
        )
    else:
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                dirpath=ckpt_dir,
                save_last=True,
                mode=mode,
                save_top_k=2,
            )
        )
    logger_dir = os.path.join(BASE_DIR, f"./logs")
    os.makedirs(logger_dir, exist_ok=True)
    if args.img_cls_ft:
        if args.embed:
            project = "GLAM_img_embed_ft"
        elif args.vindr:
            project = "GLAM_img_vindr_ft"
        else:
            project = "GLAM_img_cls_ft"
        if "fft" in args.experiment_name:
            project = project.replace("ft", "fft")
    elif args.embed:
        project = "GLAM_Embed"
    else:
        project = "GLAM_fix_step"
    wandb_logger = WandbLogger(project=project, save_dir=logger_dir, name=extension)
    num_available_gpus = torch.cuda.device_count()
    if args.devices > num_available_gpus:
        print(
            f"### Using less GPUs than requested: {args.devices} > {num_available_gpus}"
        )
        args.devices = num_available_gpus
    print(f"### Using {args.strategy} Strategy with {args.devices} GPUs")
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        fast_dev_run=args.dev,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=int(1 / args.data_pct),
    )

    model.training_steps = model.num_training_steps(trainer, datamodule)

    dtype = None
    if args.strategy == "fsdp":
        for name, param in model.named_parameters():
            if dtype is None:
                dtype = param.dtype
            elif dtype != param.dtype:
                print(f"Parameter {name} has dtype {param.dtype}, expected {dtype}")
        print(dtype)

    print(f"\n### Resume from {args.resume_ckpt}...\n")

    if args.find_max_bsz:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=datamodule, mode="binsearch")
        print(f"### Best batch size: {datamodule.batch_size}")

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_ckpt,
    )

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)
    return model


def eval(args, model, datamodule):
    model.eval()

    # Single GPU inference
    trainer = Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        devices=1,
        fast_dev_run=args.dev,
        max_epochs=1,
        deterministic=args.deterministic,
        inference_mode=True,
    )

    trainer.test(model, datamodule=datamodule)


def cli_main():

    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--llm_type",
        type=str,
        default="gpt",
        help="bert, gpt, llama, llama2, or llama3",
    )
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--vindr", action="store_true")
    parser.add_argument("--rsna_mammo", action="store_true")
    parser = GLAM.add_model_specific_args(parser)

    args = parser.parse_args()
    args.deterministic = False

    if args.eval:
        args.batch_size = 32
        args.data_pct = 1.0
        args.max_epoch = 1
        args.accumulate_grad_batches = 1
        args.dev = False
        args.strategy = None
        args.devices = 1
        args.grad_ckpt = False
        args.train_sub_set = False
    num_cores = len(os.sched_getaffinity(0))
    if args.find_max_bsz:
        args.strategy = None
    if args.num_workers > num_cores:
        args.num_workers = num_cores
        print("switching to maximum num_workers = ", num_cores)
    if args.img_cls_ft:
        args.num_workers = 8 if args.num_workers > 8 else args.num_workers
        print("Fine-tuning... switching to num_workers = ", args.num_workers)

    if args.use_flash_attention:
        os.environ["XFORMERS_DISABLED"] = "1"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    # speed-up GEMM for Ampere GPUs
    torch.set_float32_matmul_precision("high")

    # seed
    seed_everything(args.seed)

    if args.embed:
        dataset = EmbedPretrainingDataset
    elif args.vindr:
        dataset = VinDr
    elif args.rsna_mammo:
        dataset = RSNAMammo
    else:
        dataset = EmbedPretrainingDataset

    if args.slip:
        if args.patch_contrast or args.fixed_view:
            transform_obj = SimCLRFixedTransform
        else:
            transform_obj = SimCLRTransform
    elif args.rsna_trans:
        transform_obj = RSNAMammoTransform
    else:
        if args.patch_contrast or args.fixed_view:
            transform_obj = SimCLRFixedTransform
        else:
            transform_obj = Moco2Transform
    # use default collect function for DataLoader
    collate_fn = multimodal_collate_fn

    datamodule = DataModule(
        dataset,
        collate_fn,
        transform_obj,
        args.data_pct,
        args.batch_size,
        args.num_workers,
        llm_type=args.llm_type,
        train_split=args.train_split,
        valid_split=args.valid_split,
        train_sub_set=args.train_sub_set,
        structural_cap=args.structural_cap,
        simple_cap=args.simple_cap,
        natural_cap=args.natural_cap,
        instance_test_cap=args.instance_test_cap,
        inter_side=args.inter_side,
        inter_view=args.inter_view,
        balanced_test=args.balanced_test,
        slip=(args.slip or args.moco),
        balance_training=args.balance_training,
        pred_density=args.pred_density,
        img_size=args.img_size,
        crop_size=args.crop_size,
        raw_caption=args.raw_caption,
        load_jpg=args.load_jpg,
        mask_ratio=args.mask_ratio,
        mask_meta=args.mask_meta,
        aug_orig_img=args.aug_orig_img,
        balance_ratio=args.balance_ratio,
        less_train_neg=args.less_train_neg,
        test_data_pct=args.test_data_pct,
        bootstrap_test=args.bootstrap_test,
        aug_text=args.aug_text,
        heavy_aug=args.heavy_aug,
        pred_only=args.pred_only,
        prob_diff_dcm=args.prob_diff_dcm,
        extra_cap=args.extra_cap,
        extract_train=args.extract_train,
        screen_only=args.screen_only,
        aligned_mlo=args.aligned_mlo,
        align_orientation=args.align_orientation,
        remove_text=args.remove_text,
        fixed_view=args.fixed_view,
        pred_mass=args.pred_mass,
        pred_calc=args.pred_calc,
    )

    # Add load from checkpoint

    if args.pretrained_model is None:
        model = GLAM(**args.__dict__)
    else:
        print(
            f"\n\n##### Loading pretrained model from {args.pretrained_model}\n\n"
        )
        model = GLAM.load_from_checkpoint(
            args.pretrained_model, map_location="cpu", strict=False, **args.__dict__
        )
        if "Downstream_evalualtion_b5_fold0" in args.pretrained_model:
            print("### load Mammo-CLIP pretrained classifier...")
            print(model.img_encoder_q.global_embed.bias)
        if args.eval and args.ema and not args.ema_no_load:
            model.set_ema_encoder()

    if args.eval:
        eval(args, model, datamodule)
    else:
        model = train(args, model, datamodule)
        # eval after training
        # args.batch_size = 32
        # args.data_pct = 1.0
        # args.max_epoch = 1
        # args.accumulate_grad_batches = 1
        # args.dev = False
        # args.strategy = None
        # args.devices = 1
        # args.grad_ckpt = False
        # eval(args, model, datamodule)


if __name__ == "__main__":
    cli_main()
