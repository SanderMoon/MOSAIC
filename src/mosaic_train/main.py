"""
This code largely follows the same code and structure as the open_clip project:
https://github.com/mlfoundations/open_clip

Further modifications have been made to the training, evaluation and data loading code for this project.
"""

import glob
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler

from mosaic_train.evaluator import Evaluator

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from mosaic.loss import create_loss
from mosaic.model_factory import create_model
from mosaic.utils import trace_model
from mosaic_train.data import get_data
from mosaic_train.distributed import (
    broadcast_object,
    init_distributed_device,
    is_master,
)
from mosaic_train.file_utils import (
    pt_load,
    remote_sync,
    start_sync_process,
)
from mosaic_train.logger import setup_logging
from mosaic_train.params import parse_args
from mosaic_train.scheduler import const_lr, const_lr_cooldown, cosine_lr
from mosaic_train.train import train_one_epoch

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(" ")[-1])
            for x in result.stdout.decode().split("\n")[:-1]
        ]
    else:
        checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    # logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_base_path = log_base_path
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
        )
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print(
                    "Error. Cannot use save-most-recent with remote_sync and resume latest."
                )
                return -1
            if args.remote_sync_protocol != "s3":
                print("Error. Sync protocol not supported when using resume latest.")
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(
                    checkpoint_path, remote=args.remote_sync is not None
                )
            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("remote sync successful.")
        else:
            logging.info("Error: remote sync failed. Exiting.")
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        remote_sync_process.start()

    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )

    if args.horovod:
        logging.info(
            f"Running in horovod mode with multiple processes / nodes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    random_seed(args.seed, 0)

    model, tokenizer, amp, input_dtype = create_model(
        model_name=args.model,
        pretrained=args.pretrained,
        precision=args.precision,
        device=device,
        init_tokenizer=True,
    )
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    params_file = None
    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **ddp_args
        )

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_split:
        assert not args.trace, "Cannot train with traced model"

        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )

        def include(n, p):
            return not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters()
            )
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler("cuda") if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        start_epoch = resume_from_checkpoint(args, model, optimizer, scaler)

    # initialize datasets
    data = get_data(
        tokenizer=tokenizer,
        args=args,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    assert len(data), "At least one train or eval dataset must be specified."

    # create scheduler if train
    scheduler = init_lr_scheduler(args, optimizer, data)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    writer = maybe_init_tensorboard(args)

    maybe_init_wandb(args, model, params_file, data)

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info("Compiling model...")
        model = torch.compile(original_model)

    loss = create_loss(
        args, pad_id=tokenizer.pad_token_id, gather_with_grad=args.gather_with_grad
    )
    eval_loss = create_loss(args, pad_id=tokenizer.pad_token_id, gather_with_grad=False)
    evaluator = Evaluator(tokenizer)

    if "train" not in data or args.test:
        # Evaluate.
        evaluator.evaluate(
            model,
            data,
            start_epoch,
            args,
            eval_loss,
            input_dtype,
            amp,
            tb_writer=writer,
            tokenizer=tokenizer,
        )
        return

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        train_one_epoch(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            args,
            tokenizer=tokenizer,
            tb_writer=writer,
        )
        completed_epoch = epoch + 1

        if epoch >= args.eval_grace_period:
            if "val" in data and (
                (epoch + 1) % args.val_frequency == 0 or (epoch + 1) == args.epochs
            ):
                evaluator.evaluate(
                    model,
                    data,
                    epoch,
                    args,
                    eval_loss,
                    input_dtype,
                    amp,
                    tb_writer=writer,
                    tokenizer=tokenizer,
                )

        # Saving checkpoints.
        if args.save_logs:
            save_checkpoint(args, optimizer, scaler, original_model, completed_epoch)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        final_sync(args, remote_sync_process)


def init_lr_scheduler(args, optimizer, data):
    scheduler = None
    if "train" in data and optimizer is not None:
        total_steps = (
            data["train"].dataloader.num_batches // args.accum_freq
        ) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert (
                args.epochs_cooldown is not None
            ), "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (
                data["train"].dataloader.num_batches // args.accum_freq
            ) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer,
                args.lr,
                args.warmup,
                total_steps,
                cooldown_steps,
                args.lr_cooldown_power,
                args.lr_cooldown_end,
            )
        else:
            logging.error(
                f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
            )
            exit(1)
    return scheduler


def maybe_init_tensorboard(args):
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    return writer


def maybe_init_wandb(args, model, params_file, data):
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume="auto" if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb.")


def save_checkpoint(args, optimizer, scaler, original_model, completed_epoch):
    checkpoint_dict = {
        "epoch": completed_epoch,
        "name": args.name,
        "state_dict": original_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    if completed_epoch == args.epochs or (
        args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
    ):
        torch.save(
            checkpoint_dict,
            os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
        )
    if args.delete_previous_checkpoint:
        previous_checkpoint = os.path.join(
            args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
        )
        if os.path.exists(previous_checkpoint):
            os.remove(previous_checkpoint)

    if args.save_most_recent:
        # try not to corrupt the latest checkpoint if save fails
        tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
        latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
        torch.save(checkpoint_dict, tmp_save_path)
        os.replace(tmp_save_path, latest_save_path)


def final_sync(args, remote_sync_process):
    logging.info("Final remote sync.")
    remote_sync_process.terminate()
    result = remote_sync(
        os.path.join(args.logs, args.name),
        os.path.join(args.remote_sync, args.name),
        args.remote_sync_protocol,
    )
    if result:
        logging.info("Final remote sync successful.")
    else:
        logging.info("Final remote sync failed.")


def resume_from_checkpoint(args, model, optimizer, scaler):
    checkpoint = pt_load(args.resume, map_location="cpu")
    start_epoch = 0
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
        logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    return start_epoch


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
