import logging
import math
import time

import torch

from mosaic.utils import get_input_dtype
from mosaic_train.precision import get_autocast

try:
    import wandb
except ImportError:
    wandb = None

import gc


class AccumulatedOutput:
    def __init__(self):
        self.images = []
        self.texts = []
        self.text_masks = []
        self.image_masks = []
        self.features = {}
        self.accum_batch_counter = 0


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metrics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler=None):
    """Performs the backward pass."""
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def clip_gradients(model, optimizer, grad_clip_norm):
    """Clips gradients to prevent exploding gradients."""
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_norm, norm_type=2.0
        )


def process_model_no_accum(
    model,
    optimizer,
    loss_fn,
    autocast,
    texts,
    images,
    text_attention_mask,
    image_features_attention_mask,
    scaler,
    args,
):
    optimizer.zero_grad()

    with autocast():
        # Forward pass
        model_out = model(
            texts, images, text_attention_mask, image_features_attention_mask
        )

        # Compute losses
        logits = model_out.pop("logits")
        labels = model_out.pop("labels")
        if logits is not None and labels is not None:
            losses = loss_fn(
                **model_out, logits=logits, labels=labels, output_dict=True
            )
        else:
            losses = loss_fn(**model_out, output_dict=True)
        logit_scale = model_out.pop("logit_scale")  # Extract logit_scale
        total_loss = sum(losses.values())
        losses["loss"] = total_loss

    if args.log_memory:
        # Log GPU memory stats before backward pass
        allocated_mem = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
        max_allocated_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Peak memory
        reserved_mem = torch.cuda.memory_reserved() / (1024**2)
        max_reserved_mem = torch.cuda.max_memory_reserved() / (1024**2)

        logging.info("Before backward pass:")
        logging.info(f"Allocated Memory: {allocated_mem:.2f} MB")
        logging.info(f"Peak Allocated Memory: {max_allocated_mem:.2f} MB")
        logging.info(f"Reserved Memory: {reserved_mem:.2f} MB")
        logging.info(f"Peak Reserved Memory: {max_reserved_mem:.2f} MB")

    # Backward pass
    backward(total_loss, scaler)

    return losses, logit_scale  # Return logit_scale


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def process_model_accum(
    model,
    optimizer,
    loss,
    autocast,
    texts,
    images,
    text_attention_mask,
    image_features_attention_mask,
    args,
    accum_data: AccumulatedOutput,
    tokenizer,
    scaler,
):
    # First, cache the features without any gradient tracking.
    with torch.no_grad():
        with autocast():
            model_out = model(
                texts, images, text_attention_mask, image_features_attention_mask
            )
            for f in ("logit_scale", "logit_bias"):
                model_out.pop(f, None)

            for key, val in model_out.items():
                if key in accum_data.features:
                    accum_data.features[key].append(val)
                else:
                    if key not in ["logits", "labels"]:
                        accum_data.features[key] = [val]

        accum_data.images.append(images)
        accum_data.texts.append(texts)
        accum_data.text_masks.append(text_attention_mask)
        accum_data.image_masks.append(image_features_attention_mask)
        accum_data.accum_batch_counter += 1

    # If we've accumulated enough batches, process them
    if accum_data.accum_batch_counter % args.accum_freq == 0:
        optimizer.zero_grad()
        accumulated_losses = {}
        for j in range(args.accum_freq):
            images = accum_data.images[j]
            texts = accum_data.texts[j]
            text_attention_mask = accum_data.text_masks[j]
            image_features_attention_mask = accum_data.image_masks[j]

            with autocast():
                model_out = model(
                    texts, images, text_attention_mask, image_features_attention_mask
                )
                inputs_no_accum = {}
                inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                    "logit_scale"
                )
                if "logit_bias" in model_out:
                    inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                # Integrate the recomputed output
                inputs = {}
                for key, val in accum_data.features.items():
                    if None not in val:
                        accumulated = accum_data.features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1 :]
                        )
                if model_out["logits"] is not None and model_out["labels"] is not None:
                    losses = loss(
                        **inputs,
                        **inputs_no_accum,
                        logits=model_out["logits"],
                        labels=model_out["labels"],
                        output_dict=True,
                    )
                else:
                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)

                del inputs
                del inputs_no_accum
                if "contrastive_loss" in losses:
                    if "caption_loss" in losses:
                        batch_loss = (
                            losses["contrastive_loss"] / args.accum_freq
                            + losses["caption_loss"]
                        )
                        accumulated_losses["caption_loss"] = losses["caption_loss"]
                    else:
                        batch_loss = losses["contrastive_loss"] / args.accum_freq
                    accumulated_losses["contrastive_loss"] = (
                        losses["contrastive_loss"] / args.accum_freq
                    )
                else:
                    batch_loss = losses["caption_loss"]
                    accumulated_losses["caption_loss"] = losses["caption_loss"]

            # Backward pass for this batch
            backward(batch_loss, scaler)

        # Now, total_loss contains the sum over accumulated batches
        return accumulated_losses, logit_scale

    else:
        # Continue accumulating batches
        return None, None


def process_batch(
    batch,
    model,
    optimizer,
    loss_fn,
    device,
    input_dtype,
    autocast,
    args,
    accum_data,
    tokenizer,
    scaler,
):
    """
    Processes a single batch: moves data to device, performs forward pass, and computes loss.

    Args:
        batch (dict): Batch data.
        model (torch.nn.Module): The model.
        loss_fn (callable): Loss function.
        device (torch.device): Device to move data to.
        input_dtype (torch.dtype): Data type for inputs.
        autocast (context manager): Autocast context for mixed precision.

    Returns:
        tuple: (loss, model_outputs)
    """
    image_features = batch["image_features"]
    text_features = batch["text_inputs"]
    text_attention_mask = text_features["attention_mask"]
    input_ids = text_features["input_ids"]
    image_features_attention_mask = image_features["attention_mask"]
    image_features = image_features["features"]

    # Move tensors to device
    images = image_features.to(device=device, dtype=input_dtype, non_blocking=True)
    texts = input_ids.to(device=device, non_blocking=True)
    text_attention_mask = text_attention_mask.to(device=device, non_blocking=True)
    image_features_attention_mask = image_features_attention_mask.to(
        device=device, non_blocking=True
    )

    if args.log_memory:
        logging.info(f"shape of images: {images.shape}")
        logging.info(f"shape of texts: {texts.shape}")

    with autocast():
        if args.accum_freq == 1:
            # def process_model_no_accum(model, optimizer, loss, autocast, texts, images, text_attention_mask, image_features_attention_mask, scaler):
            return process_model_no_accum(
                model,
                optimizer,
                loss_fn,
                autocast,
                texts,
                images,
                text_attention_mask,
                image_features_attention_mask,
                scaler,
                args,
            )
        else:
            return process_model_accum(
                model,
                optimizer,
                loss_fn,
                autocast,
                texts,
                images,
                text_attention_mask,
                image_features_attention_mask,
                args,
                accum_data,
                tokenizer,
                scaler,
            )


def update_metrics(loss_meters, losses, n):
    for name, meter in loss_meters.items():
        if name in losses:
            meter.update(losses[name].item(), n=n)


def log_metrics(
    epoch,
    batch_idx,
    dataloader,
    meters,
    optimizer,
    args,
    step,
    tb_writer=None,
    losses=None,
    logit_scale=None,
):
    """
    Logs the training metrics.

    Args:
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        dataloader (DataLoader): The data loader.
        meters (dict): Dictionary of AverageMeter instances.
        optimizer (torch.optim.Optimizer): Optimizer.
        args (Namespace): Command-line arguments.
        step (int): Current training step.
        tokenizer (Tokenizer): Tokenizer object.
        tb_writer (SummaryWriter, optional): TensorBoard writer.
        losses (dict, optional): Loss values to log.
        logit_scale (Tensor, optional): The logit scale tensor.
    """
    batch_size = args.batch_size
    world_size = (
        args.world_size if hasattr(args, "world_size") else 1
    )  # Assuming non-distributed training if not specified
    num_batches_per_epoch = math.ceil(len(dataloader) / args.accum_freq)
    batch_count = (batch_idx // args.accum_freq) + 1
    percent_complete = 100.0 * batch_count / num_batches_per_epoch
    samples_per_second = (args.accum_freq * batch_size * world_size) / meters[
        "batch_time"
    ].val
    samples_per_second_per_gpu = samples_per_second / world_size
    current_lr = optimizer.param_groups[0]["lr"]

    # Log message assembly
    log_message = (
        f"Train Epoch: {epoch + 1} "
        f"[{batch_count}/{num_batches_per_epoch} ({percent_complete:.0f}%)] "
        f"Data Time: {meters['data_time'].avg:.3f}s "
        f"Batch Time: {meters['batch_time'].avg:.3f}s "
        f"Samples/s: {samples_per_second:.2f} "
        f"Samples/s/gpu: {samples_per_second_per_gpu:.2f} "
        f"LR: {current_lr:.6f} "
    )

    # Log the losses if provided
    if losses:
        loss_log = " ".join(
            [
                f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                for loss_name, loss_m in meters["losses"].items()
            ]
        )
        log_message += loss_log + " "

    # Log the logit scale if provided
    if logit_scale is not None:
        logit_scale_scalar = logit_scale.item()
        log_message += f"Logit Scale: {logit_scale_scalar:.3f} "

    logging.info(log_message)

    # Prepare data for TensorBoard and Weights & Biases
    log_data = {
        "data_time": meters["data_time"].val,
        "batch_time": meters["batch_time"].val,
        "samples_per_second": samples_per_second,
        "samples_per_second_per_gpu": samples_per_second_per_gpu,
        "lr": current_lr,
    }
    if logit_scale is not None:
        log_data["scale"] = logit_scale_scalar
    if losses:
        log_data.update({name: val.val for name, val in meters["losses"].items()})
    log_data = {"train/" + name: val for name, val in log_data.items()}

    # Log to TensorBoard
    if tb_writer:
        for name, val in log_data.items():
            tb_writer.add_scalar(name, val, step)

    # Log to Weights & Biases if enabled
    if args.wandb:
        assert wandb is not None, "Please install wandb."
        log_data["step"] = step  # For compatibility with previous versions
        wandb.log(log_data, step=step)

    meters["batch_time"].reset()
    meters["data_time"].reset()


def step_optimizer(model, scaler, optimizer, args):
    if scaler is not None:
        if args.horovod:
            optimizer.synchronize()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
        else:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
        scaler.update()
    else:
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm, norm_type=2.0
            )
        optimizer.step()


def train_one_epoch(
    model,
    data,
    loss_fn,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
    tokenizer,
    tb_writer=None,
):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        data (dict): Dictionary containing data loaders.
        loss_fn (callable): Loss function.
        epoch (int): Current epoch number.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (callable, optional): Learning rate scheduler.
        args (Namespace): Command-line arguments.
        tokenizer (Tokenizer): Tokenizer object.
        tb_writer (SummaryWriter, optional): TensorBoard writer.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    dataloader = data["train"].dataloader
    num_batches = len(dataloader)
    num_batches_per_epoch = math.ceil(num_batches / args.accum_freq)

    meters = {
        "batch_time": AverageMeter(),
        "data_time": AverageMeter(),
        "losses": {key: AverageMeter() for key in ["contrastive_loss", "caption_loss"]},
    }

    end_time = time.time()

    accum_data = AccumulatedOutput()

    for batch_idx, batch in enumerate(dataloader):
        if args.log_memory:
            torch.cuda.reset_peak_memory_stats()
        current_step = epoch * num_batches_per_epoch + (batch_idx // args.accum_freq)

        if not args.skip_scheduler:
            scheduler(current_step)

        # Measure data loading time
        meters["data_time"].update(time.time() - end_time)

        # Process the batch
        losses, logit_scale = process_batch(
            batch,
            model,
            optimizer,
            loss_fn,
            device,
            input_dtype,
            autocast,
            args,
            accum_data,
            tokenizer,
            scaler,
        )

        if not losses:
            continue

        # Optimizer step
        step_optimizer(model, scaler, optimizer, args)

        # Reset batch accumulation
        if args.accum_freq > 1:
            del accum_data
            accum_data = AccumulatedOutput()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # if model has the variable logit_scale:
        if hasattr(unwrap_model(model), "logit_scale"):
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # Gradient clipping
        if args.grad_clip_norm:
            clip_gradients(model, optimizer, args.grad_clip_norm)

        # Update metrics for clip and caption loss
        update_metrics(meters["losses"], losses, n=dataloader.batch_size)

        # Update batch time
        meters["batch_time"].update(time.time() - end_time)
        end_time = time.time()

        # Logging
        if (
            is_master(args)
            and (batch_idx // args.accum_freq) % args.log_every_n_steps == 0
            or (batch_idx // args.accum_freq) == num_batches_per_epoch - 1
        ):
            log_metrics(
                epoch,
                batch_idx,
                dataloader,
                meters,
                optimizer,
                args,
                current_step,
                tb_writer,
                losses,
                logit_scale,
            )

        # Memory monitoring
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Final logging after epoch
    logging.info(f"Epoch [{epoch + 1}/{args.epochs}] completed.")
    for name, meter in meters["losses"].items():
        logging.info(f"Average {name}: {meter.avg:.4f}")
