import json
import logging
import os
import time

import torch

try:
    import wandb
except ImportError:
    wandb = None

import math

from torch import distributed as dist

from mosaic_train.distributed import is_master, synchronize
from mosaic_train.eval_captions import CaptionEvaluator
from mosaic_train.eval_retrieval import RetrievalEvaluator
from mosaic_train.eval_zero_shot import ZeroShotEvaluator
from mosaic_train.logger import AverageMeter
from mosaic_train.precision import get_autocast

try:
    import torch.distributed.nn

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class Evaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate(
        self,
        model,
        data,
        epoch,
        args,
        loss,
        input_dtype,
        amp,
        tb_writer=None,
        tokenizer=None,
    ):
        metrics = {}

        batch_time_m = AverageMeter()
        end = time.time()

        caption_evaluator = None
        if args.save_captions:
            caption_evaluator = CaptionEvaluator(args.eval_metric_ci)

        device = torch.device(args.device)
        model.eval()
        autocast_context = get_autocast(args.precision)

        dataloader = data["val"].dataloader

        num_samples = 0

        num_batches = len(dataloader)
        num_batches_per_epoch = math.ceil(num_batches / args.accum_freq)

        cumulative_total_loss, cumulative_gen_loss, cumulative_clip_loss = 0.0, 0.0, 0.0

        if args.accum_freq > 1:
            accum_features = {}

        all_image_features, all_text_features, all_specimen_ids, all_captions = (
            [],
            [],
            [],
            {},
        )
        (
            batch_image_features,
            batch_text_features,
            batch_specimen_ids,
            batch_captions,
        ) = (
            [],
            [],
            [],
            {},
        )
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                batch_data = self._prepare_batch(batch, device, input_dtype, amp)
                if (epoch + 1) % args.caption_val_frequency == 0:
                    new_captions = self._maybe_generate_captions(
                        args, batch_data, model, tokenizer, autocast_context
                    )
                    batch_captions.update(new_captions)

                if args.accum_freq == 1:
                    with autocast_context():
                        model_out = model(
                            batch_data["texts"],
                            batch_data["images"],
                            batch_data["text_attention_mask"],
                            batch_data["image_features_attention_mask"],
                        )
                        logit_scale = model_out["logit_scale"]

                        # Calculate loss using CoCaLoss
                        # Compute losses
                        logits = model_out.pop("logits")
                        labels = model_out.pop("labels")
                        if logits is not None and labels is not None:
                            losses = loss(
                                **model_out,
                                logits=logits,
                                labels=labels,
                                output_dict=True,
                            )
                        else:
                            losses = loss(**model_out, output_dict=True)

                        # Store features for metrics calculation
                        if model_out["image_features"] is not None:
                            batch_image_features.append(
                                model_out["image_features"].cpu()
                            )
                        batch_text_features.append(model_out["text_features"].cpu())
                        batch_specimen_ids.append(batch_data["specimen_ids"])

                    (
                        cumulative_clip_loss,
                        cumulative_gen_loss,
                        cumulative_total_loss,
                        new_num_samples,
                    ) = self._update_cum_loss(
                        cumulative_clip_loss,
                        cumulative_gen_loss,
                        cumulative_total_loss,
                        losses,
                        batch_data["batch_size"],
                    )

                # in case of batch accumulation
                else:
                    # Cache the features from the model output without gradient tracking
                    model_out = self._cache_features_no_grad(
                        model,
                        accum_features,
                        autocast_context,
                        batch_data,
                        batch_image_features,
                        batch_text_features,
                        batch_specimen_ids,
                    )

                    # If (i + 1) % accum_freq is not zero and it's not the last imperfect batch, move on to the next batch
                    if ((i + 1) % args.accum_freq) > 0 and (i + 1) != len(dataloader):
                        continue

                    # Once we have accumulated enough batches, preprocess the batches and calculate the loss
                    inputs, inputs_no_accum, logit_scale = (
                        self._preprocess_accum_batches(
                            accum_features, model_out, tokenizer
                        )
                    )
                    if (
                        model_out["logits"] is not None
                        and model_out["labels"] is not None
                    ):
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

                    (
                        cumulative_clip_loss,
                        cumulative_gen_loss,
                        cumulative_total_loss,
                        new_num_samples,
                    ) = self._update_cum_loss(
                        cumulative_clip_loss,
                        cumulative_gen_loss,
                        cumulative_total_loss,
                        losses,
                        batch_data["batch_size"],
                        args.accum_freq,
                    )

                num_samples += new_num_samples

                if args.world_size > 1:
                    logging.info(
                        f"Rank {args.rank} - Batch {i} - Loss: {cumulative_total_loss / num_samples:.6f}"
                    )
                    logging.info(
                        f"Rank {args.rank} - Collecting features from batch {i} to master"
                    )
                    # transform all_image_features and all_text_features into tensors
                    batch_image_features, batch_text_features, batch_specimen_ids = (
                        self.gather_features(
                            args,
                            batch_image_features,
                            batch_text_features,
                            batch_specimen_ids,
                        )
                    )

                    batch_captions = self.gather_captions_to_master(
                        args, batch_captions
                    )

                if is_master(args):
                    all_image_features.extend(batch_image_features)
                    all_text_features.extend(batch_text_features)
                    all_specimen_ids.extend(batch_specimen_ids)
                    all_captions.update(batch_captions)

                    logging.info(
                        f"Master - Batch {i} - Successfully collected features from batch {i}"
                    )

                # reset batch collectors
                (
                    batch_image_features,
                    batch_text_features,
                    batch_specimen_ids,
                    batch_captions,
                ) = (
                    [],
                    [],
                    [],
                    {},
                )

                if is_master(args) and (
                    (i // args.accum_freq) % args.log_every_n_steps == 0
                    or (i // args.accum_freq) == num_batches_per_epoch - 1
                ):
                    batch_count = (i // args.accum_freq) + 1
                    num_batches_per_epoch = math.ceil(len(dataloader) / args.accum_freq)
                    percent_complete = 100.0 * batch_count / num_batches_per_epoch
                    logging.info(
                        f"Eval Epoch: {epoch} [{batch_count}/{num_batches_per_epoch} ({percent_complete:.0f}%)]\t"
                        f"Avg Total Loss: {cumulative_total_loss / num_samples:.6f}\t"
                        f"Avg Clip Loss: {cumulative_clip_loss / num_samples:.6f}\t"
                        f"Avg Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"
                    )

                batch_time_m.update(time.time() - end)
                end = time.time()

                # reset accumulators
                if args.accum_freq > 1:
                    accum_features = {}

            # turn into a tensor of shape [N, D]
            if all_image_features:
                all_image_features = torch.cat(all_image_features, dim=0)
            else:
                all_image_features = torch.empty(0)

            if all_text_features:
                all_text_features = torch.cat(all_text_features, dim=0)
            else:
                all_text_features = torch.empty(0)
            all_specimen_ids = (
                [item for sublist in all_specimen_ids for item in sublist]
                if all_specimen_ids
                else []
            )

            if is_master(args):
                self._store_image_features(
                    args, all_image_features, all_text_features, all_specimen_ids, epoch
                )
                self._update_metrics(
                    model,
                    metrics,
                    args,
                    all_image_features,
                    all_text_features,
                    all_specimen_ids,
                    logit_scale,
                    caption_evaluator,
                    all_captions,
                    cumulative_total_loss,
                    cumulative_clip_loss,
                    cumulative_gen_loss,
                    num_samples,
                    epoch,
                    data,
                )

                self._log_metrics(
                    metrics, epoch, args, all_captions, data, tb_writer=tb_writer
                )

            if args.world_size > 1:
                synchronize(args)

            return metrics

    def _store_image_features(
        self, args, all_image_features, all_text_features, all_specimen_ids, epoch
    ):
        """
        Store image and text features to disk for later retrieval.

        Args:
            args: Configuration arguments.
            all_image_features (list of tensors): Image features from all samples.
            all_text_features (list of tensors): Text features from all samples.
            all_specimen_ids (list): Specimen IDs for all samples.
            epoch (int): The current epoch number.
        """
        if not is_master(args):
            return

        if not args.log_base_path:
            return

        storage_path = os.path.join(args.log_base_path, f"features_epoch_{epoch}")

        # Ensure the output directory exists
        os.makedirs(storage_path, exist_ok=True)

        # Save image and text features to disk
        image_features_path = os.path.join(
            storage_path, f"image_features_epoch_{epoch}.pt"
        )
        text_features_path = os.path.join(
            storage_path, f"text_features_epoch_{epoch}.pt"
        )
        specimen_ids_path = os.path.join(
            storage_path, f"specimen_ids_epoch_{epoch}.json"
        )

        torch.save(all_image_features, image_features_path)
        torch.save(all_text_features, text_features_path)

        with open(specimen_ids_path, "w") as f:
            json.dump(all_specimen_ids, f)

        logging.info(f"Saved image features to: {image_features_path}")
        logging.info(f"Saved text features to: {text_features_path}")
        logging.info(f"Saved specimen IDs to: {specimen_ids_path}")

    def collect_batch_outputs_to_master(
        self,
        args,
        all_image_features,
        all_text_features,
        all_specimen_ids,
        all_captions,
        batch_image_features,
        batch_text_features,
        batch_specimen_ids,
        batch_captions,
    ):
        (
            total_batch_image_features,
            total_batch_text_features,
            total_batch_specimen_ids,
        ) = self.gather_features(
            args, batch_image_features, batch_text_features, batch_specimen_ids
        )
        total_batch_captions = self.gather_captions_to_master(args, batch_captions)
        if is_master(args):
            all_image_features.extend(total_batch_image_features)
            all_text_features.extend(total_batch_text_features)
            all_specimen_ids.extend(total_batch_specimen_ids)
            all_captions.update(total_batch_captions)

    def gather_features(
        self, args, all_image_features, all_text_features, all_specimen_ids
    ):
        """
        Gathers image and text features from all processes in a distributed setting.
        Supports both PyTorch DDP and Horovod backends.

        Args:
            args: Configuration arguments.
            all_image_features (list of tensors): Local image features.
            all_text_features (list of tensors): Local text features.

        Returns:
            tuple: (gathered_image_features, gathered_text_features, gathered_specimen_ids)
            Each is gathered from all processes and returned on the master rank.
            On non-master ranks, returns (None, None, None).
        """
        if not args.distributed and not getattr(args, "horovod", False):
            # Single-process training: return local features in lists
            return (
                [torch.cat(all_image_features, dim=0)],
                [torch.cat(all_text_features, dim=0)],
                all_specimen_ids,
            )

        if getattr(args, "horovod", False):
            if hvd is None:
                raise ImportError(
                    "Horovod is not installed but 'horovod' is set to True."
                )
            if not hvd.is_initialized():
                raise RuntimeError(
                    "Horovod is not initialized. Call hvd.init() before using."
                )

            # Concatenate local features
            local_image_features = torch.cat(all_image_features, dim=0)
            local_text_features = torch.cat(all_text_features, dim=0)
            local_specimen_ids = all_specimen_ids

            # Use Horovod's allgather_object for variable-sized tensors
            gathered_image_features = hvd.allgather_object(local_image_features)
            gathered_text_features = hvd.allgather_object(local_text_features)
            gathered_specimen_ids = hvd.allgather_object(local_specimen_ids)

            return (
                gathered_image_features,
                gathered_text_features,
                gathered_specimen_ids,
            )
        else:
            # PyTorch DDP
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available.")
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed is not initialized.")

            rank = args.rank
            world_size = args.world_size

            # Concatenate local features and move to CPU for serialization
            local_image_features = torch.cat(all_image_features, dim=0).cpu()
            local_text_features = torch.cat(all_text_features, dim=0).cpu()
            local_specimen_ids = [
                item for sublist in all_specimen_ids for item in sublist
            ]

            if rank == 0:
                # Prepare lists to receive tensors from all ranks
                gathered_image_features = [None for _ in range(world_size)]
                gathered_text_features = [None for _ in range(world_size)]
                gathered_specimen_ids = [None for _ in range(world_size)]
            else:
                gathered_image_features = None
                gathered_text_features = None
                gathered_specimen_ids = None

            # Gather tensors to the master node (rank 0)
            dist.gather_object(local_image_features, gathered_image_features, dst=0)
            dist.gather_object(local_text_features, gathered_text_features, dst=0)
            dist.gather_object(local_specimen_ids, gathered_specimen_ids, dst=0)

            if rank == 0:
                return (
                    gathered_image_features,
                    gathered_text_features,
                    gathered_specimen_ids,
                )
            else:
                # Other ranks return None
                return None, None, None

    def gather_captions_to_master(self, args, local_captions):
        """
        Gathers captions dictionaries from all processes to the master node (rank 0).

        Args:
            args: Configuration arguments containing distributed settings.
            local_captions (dict): The captions dictionary from the local process.

        Returns:
            dict or None: Merged captions dictionary on rank 0, None on other ranks.
        """
        if not args.distributed:
            return local_captions  # Single-process execution

        # Ensure torch.distributed is initialized
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available.")
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized.")

        rank = args.rank
        world_size = args.world_size

        # Only the master node needs to gather the objects
        if rank == 0:
            gathered_captions = [None for _ in range(world_size)]
        else:
            gathered_captions = None  # Other ranks don't need to allocate space

        # Perform the gather_object operation
        dist.gather_object(local_captions, gathered_captions, dst=0)

        # On the master node, merge the gathered captions
        if rank == 0:
            all_captions = {}
            for cap in gathered_captions:
                all_captions.update(cap)
            return all_captions
        else:
            return None  # Other ranks return None

    def _prepare_batch(self, batch, device, input_dtype, amp):
        """
        Prepares and casts the batch based on AMP settings.

        Args:
            batch: The raw batch data.
            device: The device to move tensors to.
            input_dtype: The desired input dtype when AMP is not used.
            amp (bool): Flag indicating whether AMP is enabled.

        Returns:
            Tuple containing processed tensors and patient_ids.
        """
        image_features = batch["image_features"]
        text_features = batch["text_inputs"]
        patient_ids = batch["patient_ids"]
        specimen_ids = batch["specimen_ids"]

        text_attention_mask = text_features["attention_mask"]
        input_ids = text_features["input_ids"]
        image_features_attention_mask = image_features["attention_mask"]
        image_features = image_features["features"]

        texts = input_ids.to(device=device, non_blocking=True)
        image_features_attention_mask = image_features_attention_mask.to(
            device=device, non_blocking=True
        )
        text_attention_mask = text_attention_mask.to(device=device, non_blocking=True)
        image_features_attention_mask = image_features_attention_mask.to(
            device=device, non_blocking=True
        )

        if amp:
            # When using AMP, keep inputs in Float32 and let autocast handle lower precision
            images = image_features.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        else:
            # When not using AMP, cast inputs to the desired dtype
            images = image_features.to(
                device=device, dtype=input_dtype, non_blocking=True
            )

        batch_size = len(images)

        return {
            "images": images,
            "texts": texts,
            "image_features_attention_mask": image_features_attention_mask,
            "text_attention_mask": text_attention_mask,
            "patient_ids": patient_ids,
            "specimen_ids": specimen_ids,
            "batch_size": batch_size,
        }

    def _update_cum_loss(
        self,
        cumulative_clip_loss,
        cumulative_gen_loss,
        cumulative_total_loss,
        losses,
        batch_size,
        accum_freq=1,
    ):
        if "caption_loss" in losses:
            gen_loss = losses["caption_loss"]
        else:
            gen_loss = torch.tensor(0.0)
        if "contrastive_loss" in losses:
            clip_loss = losses["contrastive_loss"] / accum_freq
        else:
            clip_loss = torch.tensor(0.0)

        # Scale and update cumulative losses
        cumulative_clip_loss += clip_loss.item() * batch_size * accum_freq
        cumulative_gen_loss += gen_loss.item() * batch_size * accum_freq
        cumulative_total_loss += (
            (clip_loss.item() + gen_loss.item()) * batch_size * accum_freq
        )
        num_samples = batch_size * accum_freq
        return (
            cumulative_clip_loss,
            cumulative_gen_loss,
            cumulative_total_loss,
            num_samples,
        )

    def _cache_features_no_grad(
        self,
        model,
        accum_features,
        autocast,
        batch_data,
        all_image_features,
        all_text_features,
        all_specimen_ids,
    ):
        with autocast():
            model_out = model(
                batch_data["texts"],
                batch_data["images"],
                batch_data["text_attention_mask"],
                batch_data["image_features_attention_mask"],
            )
            for f in "logit_bias":
                model_out.pop(f, None)

            for key, val in model_out.items():
                if val is not None:
                    if key in accum_features:
                        accum_features[key].append(val)
                    else:
                        accum_features[key] = [val]

            # Store features for metrics calculation
            if model_out["image_features"] is not None:
                all_image_features.append(model_out["image_features"].cpu())
            all_text_features.append(model_out["text_features"].cpu())
            all_specimen_ids.append(batch_data["specimen_ids"])

        return model_out

    def _preprocess_accum_batches(self, accum_features, model_out, tokenizer):
        # pad the logits and labels to the same length
        # obtain the logits
        inputs_no_accum = {}

        # Taking an arbitrary logit scale should be OK, given that the scale is unchanged across batches until backprop
        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
        if "logit_bias" in model_out:
            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

        accum_features.pop("logit_scale", None)
        inputs = {}
        for key, val in accum_features.items():
            if key not in ["logits", "labels"]:
                accumulated = accum_features[key]
                inputs[key] = torch.cat(accumulated, dim=0)

        return inputs, inputs_no_accum, logit_scale

    def _update_metrics(
        self,
        model,
        metrics,
        args,
        all_image_features,
        all_text_features,
        all_specimen_ids,
        logit_scale,
        caption_evaluator: CaptionEvaluator,
        captions,
        cumulative_total_loss,
        cumulative_clip_loss,
        cumulative_gen_loss,
        num_samples,
        epoch,
        data,
    ):
        val_metrics = {}
        zsc_metrics = {}
        fsc_metrics = {}
        if len(all_image_features) != 0:
            retrieval_eval = RetrievalEvaluator(
                log_dir=args.log_base_path, confidence_level=args.eval_metric_ci
            )
            val_metrics = retrieval_eval.compute_metrics(
                all_image_features,
                all_text_features,
                logit_scale,
                all_specimen_ids,
                epoch,
                bootstraps=args.eval_metric_bootstraps,
            )

            if args.zsc_specimen_class_mapping and args.zsc_class_prompt_mapping:
                zsc = ZeroShotEvaluator(args.eval_metric_ci)
                zsc_tasks = zsc.setup_zero_shot_classification(
                    args, model, self.tokenizer, all_specimen_ids, epoch
                )
                for i, task in enumerate(zsc_tasks):
                    task_metrics = zsc.zero_shot_classification(
                        all_image_features,
                        all_specimen_ids,
                        task,
                        i,
                        epoch,
                        args.log_base_path,
                        logit_scale,
                        args.eval_metric_bootstraps,
                    )
                    zsc_metrics.update(task_metrics)

        caption_metrics = {}
        if args.save_captions and (epoch + 1) % args.caption_val_frequency == 0:
            caption_metrics = caption_evaluator.compute_metrics(
                captions, args.log_base_path, epoch, args.eval_metric_bootstraps
            )

        avg_total_loss = cumulative_total_loss / num_samples
        avg_clip_loss = cumulative_clip_loss / num_samples
        avg_gen_loss = cumulative_gen_loss / num_samples

        metrics.update(
            {
                **val_metrics,
                "val_total_loss": avg_total_loss,
                "val_clip_loss": avg_clip_loss,
                "val_generative_loss": avg_gen_loss,
                "epoch": epoch,
                "num_samples": num_samples,
                **caption_metrics,
                **zsc_metrics,  # Add caption metrics
                **fsc_metrics,
            }
        )

    def _maybe_generate_captions(
        self, args, batch_data, model, tokenizer, autocast_context
    ):
        """
        Generates captions for a batch of images and maps them to patient IDs alongside ground truth captions.

        This method decodes ground truth captions, generates new captions using the provided model,
        and constructs a dictionary mapping each patient ID to its corresponding ground truth
        and generated captions.

        Args:
            args: Configuration arguments. Must contain the attribute `save_captions` (bool)
                indicating whether to perform caption storage.
            batch_data (dict): A dictionary containing model inputs with the following keys:
                - 'texts': List or tensor of tokenized ground truth captions.
                - 'images': Tensor of image features.
                - 'image_features_attention_mask': Tensor representing attention masks for image features.
                - 'patient_ids': List of patient IDs corresponding to the batch.
            model: The model instance used to generate captions. It should have a `generate` method.
            tokenizer: The tokenizer instance used to decode token IDs into strings.

        Returns:
            dict: A dictionary mapping each patient ID to a sub-dictionary containing:
                - "ground_truth": The decoded ground truth caption (str).
                - "generated": The decoded generated caption (str).
                Example:
                {
                    "PATIENT_ID_1": { "ground_truth": "Ground truth caption 1", "generated": "Generated caption 1" },
                    "PATIENT_ID_2": { "ground_truth": "Ground truth caption 2", "generated": "Generated caption 2" },
                    ...
                }
                If `args.save_captions` is False, returns an empty dictionary.
        """
        # Check if caption saving is enabled
        if not getattr(args, "save_captions", False):
            logging.debug("Caption saving is disabled. Exiting method.")
            return {}

        start_time = time.time()
        captions = {}

        # Extract necessary data from batch_data with default fallbacks
        specimen_ids = batch_data.get("specimen_ids", [])
        ground_truth_texts = batch_data.get("texts", None)
        image_features = batch_data.get("images", None)
        image_attention_mask = batch_data.get("image_features_attention_mask", None)

        # Validate the presence of required data
        if not specimen_ids:
            logging.warning("No patient IDs found in batch_data.")
            return {}
        if ground_truth_texts is None:
            logging.warning("No ground truth texts found in batch_data.")
        if image_features is None:
            logging.warning("No image features found in batch_data.")
            return {}
        if image_attention_mask is None:
            logging.warning("No image attention mask found in batch_data.")

        # Decode ground truth captions
        for text, specimen_id in zip(ground_truth_texts, specimen_ids):
            try:
                # Decode text and remove any leading/trailing whitespace
                decoded_text = tokenizer.decode(text, skip_special_tokens=True).strip()
                captions[specimen_id] = {"ground_truth": decoded_text, "generated": ""}
            except Exception as e:
                logging.error(
                    f"Error decoding ground truth for specimen {specimen_id}: {e}"
                )
                captions[specimen_id] = {"ground_truth": "", "generated": ""}

        # Generate captions using the model without beam search
        num_beams = args.val_gen_num_beams
        num_beam_groups = args.val_gen_num_beam_groups
        top_k = args.val_gen_top_k
        max_seq_len = args.caption_val_max_seq_len
        generation_type = "top_k" if top_k else "beam_search"

        # Determine whether the model is wrapped in DDP
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # Access the underlying model
            model_to_use = model.module
        else:
            model_to_use = model
        try:
            with autocast_context():
                generated_ids = model_to_use.generate(
                    image_features,
                    sot_token_id=tokenizer.all_special_ids[0],
                    eos_token_id=tokenizer.all_special_ids[1],
                    pad_token_id=tokenizer.all_special_ids[3],
                    max_seq_len=max_seq_len,
                    visual_attention_mask=image_attention_mask,
                    generation_type=generation_type,
                    seq_len=args.caption_val_max_seq_len,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    top_k=top_k,
                )

        except Exception as e:
            logging.error(f"Error during caption generation: {e}")
            # Populate generated captions with empty strings in case of failure
            raise
            # for specimen_id in specimen_ids:
            #     captions[specimen_id]['generated'] = ""
            # return captions

        # Decode generated captions and update the captions dictionary
        for idx, caption_ids in enumerate(generated_ids):
            specimen_id = (
                specimen_ids[idx] if idx < len(specimen_ids) else f"UNKNOWN_{idx}"
            )
            try:
                # Decode caption and remove any leading/trailing whitespace
                decoded_caption = tokenizer.decode(
                    caption_ids, skip_special_tokens=True
                ).strip()
                captions[specimen_id]["generated"] = decoded_caption
                logging.info(f"Patient {specimen_id}: {decoded_caption}")
            except Exception as e:
                logging.error(
                    f"Error decoding generated caption for patient {specimen_id}: {e}"
                )
                captions[specimen_id]["generated"] = ""

        # Log the time taken for the entire process
        elapsed_time = time.time() - start_time
        logging.info(f"Time taken to generate captions: {elapsed_time:.2f} seconds")

        return captions

    def _log_metrics(
        self,
        metrics,
        epoch,
        args,
        captions,
        data,
        tb_writer=None,
    ):
        if not metrics:
            return

        logging.info(
            f"Epoch: {epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

        log_data = {"val/" + name: val for name, val in metrics.items()}

        self._maybe_save_logs(args, log_data, metrics, epoch, tb_writer)
        self._maybe_log_wandb(args, log_data, epoch, data)

    def _maybe_save_logs(self, args, log_data, metrics, epoch, tb_writer=None):
        if args.save_logs:
            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

    def _maybe_log_wandb(self, args, log_data, epoch, data):
        if args.wandb:
            assert wandb is not None, "Please install wandb."
            if "train" in data:
                dataloader = data["train"].dataloader
                num_batches_per_epoch = dataloader.num_batches // args.accum_freq
                step = num_batches_per_epoch * epoch
            else:
                step = None
            log_data["epoch"] = epoch
            wandb.log(log_data, step=step)
