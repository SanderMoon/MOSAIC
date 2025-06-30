import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


class ZeroShotEvaluator:
    def __init__(self, confidence_level=0.95):
        """
        Initialize the ZeroShotEvaluator with a specified confidence level for bootstrapping.

        Args:
            confidence_level (float): Confidence level for the intervals (e.g., 0.95 for 95% CI).
        """
        self.confidence_level = confidence_level

    def setup_zero_shot_classification(
        self, args, model, tokenizer, specimen_ids, epoch
    ):
        """
        Setup zero-shot classification tasks by loading specimen-class mappings and class prompts.

        Args:
            args: Argument parser containing file paths and device information.
            model: The model used for encoding class descriptions.
            tokenizer: Tokenizer for processing class prompts.

        Returns:
            list: A list of zero-shot classification task dictionaries.
        """
        # Retrieve file paths from args
        specimen_class_files = args.zsc_specimen_class_mapping
        class_prompt_files = args.zsc_class_prompt_mapping

        # Initialize a list to hold all zero-shot classification tasks
        zsc_tasks = []

        # Iterate over each pair of specimen and prompt files
        for specimen_file, prompt_file in zip(specimen_class_files, class_prompt_files):
            # 1. Read the specimen-to-class mapping
            with open(specimen_file, "r") as f:
                specimen_map = json.load(f)
            # specimen_map: { "specimen_id_1": "class_label_1", "specimen_id_2": "class_label_2", ... }

            # Filter the specimen_map to only include specimens in the current dataset
            specimen_map = {
                sid: label for sid, label in specimen_map.items() if sid in specimen_ids
            }

            # 2. Read the class prompt mapping
            with open(prompt_file, "r") as f:
                prompt_map = json.load(f)
            # prompt_map: { "class_label_1": "class_description_1", "class_label_2": "class_description_2", ... }

            # Extract class labels and corresponding prompt texts
            class_labels = list(prompt_map.keys())
            class_prompts = [prompt_map[label] for label in class_labels]

            # Tokenize the prompts
            tokens = tokenizer(
                class_prompts, return_tensors="pt", padding=True, truncation=True
            )

            # Move tokens to the same device as the model if necessary
            tokens = {k: v.to(args.device) for k, v in tokens.items()}

            # 3. Encode the class descriptions to obtain embeddings
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # Access the underlying model
                model_to_use = model.module
            else:
                model_to_use = model
            with torch.no_grad():
                prompt_embeddings = model_to_use.encode_text(**tokens)[
                    1
                ]  # Shape: (num_classes, embedding_dim)

            # Store the data for this task
            task_data = {
                "specimen_to_class_map": specimen_map,
                "class_labels": class_labels,
                "class_prompt_embeddings": prompt_embeddings,
            }

            # Filter the specimen_map to only include specimens in the current dataset
            zsc_tasks.append(task_data)

        # store prompt embeddings to disk
        storage_path = os.path.join(args.log_base_path, f"features_epoch_{epoch}")
        os.makedirs(storage_path, exist_ok=True)
        # store full dictionary as .pt
        torch.save(zsc_tasks, os.path.join(storage_path, "zsc_tasks.pt"))

        return zsc_tasks

    def _bootstrap_confidence_intervals(
        self, y_true, y_pred, y_prob, metric_func, bootstrap_iterations
    ):
        """
        Compute bootstrap confidence intervals for the metrics returned by metric_func.

        Args:
            y_true (np.array): Ground truth labels.
            y_pred (np.array): Predicted labels.
            y_prob (np.array): Predicted probabilities, shape [N, C].
            metric_func (callable): A function that takes (y_true, y_pred, y_prob) and returns a dict of metrics.
            bootstrap_iterations (int): Number of bootstrap iterations.

        Returns:
            dict: Dictionary where keys are metric names and values are tuples of
                  (mean_metric, lower_bound, upper_bound).
        """
        N = len(y_true)
        metrics_list = []

        for _ in range(bootstrap_iterations):
            # Sample indices with replacement
            sample_indices = np.random.randint(0, N, size=N)
            y_true_sample = y_true[sample_indices]
            y_pred_sample = y_pred[sample_indices]
            y_prob_sample = y_prob[sample_indices]

            # Compute metrics for this bootstrap sample
            m = metric_func(y_true_sample, y_pred_sample, y_prob_sample)
            metrics_list.append(m)

        # Convert list of dicts into a dict of lists
        all_metrics = {}
        for m in metrics_list:
            for k, v in m.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        # Compute mean and percentile-based CI
        ci_results = {}
        alpha = self.confidence_level
        lower_percentile = (1 - alpha) / 2 * 100
        upper_percentile = (1 + alpha) / 2 * 100

        for k, vals in all_metrics.items():
            vals = np.array(vals)
            mean_val = np.mean(vals)
            lower_bound = np.percentile(vals, lower_percentile)
            upper_bound = np.percentile(vals, upper_percentile)
            ci_results[k] = (mean_val, lower_bound, upper_bound)

        return ci_results

    def _compute_binary_metrics(self, y_true, y_pred, y_prob):
        """
        Compute binary classification metrics including AUC-ROC, Precision, Recall, and F1.

        Args:
            y_true (np.array): True binary labels.
            y_pred (np.array): Predicted binary labels.
            y_prob (np.array): Predicted probabilities for the positive class.

        Returns:
            dict: Dictionary of computed binary metrics.
        """
        metrics = {}
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = np.nan  # Handle cases where AUC cannot be computed
        return metrics

    def _compute_multiclass_metrics(self, y_true, y_pred, y_prob, class_labels):
        """
        Compute multi-class metrics such as accuracy, balanced accuracy, macro-F1, and macro AUC-ROC.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
            y_prob (np.array): Predicted probabilities.
            class_labels (list): List of class label strings.

        Returns:
            dict: Dictionary of computed multi-class metrics.
        """
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        # Macro-average F1
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # For AUC in multiclass, we can do one-vs-rest
        n_classes = len(class_labels)
        y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))
        # Attempt macro AUC-ROC
        try:
            metrics["macro_auc_roc"] = roc_auc_score(
                y_true_binarized, y_prob, average="macro", multi_class="ovr"
            )
        except ValueError:
            metrics["macro_auc_roc"] = np.nan
        return metrics

    def _store_predictions(
        self,
        y_true,
        y_pred,
        y_prob,
        specimen_ids,
        log_dir,
        epoch,
        task_idx,
        metrics=None,
    ):
        """
        Stores predictions and metrics to a JSON file.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
            specimen_ids (list of str): Specimen IDs.
            log_dir (str): Directory to store the results.
            epoch (int): Current epoch number.
            task_idx (int): Index of the task.
            metrics (dict, optional): Dictionary of global metrics and their confidence intervals.
        """
        try:
            if not (len(y_true) == len(y_pred) == len(specimen_ids)):
                raise ValueError("Input lists must have the same length.")

            # Store per-specimen predictions
            predictions = {}
            for i, sid in enumerate(specimen_ids):
                predictions[sid] = {
                    "true_label": int(y_true[i]),
                    "predicted_label": int(y_pred[i]),
                    "probabilities": [float(p) for p in y_prob[i]],
                }

            # Structure the JSON with predictions and metrics
            results = {"predictions": predictions}

            if metrics:
                # clean metrics to removed NaN values, set to None
                metrics = {k: None if np.isnan(v) else v for k, v in metrics.items()}
                results["metrics"] = metrics

            # Define the result path and filename
            result_path = os.path.join(log_dir, "zero_shot_results")
            os.makedirs(result_path, exist_ok=True)
            filename = f"zero_shot_results_epoch_{epoch}_task_{task_idx}.json"
            file_path = os.path.join(result_path, filename)

            # Write to JSON file
            with open(file_path, "w") as f:
                json.dump(results, f, indent=4)

            logging.info(f"Stored predictions and metrics to {file_path}")

        except (ValueError, IndexError, TypeError) as e:
            logging.error(f"Error storing predictions: {e}")
        except OSError as e:
            logging.error(f"Error writing to file: {e}")

    def zero_shot_classification(
        self,
        image_features,
        specimen_ids,
        zsc_task,
        task_idx,
        epoch,
        log_dir,
        logit_scale,
        zsc_bootstraps=1,
    ):
        """
        Perform zero-shot classification for a given task with optional bootstrapping.

        Args:
            image_features (Tensor): Image features to classify (shape: [N, D]).
            specimen_ids (list of str): A list of specimen IDs corresponding to each image feature.
            zsc_task (dict): Zero-shot classification task data, with keys:
                - "specimen_to_class_map": dict mapping specimen_id -> class_label
                - "class_labels": list of class label strings
                - "class_prompt_embeddings": Tensor of class embeddings [num_classes, D]
            task_idx (int): Index of the task in the list of tasks.
            epoch (int): Current epoch number.
            log_dir (str): Directory to store the results.
            zsc_bootstraps (int): Number of bootstrap iterations. If 1, no bootstrapping is performed.

        Returns:
            dict: Zero-shot classification metrics for the task (with or without bootstrap confidence intervals).
        """
        specimen_to_class_map = zsc_task["specimen_to_class_map"]
        class_labels = zsc_task["class_labels"]
        class_prompt_embeddings = zsc_task["class_prompt_embeddings"]

        # Ensure embeddings and features are on the same device
        device = class_prompt_embeddings.device
        image_features = image_features.to(device)

        # Step 1: Filter out specimen_ids not present in specimen_to_class_map
        valid_mask = [sid in specimen_to_class_map for sid in specimen_ids]
        num_invalid = len(specimen_ids) - sum(valid_mask)

        if num_invalid > 0:
            logging.warning(
                f"{num_invalid} specimen_ids not found in specimen_to_class_map and will be excluded from evaluation."
            )

        # If no valid specimens, return empty metrics
        if sum(valid_mask) == 0:
            logging.warning(
                "No valid specimen_ids found for zero-shot classification. Returning empty metrics."
            )
            return {}

        # Apply mask to image_features and specimen_ids
        valid_mask_tensor = torch.tensor(
            valid_mask, dtype=torch.bool, device=image_features.device
        )
        image_features = image_features[valid_mask_tensor]
        filtered_specimen_ids = [
            sid for sid, valid in zip(specimen_ids, valid_mask) if valid
        ]

        if isinstance(logit_scale, torch.Tensor):
            logit_scale = logit_scale.to(device)
            image_features = logit_scale * image_features
        else:
            image_features = logit_scale * image_features

        # Normalize embeddings and features
        image_features = F.normalize(image_features, p=2, dim=1)
        class_prompt_embeddings = F.normalize(class_prompt_embeddings, p=2, dim=1)

        # Compute similarity scores by dot product
        # similarity: [N_valid, C]
        similarity = image_features @ class_prompt_embeddings.T

        # Convert similarity to float32 before applying softmax
        similarity = similarity.to(torch.float32)

        # Convert similarity scores to probabilities using softmax
        probs = F.softmax(similarity, dim=-1).cpu().numpy()  # shape: [N_valid, C]

        # Predicted class index is the one with the highest similarity
        predicted_indices = similarity.argmax(dim=-1).cpu().numpy()
        # Ensure class_labels are properly mapped
        predicted_classes = [int(class_labels[i]) for i in predicted_indices]

        # Get ground truth classes from filtered specimen_ids
        ground_truth_classes = [
            specimen_to_class_map[sid] for sid in filtered_specimen_ids
        ]

        y_true = np.array(ground_truth_classes)
        y_pred = np.array(predicted_classes)

        # Define metric functions based on classification type
        if len(class_labels) == 2:
            # Binary classification
            def metric_func(yt, yp, ypb):
                # ypb is the probability for the positive class
                return self._compute_binary_metrics(yt, yp, ypb)

        else:
            # Multi-class classification
            def metric_func(yt, yp, ypb):
                return self._compute_multiclass_metrics(yt, yp, ypb, class_labels)

        # Compute point estimates
        point_metrics = metric_func(
            y_true, y_pred, probs[:, 1] if len(class_labels) == 2 else probs
        )

        # Initialize metrics dictionary with task index prefix
        metrics = {f"task_{task_idx}_{k}": v for k, v in point_metrics.items()}

        # Perform bootstrapping if requested
        if zsc_bootstraps > 1:
            # Compute confidence intervals using bootstrapping
            ci_results = self._bootstrap_confidence_intervals(
                y_true,
                y_pred,
                probs[:, 1] if len(class_labels) == 2 else probs,
                metric_func,
                zsc_bootstraps,
            )

            # Add CI results to metrics with task index prefix
            for metric_name, (mean_val, lower_ci, upper_ci) in ci_results.items():
                metrics[f"task_{task_idx}_{metric_name}_mean"] = mean_val
                metrics[f"task_{task_idx}_{metric_name}_ci_lower"] = lower_ci
                metrics[f"task_{task_idx}_{metric_name}_ci_upper"] = upper_ci

        # Store predictions and metrics together
        self._store_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=probs,
            specimen_ids=filtered_specimen_ids,
            log_dir=log_dir,
            epoch=epoch,
            task_idx=task_idx,
            metrics=metrics,  # Pass the metrics dictionary
        )

        return metrics
