import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F


class RetrievalEvaluator:
    """
    A class to evaluate cross-modal retrieval metrics between image and text features.
    It computes metrics like Mean Rank, Median Rank, and Recall@K, and stores the
    results mapped to their corresponding specimen IDs. Additionally, it computes
    confidence intervals for these metrics using bootstrapping, where we only
    resample from the query set and keep the reference set static.
    """

    def __init__(self, log_dir, confidence_level=0.95):
        """
        Initializes the RetrievalEvaluator with the directory where results will be stored.
        Also sets the confidence level for bootstrapping.

        Args:
            log_dir (str): Directory path to store the JSON results.
            confidence_level (float): Confidence level for bootstrapping (e.g., 0.95 for 95% CI).
        """
        self.log_dir = os.path.join(log_dir, "retrieval_metrics")
        os.makedirs(self.log_dir, exist_ok=True)
        # Define the metrics to compute
        self.metric_names = ["mean_rank", "median_rank", "R@1", "R@5", "R@10"]
        self.confidence_level = confidence_level
        logging.basicConfig(level=logging.INFO)

    def compute_metrics(
        self,
        image_features,
        text_features,
        logit_scale,
        all_specimen_ids,
        epoch,
        device="cpu",
        bootstraps=1,
    ):
        """
        Computes cross-modal retrieval metrics and stores the results, including confidence intervals if bootstraps > 1.

        Args:
            image_features (torch.Tensor): Tensor of image features of shape (N, D).
            text_features (torch.Tensor): Tensor of text features of shape (N, D).
            logit_scale (float or torch.Tensor): Scaling factor for logits.
            all_specimen_ids (list): List of specimen IDs corresponding to the features.
            epoch (int): Current epoch number for logging purposes.
            device (str, optional): Device to perform computations on. Defaults to 'cpu'.
            bootstraps (int, optional): Number of bootstrap iterations for confidence intervals. Defaults to 1.

        Returns:
            dict: Dictionary containing the computed retrieval metrics and their confidence intervals.
        """
        # Ensure the features are on the specified device
        image_features = image_features.float()
        text_features = text_features.float()
        logit_scale = logit_scale.float()

        # Normalize the features
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-8)
        text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-8)

        # Compute the scaled logits
        if isinstance(logit_scale, torch.Tensor):
            logit_scale = logit_scale.to(device)
            logits_per_image = (
                (logit_scale * image_features @ text_features.t()).detach().cpu()
            )
        else:
            logits_per_image = (
                (logit_scale * image_features @ text_features.t()).detach().cpu()
            )

        # Compute logits for text-to-image retrieval
        logits_per_text = logits_per_image.t().detach().cpu()

        logits = {
            "image_to_text": logits_per_image,  # shape: (N, N) => images x texts
            "text_to_image": logits_per_text,  # shape: (N, N) => texts x images
        }

        # Ground-truth indices for the full set
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

        # Compute "full set" ranks & metrics (as before)
        metrics = {}
        per_specimen_ranks = {"image_to_text": {}, "text_to_image": {}}

        for name, logit in logits.items():
            # Sort the logits in descending order to get rankings
            ranking = torch.argsort(logit, descending=True)
            # Find the rank positions of the ground truth
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()

            # Compute Mean Rank and Median Rank (1-based)
            metrics[f"{name}_mean_rank"] = float(preds.mean() + 1)
            metrics[f"{name}_median_rank"] = float(np.floor(np.median(preds)) + 1)

            # Compute Recall@K
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = float(np.mean(preds < k))

            # Map ranks to specimen IDs (for final storage)
            for idx, specimen_id in enumerate(all_specimen_ids):
                per_specimen_ranks[name][specimen_id] = int(
                    preds[idx] + 1
                )  # 1-based rank

        # If bootstrapping is requested, compute confidence intervals
        if bootstraps > 1:
            ci_results = self._bootstrap_confidence_intervals(
                logits, bootstraps=bootstraps
            )
            # Add CI results to metrics
            for metric_name, ci in ci_results.items():
                metrics[f"{metric_name}_mean"] = ci["mean"]
                metrics[f"{metric_name}_ci_lower"] = ci["ci_lower"]
                metrics[f"{metric_name}_ci_upper"] = ci["ci_upper"]

        # Store the metrics and per-specimen ranks
        self._store_metrics(
            metrics, per_specimen_ranks, epoch, logits, all_specimen_ids
        )

        return metrics

    def _bootstrap_confidence_intervals(self, logits, bootstraps):
        """
        Computes bootstrap confidence intervals for retrieval metrics by resampling
        the *query* set only, while keeping the reference set intact.

        Args:
            logits (dict): A dictionary containing:
                           - logits["image_to_text"]: shape (N, N)
                           - logits["text_to_image"]: shape (N, N)
            bootstraps (int): Number of bootstrap iterations.

        Returns:
            dict: Dictionary with metric names as keys and dictionaries containing
                  mean, ci_lower, and ci_upper as values.
        """
        alpha = self.confidence_level
        lower_p = (1 - alpha) / 2 * 100
        upper_p = (1 + alpha) / 2 * 100
        N = logits["image_to_text"].shape[0]  # number of samples (images/text)

        ci_results = {}

        # We’ll define a helper to compute rank-based metrics for a set of ranks:
        def compute_rank_metrics(ranks_array):
            # ranks_array is 0-based rank
            mean_rank = np.mean(ranks_array) + 1.0  # 1-based
            median_rank = np.floor(np.median(ranks_array)) + 1.0
            r1 = np.mean(ranks_array < 1.0)
            r5 = np.mean(ranks_array < 5.0)
            r10 = np.mean(ranks_array < 10.0)
            return mean_rank, median_rank, r1, r5, r10

        # For each retrieval direction, do query-only bootstrapping
        # 'image_to_text' => images are queries, text is reference
        # 'text_to_image' => text is query, images are reference
        for name in ["image_to_text", "text_to_image"]:
            logits_2d = logits[name].numpy()  # shape (N, N), query x reference
            bootstrap_mean_rank = []
            bootstrap_median_rank = []
            bootstrap_R1 = []
            bootstrap_R5 = []
            bootstrap_R10 = []

            for _ in range(bootstraps):
                # Sample query indices (with replacement)
                sampled_indices = np.random.randint(0, N, size=(N,))

                # Compute ranks for each query in 'sampled_indices'
                ranks_resampled = []
                # For i-th query, the correct reference is the same index (i.e., i)
                # but since we are sampling with replacement, we need to find
                # the rank of 'sampled_indices[i]' in row i of the 2D logits.
                # Actually, if name == 'image_to_text', the correct text index is the same
                # as the image index. But here, we have "row = image_i, col = text_i".

                # So for each bootstrapped query i, the "real" row is 'sampled_indices[i]',
                # and the correct reference is 'sampled_indices[i]' as well.

                # 1) Get that row's logits
                # 2) Sort descending
                # 3) Find where 'sampled_indices[i]' sits
                row_logits = logits_2d[sampled_indices]  # shape (N, N)
                # row_logits[i] is the logits for the query "sampled_indices[i]"
                # against all references. We find the rank of reference = sampled_indices[i] in that row.

                # Sort indices in descending order for each row
                ranking = np.argsort(-row_logits, axis=1)  # shape (N, N) (descending)

                for i in range(N):
                    correct_ref = sampled_indices[i]
                    row_i = ranking[
                        i
                    ]  # the sorted indices for the i-th query in the resampled set
                    # where is correct_ref in row_i?
                    rank_pos = np.where(row_i == correct_ref)[0][0]  # 0-based rank
                    ranks_resampled.append(rank_pos)

                ranks_resampled = np.array(ranks_resampled)

                # Compute the metrics
                m_rank, md_rank, R1_, R5_, R10_ = compute_rank_metrics(ranks_resampled)
                bootstrap_mean_rank.append(m_rank)
                bootstrap_median_rank.append(md_rank)
                bootstrap_R1.append(R1_)
                bootstrap_R5.append(R5_)
                bootstrap_R10.append(R10_)

            # Convert bootstrap samples to np.array
            bootstrap_mean_rank = np.array(bootstrap_mean_rank)
            bootstrap_median_rank = np.array(bootstrap_median_rank)
            bootstrap_R1 = np.array(bootstrap_R1)
            bootstrap_R5 = np.array(bootstrap_R5)
            bootstrap_R10 = np.array(bootstrap_R10)

            # Store mean & CI bounds
            ci_results[f"{name}_mean_rank"] = {
                "mean": float(np.mean(bootstrap_mean_rank)),
                "ci_lower": float(np.percentile(bootstrap_mean_rank, lower_p)),
                "ci_upper": float(np.percentile(bootstrap_mean_rank, upper_p)),
            }
            ci_results[f"{name}_median_rank"] = {
                "mean": float(np.mean(bootstrap_median_rank)),
                "ci_lower": float(np.percentile(bootstrap_median_rank, lower_p)),
                "ci_upper": float(np.percentile(bootstrap_median_rank, upper_p)),
            }
            ci_results[f"{name}_R@1"] = {
                "mean": float(np.mean(bootstrap_R1)),
                "ci_lower": float(np.percentile(bootstrap_R1, lower_p)),
                "ci_upper": float(np.percentile(bootstrap_R1, upper_p)),
            }
            ci_results[f"{name}_R@5"] = {
                "mean": float(np.mean(bootstrap_R5)),
                "ci_lower": float(np.percentile(bootstrap_R5, lower_p)),
                "ci_upper": float(np.percentile(bootstrap_R5, upper_p)),
            }
            ci_results[f"{name}_R@10"] = {
                "mean": float(np.mean(bootstrap_R10)),
                "ci_lower": float(np.percentile(bootstrap_R10, lower_p)),
                "ci_upper": float(np.percentile(bootstrap_R10, upper_p)),
            }

        return ci_results

    def _store_metrics(
        self, metrics, per_specimen_ranks, epoch, logits, all_specimen_ids
    ):
        """
        Stores the computed metrics and per-specimen ranks to a JSON file.

        Args:
            metrics (dict): Dictionary containing the aggregated retrieval metrics and confidence intervals.
            per_specimen_ranks (dict): Dictionary mapping specimen IDs to their ranks for 'image_to_text' and 'text_to_image'.
            epoch (int): Current epoch number for logging purposes.
        """
        # Prepare the data to be stored
        stored_data = {
            "epoch": epoch,
            "metrics": metrics,
            "per_specimen_ranks": per_specimen_ranks,
        }

        # Define the output file path
        output_file = os.path.join(
            self.log_dir, f"retrieval_metrics_epoch_{epoch}.json"
        )

        # Write the data to the JSON file with indentation for readability
        with open(output_file, "w") as f:
            json.dump(stored_data, f, indent=4)

        # store logits as well
        logits_file = os.path.join(self.log_dir, f"retrieval_logits_epoch_{epoch}.pt")
        torch.save(logits, logits_file)

        # store all specimen IDs, which is a list of strings
        specimen_ids_file = os.path.join(
            self.log_dir, f"specimen_ids_epoch_{epoch}.json"
        )
        with open(specimen_ids_file, "w") as f:
            json.dump(all_specimen_ids, f, indent=4)

        logging.info(f"Retrieval metrics successfully stored in {output_file}")
