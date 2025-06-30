import json
import logging
import os

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class CaptionEvaluator:
    _instance = None

    def __new__(cls, confidence_level=0.95):
        if cls._instance is None:
            cls._instance = super(CaptionEvaluator, cls).__new__(cls)
            # Initialize the metrics
            cls._instance.tokenizer = PTBTokenizer()
            cls._instance.bleu_scorer = Bleu(n=4)  # BLEU with up to 4-grams
            cls._instance.meteor_scorer = Meteor()
            cls._instance.rouge_scorer = Rouge()
            cls._instance.cider_scorer = Cider()
            cls._instance.should_skip_meteor = False
            cls._instance.confidence_level = confidence_level
        return cls._instance

    def _bootstrap_confidence_interval(
        self, metric_name, scorer, gts_tokenized, res_tokenized, bootstraps
    ):
        """
        Resamples the data and computes the metric for each bootstrap sample to estimate confidence intervals.

        Args:
            metric_name (str): Name of the metric (e.g., 'bleu', 'rouge').
            scorer: The scorer object (e.g., self.bleu_scorer).
            gts_tokenized (dict): Tokenized ground truth captions.
            res_tokenized (dict): Tokenized generated captions.
            bootstraps (int): Number of bootstrap iterations.

        Returns:
            tuple: (mean_val, lower_bound, upper_bound) of the metric.
        """
        N = len(gts_tokenized)
        alpha = self.confidence_level
        lower_p = (1 - alpha) / 2 * 100
        upper_p = (1 + alpha) / 2 * 100

        bootstrap_metrics = []

        for _ in range(bootstraps):
            # Resample indices with replacement
            indices = np.random.randint(0, N, N)
            # Create new dictionaries for resampled data
            gts_resampled = {}
            res_resampled = {}
            for i, idx in enumerate(indices):
                # To ensure unique keys in the resampled dictionaries, append an index
                # This is necessary because scorers expect unique keys
                new_key = f"{idx}_{i}"
                gts_resampled[new_key] = gts_tokenized[idx]
                res_resampled[new_key] = res_tokenized[idx]

            # Compute the metric on the resampled data
            if metric_name == "bleu":
                # BLEU scorer returns (bleu_scores, bleu_scores_per_ngram)
                score, _ = scorer.compute_score(gts_resampled, res_resampled)
                bleu4_score = score[3]  # BLEU-4
                bootstrap_metrics.append(bleu4_score)
            else:
                # For ROUGE, CIDEr, METEOR, compute the mean score
                score, _ = scorer.compute_score(gts_resampled, res_resampled)
                bootstrap_metrics.append(score)

        bootstrap_metrics = np.array(bootstrap_metrics)
        mean_val = np.mean(bootstrap_metrics)
        lower_bound = np.percentile(bootstrap_metrics, lower_p)
        upper_bound = np.percentile(bootstrap_metrics, upper_p)

        return mean_val, lower_bound, upper_bound

    def _process_metric(
        self, metric_name, scorer, gts_tokenized, res_tokenized, bootstraps, **kwargs
    ):
        """
        Generic method to compute metric scores and confidence intervals by resampling data.

        Args:
            metric_name (str): Name of the metric (e.g., 'bleu', 'rouge').
            scorer: The scorer object (e.g., self.bleu_scorer).
            gts_tokenized (dict): Tokenized ground truth captions.
            res_tokenized (dict): Tokenized generated captions.
            bootstraps (int): Number of bootstrap iterations.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            tuple:
                - dict: Dictionary containing the metric's scores and confidence intervals.
                - np.array: The per-sample scores for the metric.
        """
        metrics = {}
        sample_scores = None  # Initialize sample_scores

        # Compute the metric without bootstrapping (original scores)
        if metric_name == "bleu":
            # BLEU scorer returns (bleu_scores, bleu_scores_per_ngram)
            score, sample_scores_per_ngram = scorer.compute_score(
                gts_tokenized, res_tokenized, **kwargs
            )
            bleu4_mean_score = score[3]  # BLEU-4
            metrics["bleu4_mean"] = bleu4_mean_score
            sample_scores = np.array(
                sample_scores_per_ngram[3]
            )  # BLEU-4 per-sample scores
        else:
            # For ROUGE, CIDEr, METEOR
            score, sample_scores = scorer.compute_score(
                gts_tokenized, res_tokenized, **kwargs
            )
            metrics[f"{metric_name}_mean"] = score
            sample_scores = np.array(sample_scores)

        # If bootstrapping is requested, compute confidence intervals
        if bootstraps > 1:
            mean_bs, lower_bs, upper_bs = self._bootstrap_confidence_interval(
                metric_name, scorer, gts_tokenized, res_tokenized, bootstraps
            )

            if metric_name == "bleu":
                metrics["bleu4_ci_lower"] = lower_bs
                metrics["bleu4_ci_upper"] = upper_bs
            else:
                metrics[f"{metric_name}_ci_lower"] = lower_bs
                metrics[f"{metric_name}_ci_upper"] = upper_bs

        return metrics, sample_scores

    def _store_captions(
        self, original_specimen_ids, captions, scores, metrics, log_dir, epoch
    ):
        """
        Stores predictions and overall metrics to a JSON file.

        Args:
            original_specimen_ids (list of str): List of original specimen IDs.
            captions (dict): Dictionary mapping integer IDs to captions with 'ground_truth' and 'generated'.
            scores (list): List of per-metric per-sample scores (BLEU, ROUGE, CIDEr, METEOR).
            metrics (dict): Dictionary of overall metrics and confidence intervals.
            log_dir (str): Directory to store the results.
            epoch (int): Current epoch number.
        """
        try:
            if not original_specimen_ids:
                raise ValueError("Original specimen IDs list is empty.")
            if not captions:
                raise ValueError("Captions dictionary is empty.")

            # Define the result path and ensure it exists
            result_path = os.path.join(log_dir, "caption_results")
            os.makedirs(result_path, exist_ok=True)

            # Initialize a dictionary to hold all results
            captions_with_scores = {"predictions": {}, "metrics": metrics}

            # Iterate over each specimen and assign true/predicted labels
            for i, sid in enumerate(original_specimen_ids):
                captions_with_scores["predictions"][sid] = {
                    "ground_truth": captions[i]["ground_truth"],
                    "generated": captions[i]["generated"],
                    "scores": {
                        "bleu": (
                            scores[0][i]
                            if len(scores) > 0 and scores[0] is not None
                            else None
                        ),
                        "rouge": (
                            scores[1][i]
                            if len(scores) > 1 and scores[1] is not None
                            else None
                        ),
                        "cider": (
                            scores[2][i]
                            if len(scores) > 2 and scores[2] is not None
                            else None
                        ),
                        "meteor": (
                            scores[3][i]
                            if len(scores) > 3 and scores[3] is not None
                            else None
                        ),
                    },
                }

            # Define the filename
            filename = f"captions_epoch_{epoch}.json"
            file_path = os.path.join(result_path, filename)

            # Write the results to a JSON file
            with open(file_path, "w") as f:
                json.dump(captions_with_scores, f, indent=4)

            logging.info(f"Stored captions and metrics to {file_path}")

        except (ValueError, IndexError, TypeError) as e:
            logging.error(f"Error storing captions and scores: {e}")
        except OSError as e:
            logging.error(f"Error writing to file: {e}")

    def compute_metrics(self, captions, log_dir, epoch, bootstraps=1):
        """
        Compute BLEU-4, ROUGE-L, CIDEr, and METEOR on the given captions dict.
        Optionally perform bootstrap resampling if bootstraps > 1.

        Args:
            captions (dict): Dictionary mapping image IDs to captions with 'ground_truth' and 'generated'.
            log_dir (str): Directory to store the results.
            epoch (int): Current epoch number.
            bootstraps (int): Number of bootstrap iterations. If >1, computes CIs.

        Returns:
            dict: Dictionary containing the computed metrics and their confidence intervals.
        """
        # Keep the original specimen_ids
        original_specimen_ids = list(captions.keys())

        # Re-map the img_ids from a string to an integer in order
        remapped_captions = {
            i: captions[key] for i, key in enumerate(original_specimen_ids)
        }

        # Extract ground truth and generated captions
        gts = {
            img_id: [{"caption": data["ground_truth"]}]
            for img_id, data in remapped_captions.items()
        }
        res = {
            img_id: [{"caption": data["generated"]}]
            for img_id, data in remapped_captions.items()
        }

        # Tokenize the captions
        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)

        metrics = {}

        # ---- Compute BLEU ----
        bleu_metrics, bleu_sample_scores = self._process_metric(
            metric_name="bleu",
            scorer=self.bleu_scorer,
            gts_tokenized=gts_tokenized,
            res_tokenized=res_tokenized,
            bootstraps=bootstraps,
        )
        metrics.update(bleu_metrics)

        # ---- Compute ROUGE-L ----
        rouge_metrics, rouge_sample_scores = self._process_metric(
            metric_name="rouge",
            scorer=self.rouge_scorer,
            gts_tokenized=gts_tokenized,
            res_tokenized=res_tokenized,
            bootstraps=bootstraps,
        )
        metrics.update(rouge_metrics)

        # ---- Compute CIDEr ----
        cider_metrics, cider_sample_scores = self._process_metric(
            metric_name="cider",
            scorer=self.cider_scorer,
            gts_tokenized=gts_tokenized,
            res_tokenized=res_tokenized,
            bootstraps=bootstraps,
        )
        metrics.update(cider_metrics)

        # ---- Compute METEOR ----
        if self.should_skip_meteor:
            metrics["meteor_mean"] = -1
            if bootstraps > 1:
                metrics["meteor_ci_lower"] = -1
                metrics["meteor_ci_upper"] = -1
        else:
            try:
                meteor_metrics, meteor_sample_scores = self._process_metric(
                    metric_name="meteor",
                    scorer=self.meteor_scorer,
                    gts_tokenized=gts_tokenized,
                    res_tokenized=res_tokenized,
                    bootstraps=bootstraps,
                )
                metrics.update(meteor_metrics)
            except Exception as e:
                metrics["meteor_mean"] = -1
                if bootstraps > 1:
                    metrics["meteor_ci_lower"] = -1
                    metrics["meteor_ci_upper"] = -1
                self.should_skip_meteor = True
                logging.warning(f"Error computing METEOR, skipping: {e}")

        # Collect per-sample scores for storage
        # Order: BLEU, ROUGE, CIDEr, METEOR
        scores = [
            bleu_sample_scores if "bleu4_mean" in metrics else None,
            rouge_sample_scores if "rouge_mean" in metrics else None,
            cider_sample_scores if "cider_mean" in metrics else None,
            meteor_sample_scores if "meteor_mean" in metrics else None,
        ]

        # Pass original_specimen_ids, remapped_captions, scores, and metrics to _store_captions
        self._store_captions(
            original_specimen_ids=original_specimen_ids,
            captions=remapped_captions,
            scores=scores,
            metrics=metrics,
            log_dir=log_dir,
            epoch=epoch,
        )

        return metrics
