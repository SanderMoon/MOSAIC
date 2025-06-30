# Training Script Arguments

This document outlines the command-line arguments that can be passed to the script for training deep learning models. These arguments control various aspects of the training process, including data loading, model selection, optimization, logging, and evaluation. The arguments are parsed using the `parse_args` function. Notably, if the optimizer parameters `--lr`, `--beta1`, `--beta2`, and `--eps` are not explicitly provided, their default values are determined by the `get_default_params` function based on the chosen model architecture (specified by `--model`).

## Data and Dataset Configuration

* **`--train-split`** (string, default: `None`)
    Path to the file with patient IDs or other identifiers used for the training dataset.

* **`--test-split`** (string, default: `None`)
    Path to the file with patient IDs or other identifiers used for the testing dataset.

* **`--val-split`** (string, default: `None`)
    Path to file(s) with data for the validation dataset.

* **`--workers`** (int, default: `4`)
    Number of dataloader worker processes per GPU.

* **`--batch-size`** (int, default: `64`)
    Batch size to be used per GPU.

* **`--aug-cfg`** (list of key-value pairs, default: `{}`)
    Augmentation configuration options. Expects a list of `key=value` pairs (e.g., `transform_type=random_flip probability=0.5`). Values are parsed using `ast.literal_eval` where possible, otherwise treated as strings.

* **`--text-data-file`** (string, default: `None`)
    Path to a file containing text data.

* **`--hdf5_filename`** (string, default: `None`)
    Path to an HDF5 file that contains all necessary data.

* **`--hdf5_feature_attribute`** (string, default: `None`)
    The attribute name for feature data within the HDF5 file specified by `--hdf5_filename`.

* **`--hdf5_text_attribute`** (string, default: `None`)
    The attribute name for text data within the HDF5 file specified by `--hdf5_filename`.

* **`--num-samples`** (int, default: `None`)
    Number of samples to use from the dataset. If not specified, all available samples will be used.

* **`--root-dir`** (string, default: `None`)
    Root directory for the dataset.

* **`--image-features-cutoff`** (int, default: `100000`)
    Maximum number of image features to use. If the number of features exceeds this value, random sampling will be applied to keep the sequence length manageable.

## Model Configuration

* **`--model`** (string, default: `"coca_stage_1"`)
    Name of the model architecture to use (e.g., vision backbone name). This choice also influences the default optimizer parameters if they are not set.

* **`--pretrained`** (string, default: `None`)
    Path to a file or a string identifier for pretrained model weights.

* **`--lock-image`** (bool, default: `False`)
    If set, freezes the entire image tower of the model by disabling gradient calculations for its parameters.

* **`--lock-image-unlocked-groups`** (int, default: `0`)
    Specifies the number of final layer groups in the image tower to leave unfrozen (trainable) when `--lock-image` is active.

* **`--lock-image-freeze-bn-stats`** (bool, default: `False`)
    If set, freezes the running statistics (mean and variance) of BatchNorm layers within the image tower for any layers that are locked.

* **`--lock-text`** (bool, default: `False`)
    If set, freezes the entire text tower of the model by disabling gradient calculations for its parameters.

* **`--lock-text-unlocked-layers`** (int, default: `0`)
    Specifies the number of final layers in the text tower to leave unfrozen (trainable) when `--lock-text` is active.

* **`--lock-text-freeze-layer-norm`** (bool, default: `False`)
    If set, freezes the running statistics of LayerNorm layers within the text tower for any layers that are locked.

* **`--use-bnb-linear`** (string, default: `None`)
    Replace the network linear layers with layers from the `bitsandbytes` library. Allows int8 training/inference, etc. Example values might be 'int8' or similar, depending on library usage.

* **`--torchscript`** (bool, default: `False`)
    If set, the model will be compiled using `torch.jit.script`. If `pretrained=='openai'`, the JIT version of OpenAI models will also be used.

* **`--torchcompile`** (bool, default: `False`)
    If set, the model will be compiled using `torch.compile()` (requires PyTorch 2.0 or later).

* **`--trace`** (bool, default: `False`)
    If set, the model will be traced using `torch.jit.trace`. This is typically used for inference or evaluation only.

* **`--grad-checkpointing`** (bool, default: `False`)
    If set, enables gradient checkpointing to save memory during training at the cost of a small amount of recomputation.

## Training Hyperparameters

* **`--epochs`** (int, default: `32`)
    Total number of epochs to train the model for.

* **`--epochs-cooldown`** (int, default: `None`)
    When using a learning rate scheduler with a cooldown phase, this specifies the number of epochs at the end of training during which cooldown occurs. Cooldown starts from `total_epochs - cooldown_epochs`.

* **`--lr`** (float, default: model-dependent)
    Learning rate. If not specified, the default is determined by `get_default_params(args.model)`.
    (e.g., `5.0e-4` for ViT models, `5.0e-4` for others by default in the script).

* **`--beta1`** (float, default: model-dependent)
    Adam optimizer's beta1 parameter. If not specified, the default is determined by `get_default_params(args.model)`.
    (e.g., `0.9` for ViT models, `0.9` for others by default in the script).

* **`--beta2`** (float, default: model-dependent)
    Adam optimizer's beta2 parameter. If not specified, the default is determined by `get_default_params(args.model)`.
    (e.g., `0.98` for ViT models, `0.999` for others by default in the script).

* **`--eps`** (float, default: model-dependent)
    Adam optimizer's epsilon parameter. If not specified, the default is determined by `get_default_params(args.model)`.
    (e.g., `1.0e-6` for ViT models, `1.0e-8` for others by default in the script).

* **`--wd`** (float, default: `0.2`)
    Weight decay parameter.

* **`--warmup`** (int, default: `100`)
    Number of initial training steps during which the learning rate is gradually increased (warmed up).

* **`--skip-scheduler`** (bool, default: `False`)
    If set, the learning rate scheduler (and decay) will be skipped, maintaining a constant learning rate after warmup (if any).

* **`--lr-scheduler`** (string, default: `'cosine'`)
    Learning rate scheduler type.
    Choices: `'cosine'`, `'const'` (constant LR), `'const-cooldown'` (constant LR with a cooldown phase).

* **`--lr-cooldown-end`** (float, default: `0.0`)
    The target learning rate at the end of the cooldown schedule.

* **`--lr-cooldown-power`** (float, default: `1.0`)
    The power for the polynomial cooldown schedule. A value of `1.0` corresponds to a linear decay during cooldown.

* **`--accum-freq`** (int, default: `1`)
    Gradient accumulation frequency. The model weights will be updated every `--accum-freq` steps (batches).

* **`--grad-clip-norm`** (float, default: `None`)
    If specified, gradients will be clipped to this maximum norm value.

* **`--precision`** (string, default: `"amp"`)
    Floating point precision for training.
    Choices: `"amp"`, `"amp_bf16"`, `"amp_bfloat16"`, `"bf16"`, `"fp16"`, `"pure_bf16"`, `"pure_fp16"`, `"fp32"`.

* **`--device`** (string, default: `'cuda'`)
    Device to use for training (e.g., `'cuda'`, `'cpu'`).

## Loss Configuration

* **`--local-loss`** (bool, default: `False`)
    If set, calculates loss using local features at a global level, instead of realizing the full global-to-global similarity matrix. This can save memory.

* **`--gather-with-grad`** (bool, default: `False`)
    If set, enables full distributed gradient for feature gathering operations.

* **`--coca-caption-loss-weight`** (float, default: `2.0`)
    Weight assigned to the captioning loss component when training CoCa-style models.

* **`--coca-contrastive-loss-weight`** (float, default: `1.0`)
    Weight assigned to the contrastive loss component when training CoCa-style models.

* **`--siglip`** (bool, default: `False`)
    If set, uses the SigLip (sigmoid) loss function instead of the default contrastive loss.

## Logging and Reporting

* **`--logs`** (string, default: `"./logs/"`)
    Directory where TensorBoard logs (and potentially other logs) will be stored. Use `None` (as a string: `"None"`) to disable storing logs.

* **`--log-local`** (bool, default: `False`)
    If set, log files are written on the local master node. Otherwise, logging might be restricted to the global master node in a distributed setup.

* **`--name`** (string, default: `None`)
    An optional identifier for the experiment when storing logs. If not provided, the current timestamp is typically used.

* **`--report-to`** (string, default: `''`)
    Comma-separated list of reporting destinations.
    Options include: `'wandb'`, `'tensorboard'`, or combined as `'wandb,tensorboard'`.

* **`--wandb-notes`** (string, default: `''`)
    Notes to associate with the run if logging with Weights & Biases (`wandb`).

* **`--wandb-project-name`** (string, default: `'open-clip'`)
    The project name to use when logging with Weights & Biases (`wandb`).

* **`--log-every-n-steps`** (int, default: `100`)
    Frequency (in training steps) for logging information to TensorBoard, console, and/or `wandb`.

* **`--log-memory`** (bool, default: `False`)
    If true, log memory usage during training.

* **`--debug`** (bool, default: `False`)
    If set, enables more verbose logging and debugging information.

* **`--copy-codebase`** (bool, default: `False`)
    If set, the entire codebase will be copied to the log directory, and the script will be executed from there. This helps in reproducibility.

## Checkpointing and Saving

* **`--save-frequency`** (int, default: `20`)
    Frequency (in epochs) for saving model checkpoints.

* **`--save-most-recent`** (bool, default: `False`)
    If set, always saves the model checkpoint from the most recent epoch to a file named `epoch_latest.pt` (or similar).

* **`--resume`** (string, default: `None`)
    Path to a checkpoint file (`.pt`) to resume training from.

* **`--remote-sync`** (string, default: `None`)
    An optional remote path (e.g., S3 bucket URI) to synchronize checkpoints and logs with.

* **`--remote-sync-frequency`** (int, default: `300`)
    Frequency (in seconds) for synchronizing files to the remote storage if `--remote-sync` is specified.

* **`--remote-sync-protocol`** (string, default: `"s3"`)
    Protocol to use for remote synchronization.
    Choices: `"s3"`, `"fsspec"`.

* **`--delete-previous-checkpoint`** (bool, default: `False`)
    If set, the previously saved checkpoint will be deleted after a new one is successfully stored.

## Distributed Training

* **`--dist-url`** (string, default: `"env://"`)
    URL used to set up distributed training. `"env://"` typically means settings are read from environment variables.

* **`--dist-backend`** (string, default: `"nccl"`)
    Distributed communication backend to use (e.g., `"nccl"`, `"gloo"`).

* **`--use-bn-sync`** (bool, default: `False`)
    If set, enables synchronized Batch Normalization across multiple GPUs/nodes. This is useful in distributed training.

* **`--horovod`** (bool, default: `False`)
    If set, uses Horovod for distributed training instead of PyTorch's native DDP.

* **`--ddp-static-graph`** (bool, default: `False`)
    If set, enables static graph optimization for DistributedDataParallel (DDP) in PyTorch versions 1.11 and later. Can improve performance.

* **`--no-set-device-rank`** (bool, default: `False`)
    If set, the script will not attempt to set the device index from the local rank. This is relevant when `CUDA_VISIBLE_DEVICES` is already configured to restrict visibility to one GPU per process.

## Evaluation and Miscellaneous

* **`--val-frequency`** (int, default: `1`)
    Frequency (in epochs) for running evaluation on the validation dataset.

* **`--save-captions`** (bool, default: `False`)
    If set, generated captions will be saved to disk, and caption evaluation will be performed.

* **`--caption-val-frequency`** (int, default: `1`)
    Frequency (in epochs) for running caption evaluation specifically.

* **`--caption-val-max-seq-len`** (int, default: `100`)
    Maximum sequence length for caption generation during validation.

* **`--val-gen-top-k`** (int, default: `50`)
    In validation generation, considers the `top-k` most likely tokens at each step.

* **`--val-gen-num-beams`** (int, default: `6`)
    Number of beams to use for beam search during validation generation.

* **`--val-gen-num-beam-groups`** (int, default: `3`)
    Number of beam groups to use for diverse beam search during validation generation.

* **`--zsc-class-prompt-mapping`** (list of strings, default: `None`)
    One or more filenames representing zero-shot classification prompt-to-class mappings.

* **`--zsc-specimen-class-mapping`** (list of strings, default: `None`)
    One or more filenames representing zero-shot classification specimen-to-class mappings.

* **`--eval-metric-bootstraps`** (int, default: `1`)
    Number of bootstrap samples to use for calculating confidence intervals for zero-shot classification metrics.

* **`--eval-metric-ci`** (float, default: `0.95`)
    Confidence interval for zero-shot classification metrics (e.g., `0.95` for a 95% CI, covering the 0.025 to 0.975 quantiles).

* **`--eval-grace-period`** (int, default: `0`)
    Number of initial epochs to wait before starting any evaluation.

* **`--test`** (bool, default: `False`)
    If set, the training phase will be skipped entirely. Useful for running only evaluation or testing.

* **`--few-shot-examples`** (list of strings, default: `None`)
    One or more filenames representing few-shot classification examples for each class.

* **`--seed`** (int, default: `0`)
    Default random seed for reproducibility.
