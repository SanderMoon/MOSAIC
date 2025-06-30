import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(
                    value
                )  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-split",
        type=str,
        default=None,
        help="Path to file with the patient IDs used for training",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help="Path to file with the patient IDs used for testing",
    )

    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown",
        type=int,
        default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=100, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end",
        type=float,
        default=0.0,
        help="End learning rate for cooldown schedule. Default: 0",
    )
    parser.add_argument(
        "--lr-cooldown-power",
        type=float,
        default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=20, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=[
            "amp",
            "amp_bf16",
            "amp_bfloat16",
            "bf16",
            "fp16",
            "pure_bf16",
            "pure_fp16",
            "fp32",
        ],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="coca_stage_1",
        help="Name of the vision backbone to use.",
    )

    parser.add_argument(
        "--lock-image",
        default=False,
        action="store_true",
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument("--aug-cfg", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action="store_true",
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default="",
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
    )
    parser.add_argument(
        "--wandb-notes", default="", type=str, help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="open-clip",
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action="store_true",
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action="store_true",
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa.",
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa.",
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one.",
    )

    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help="Replace the network linear layers from the bitsandbytes library. "
        "Allows int8 training/inference, etc.",
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help="Use SigLip (sigmoid) loss.",
    )
    parser.add_argument(
        "--save-captions",
        default=False,
        action="store_true",
        help="Save captions to disk and perform caption evaluation.",
    )
    parser.add_argument(
        "--text-data-file", default=None, help="Path to text data file."
    )

    parser.add_argument(
        "--hdf5_filename", default=None, help="Path to hdf5 file with all data."
    )
    parser.add_argument(
        "--hdf5_feature_attribute",
        default=None,
        help="Attribute name for feature data in HDF5.",
    )
    parser.add_argument(
        "--hdf5_text_attribute",
        default=None,
        help="Attribute name for text data in HDF5.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="String path to pretrained model weights. ",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to use from the dataset, if not using all.",
    )

    parser.add_argument(
        "--log-memory",
        default=False,
        action="store_true",
        help="If true, log the memory usage.",
    )

    parser.add_argument(
        "--caption-val-frequency",
        type=int,
        default=1,
        help="How often to run caption evaluation.",
    )

    parser.add_argument(
        "--caption-val-max-seq-len",
        type=int,
        default=100,
        help="Maximum sequence length for caption generation during evaluation.",
    )

    parser.add_argument(
        "--val-gen-top-k",
        type=int,
        default=50,
        help="Top k most likely tokens to consider for validation generation.",
    )

    parser.add_argument(
        "--val-gen-num-beams",
        type=int,
        default=6,
        help="Number of beams to use for beamsearch during validation generation.",
    )

    parser.add_argument(
        "--val-gen-num-beam-groups",
        type=int,
        default=3,
        help="Number of beam groups to use for beamsearch during validation generation.",
    )
    parser.add_argument(
        "--root-dir", type=str, default=None, help="Root directory for the dataset."
    )
    parser.add_argument(
        "--image-features-cutoff",
        type=int,
        default=100000,
        help="Max number of image features after which use random sampling to contain sequence length.",
    )
    parser.add_argument(
        "--zsc-class-prompt-mapping",
        type=str,
        nargs="+",
        help="Filenames representing zero-shot classification prompt-to-class mappings.",
        default=None,
    )
    parser.add_argument(
        "--zsc-specimen-class-mapping",
        type=str,
        nargs="+",
        help="Filenames representing zero-shot classification specimen-to-class mappings.",
        default=None,
    )

    parser.add_argument(
        "--eval-metric-bootstraps",
        type=int,
        default=1,
        help="Number of bootstraps for zero-shot classification.",
    )

    parser.add_argument(
        "--eval-metric-ci",
        type=float,
        default=0.95,
        help="Confidence interval for zero-shot classification, 0.95 is 0.025 to 0.975.",
    )

    parser.add_argument(
        "--eval-grace-period",
        type=int,
        default=0,
        help="Number of epochs to wait before starting evaluation.",
    )

    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="If true, training will be skipped!",
    )

    parser.add_argument(
        "--few-shot-examples",
        type=str,
        nargs="+",
        help="Filenames representing few-shot classification examples per class",
        default=None,
    )

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
