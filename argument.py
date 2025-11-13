import argparse
import os
import math

parser = argparse.ArgumentParser("RDED")

# ==== SYNTHESIS PHASE ====
parser.add_argument("--arch-name", type=str, default="resnet18",
                    help="Teacher architecture name")
parser.add_argument("--subset", type=str, default="cifar10",
                    help="Dataset name (e.g., cifar10)")
parser.add_argument("--train-dir", type=str, default="./cifar10/train/",
                    help="Path to training dataset")
parser.add_argument("--nclass", type=int, default=10)
parser.add_argument("--mipc", type=int, default=600,
                    help="Number of pre-loaded images per class")
parser.add_argument("--ipc", type=int, default=1,
                    help="Images per class for synthesis")
parser.add_argument("--num-crop", type=int, default=1)
parser.add_argument("--input-size", default=32, type=int)
parser.add_argument("--factor", default=2, type=int)

# ==== RETRAIN / VALIDATION PHASE ====
parser.add_argument("--re-batch-size", default=0, type=int)
parser.add_argument("--re-accum-steps", type=int, default=1)
parser.add_argument("--mix-type", default="cutmix",
                    choices=["mixup", "cutmix", None])
parser.add_argument("--stud-name", type=str, default="resnet18")
parser.add_argument("--val-ipc", type=int, default=100)
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--temperature", type=float, default=20)
parser.add_argument("--val-dir", type=str, default="./data/cifar10/val/")
parser.add_argument("--re-epochs", default=50, type=int)
parser.add_argument("--syn-data-path", type=str, default="syn_data")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--adamw-lr", type=float, default=0.001)
parser.add_argument("--exp-name", type=str, default=None)

args = parser.parse_args()
# === Handle experiment name and folder ===
args.exp_name = f"{args.subset}_{args.arch_name}_ipc{args.ipc}"
if not os.path.exists(f"./exp/{args.exp_name}"):
    os.makedirs(f"./exp/{args.exp_name}")
args.syn_data_path = os.path.join(f"./exp/{args.exp_name}", args.syn_data_path)

# === Import and Run the main synthesis & validation ===
from synthesize.main import main as synth_main
from validation.main import main as valid_main
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    synth_main(args)
    valid_main(args)
