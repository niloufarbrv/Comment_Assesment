import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.append("..")
from src.utils.constants import BASE_PATH


def get_args():
    """
    This function get arguments from user.
    :return
    """
    parser = ArgumentParser()

    parser.add_argument("--data_dir", default=BASE_PATH / "data", type=str)
    parser.add_argument("--saved_model_dir", default=BASE_PATH / "assets", type=str)
    parser.add_argument("--random_seed", default=0)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--validation_batch_size", default=2, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num_epoch", default=20, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    args = parser.parse_args()

    return args

