from pathlib import Path
import sys
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

sys.path.append("..")


class Comment:

    def __init__(self, method_id: str, method_name: str, class_name: str):
        self.method_id = method_id
        self.method_name = method_name
        self.class_name = class_name
        self._comment_context = ""

    def __str__(self):
        return f"{self.__class__.__name__}(method_id={self.method_id}, method_name={self.method_name}," \
               f" class_name={self.class_name},\ncomment_context={self.comment_context})"

    @property
    def comment_context(self):
        return self._comment_context

    @comment_context.setter
    def comment_context(self, context: str):
        self._comment_context += context


def clean_comment(comment: str) -> str:
    comment_lines = comment.strip().split("\n")
    comment_edited = ""

    for line in comment_lines:
        # remove some keyword from comment
        if "@param" in line:
            continue
        if "@return" in line:
            continue
        comment_edited += line.strip().replace("*", "")
    return comment_edited.strip()


def read_data(path: Path):
    """
    read data from .txt file and convert it to a list of Comments
    """
    with open(path, encoding="utf8") as f:
        comments_list = []
        new_comment_flag = True
        comment_text = ""
        for i, line in enumerate(f):

            # check whether line is non-empty
            if line.split(","):
                if line.split(",")[0] == '###\n':
                    new_comment_flag = True
                    continue

            if new_comment_flag:
                if i != 0:
                    comment.comment_context = clean_comment(comment_text)
                    comments_list.append(comment)
                    comment_text = ""

                line_splits = line.split(",")
                comment = Comment(method_id=line_splits[0],
                                  method_name=line_splits[1],
                                  class_name=line_splits[2])
                new_comment_flag = False

                # the textual part of comments start with "   * "
            if line.startswith("   * "):
                comment_text += line

        return comments_list


def set_seed(seed=0):
    """
    set random seed from several libraries for reproducibility.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)


class CommentDataset(Dataset):
    def __init__(self, data_comment: list, data_label: pd.DataFrame, tokenizer=None):
        self.data_comment = data_comment
        self.texts = [tokenizer(sample.comment_context,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt")
                      for sample in tqdm(self.data_comment)]
        self.data_label = data_label
        # convert string labels to categorical
        self.data_label.labels = pd.Categorical(self.data_label.labels)
        self.data_label.labels = self.data_label.labels.cat.codes

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return self.texts[index], torch.tensor(self.data_label.values).to(torch.int64)[index, 1]

    def __len__(self):
        return len(self.data_comment)
