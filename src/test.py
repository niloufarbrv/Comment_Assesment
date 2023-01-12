import logging
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from transformers import BertConfig, BertModel, BertTokenizer

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


sys.path.append("..")
from src.utils.constants import BASE_PATH
from src.models.model import Model
from src.utils.preprocessing import CommentDataset, read_data
from src.configuration.config import get_args
from src.utils.preprocessing import set_seed

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(BASE_PATH / "documents/file.log")

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


args = get_args()
set_seed(args.random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

logger.info("Reading data and splitting it to train, validation, test")
data = read_data(path=args.data_dir / "Benchmark_Raw_Data.txt")
labels = pd.read_csv(args.data_dir / "Benchmark_Coherence_Data.csv", names=["id", "labels"])

train_size = int(0.8 * len(data))
dev_size = int(0.1 * len(data))

test_data = data[train_size + dev_size:]
test_label = labels.iloc[train_size + dev_size:]

logger.info(msg="Creating datasets")
test_dataset = CommentDataset(test_data, data_label=test_label, tokenizer=tokenizer)

logger.info(msg="Creating dataloaders")
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=args.validation_batch_size)
model = Model(config=args)
model.load_state_dict(torch.load(args.saved_model_dir / "saved_model.pt", map_location=device))
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = CrossEntropyLoss()
model.eval()

total_loss_test = 0
total_accuracy_test = 0
total_f1_test = 0

with torch.no_grad():
    # Looping trough test data
    for test_input, test_label in tqdm(test_dataloader, total=len(test_dataloader), desc="Training data batches"):

        test_label = test_label.to(device)
        mask = test_input["attention_mask"].to(device)
        input_id = test_input["input_ids"].squeeze(1).to(device)

        output = model(input_id, mask)

        output_target = torch.argmax(output, dim=1)
        batch_loss = criterion(output, test_label.long())
        total_accuracy_test += sum(output_target == test_label) / len(output_target)
        total_loss_test += batch_loss.item()
        total_f1_test += f1_score(test_label.cpu().detach(),
                                  output_target.cpu().detach(),
                                  average='macro')


total_loss_test /= len(test_dataloader)
total_accuracy_test /= len(test_dataloader)


logger.info(f"Test Loss: {torch.round(total_loss_test, decimals=3)}")
logger.info(f"Test Accuracy: {torch.round(total_accuracy_test, decimals=3)}")
logger.info(f"Test F1-macro: {torch.round(total_f1_test, decimals=3)}")
