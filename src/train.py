import logging
import sys
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score
from transformers import BertConfig, BertModel, BertTokenizer

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

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

writer = SummaryWriter(log_dir=BASE_PATH / "documents")
set_seed(0)
args = get_args()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

logger.info("Reading data and splitting it to train, test, validation")

data = read_data(path=args.data_dir / "Benchmark_Raw_Data.txt")
labels = pd.read_csv(args.data_dir / "Benchmark_Coherence_Data.csv", names=["id", "labels"])

train_size = int(0.8 * len(data))
dev_size = int(0.1 * len(data))

train_data, validation_data, test_data = data[:train_size], data[train_size: train_size + dev_size], data[
                                                                                                     train_size + dev_size:]
train_label, validation_label, test_label = labels.iloc[:train_size, :], labels.iloc[train_size: train_size + dev_size,
                                                                         :], labels.iloc[
                                                                             train_size + dev_size:]

logger.info(msg="Creating datasets")

train_dataset = CommentDataset(train_data, data_label=train_label, tokenizer=tokenizer)
validation_dataset = CommentDataset(validation_data, data_label=validation_label, tokenizer=tokenizer)

logger.info(msg="Creating dataloaders")
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size)
validation_dataloader = DataLoader(dataset=validation_dataset,
                                   batch_size=args.validation_batch_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = CrossEntropyLoss()
model = Model(config=args)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

for param in model.bert_model.parameters():
    param.requires_grad = False

optimizer = Adam(lr=0.01, params=model.parameters())
scheduler = ExponentialLR(optimizer, gamma=0.9)

best_validation_loss = None

logger.info("Starting training loop")
for epoch in range(args.num_epoch):
    total_loss_train = 0
    total_accuracy_train = 0
    total_f1_train = 0
    total_loss_validation = 0
    total_accuracy_validation = 0
    total_f1_validation = 0
    for train_input, train_label in tqdm(train_dataloader, total=len(train_dataloader), desc="Training data batches"):
        train_label = train_label.to(device)
        mask = train_input["attention_mask"].to(device)
        input_id = train_input["input_ids"].squeeze(1).to(device)
        model.zero_grad(set_to_none=True)
        output = model(input_id, mask)
        output_target = torch.argmax(output, dim=1)
        batch_loss = criterion(output, train_label.long())
        total_accuracy_train += sum(output_target == train_label) / len(output_target)
        total_f1_train += f1_score(train_label.cpu().detach(), output_target.cpu().detach(), average='macro')
        total_loss_train += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
    scheduler.step()

    total_accuracy_train /= len(train_dataloader)
    total_loss_train /= len(train_dataloader)
    total_f1_train /= len(train_dataloader)

    writer.add_scalar(f"Loss/train", total_loss_train, epoch)
    writer.add_scalar(f"Accuracy/train", total_accuracy_train, epoch)
    writer.add_scalar(f"f1/train", total_f1_train, epoch)

    logger.info(f"Epoch {epoch + 1} - Training Loss: {round(total_loss_train, 3)}")
    logger.info(f"Epoch {epoch + 1} - Training Accuracy: {total_accuracy_train}")
    logger.info(f"Epoch {epoch + 1} - Training F1-macro: {round(total_f1_train, 3)}")

    model.eval()
    with torch.no_grad():

        for validation_input, validation_label in tqdm(validation_dataloader, total=len(validation_dataloader),
                                                       desc="validation data batches"):
            validation_label = validation_label.to(device)
            mask = validation_input["attention_mask"].to(device)
            input_id = validation_input["input_ids"].squeeze(1).to(device)
            output = model(input_id, mask)
            output_target = torch.argmax(output, dim=1)
            batch_loss = criterion(output, validation_label.long())
            total_accuracy_validation += sum(output_target == validation_label) / len(output_target)
            total_loss_validation += batch_loss.item()
            total_f1_validation += f1_score(train_label.cpu().detach(), output_target.cpu().detach(), average='macro')

    total_accuracy_validation /= len(validation_dataloader)
    total_loss_validation /= len(validation_dataloader)
    total_f1_validation /= len(validation_dataloader)

    writer.add_scalar(f"Loss/validation", total_loss_validation, epoch)
    writer.add_scalar(f"Accuracy/validation", total_accuracy_validation, epoch)
    writer.add_scalar(f"Accuracy/validation", total_f1_validation, epoch)

    logger.info(f"Epoch {epoch + 1} - Validation Loss: {round(total_loss_validation, 3)}")
    logger.info(f"Epoch {epoch + 1} - Validation Accuracy: {total_accuracy_validation}")
    logger.info(f"Epoch {epoch + 1} - Validation f1-macro: {round(total_f1_validation, 3)}")

    # set best_validation_loss for the first epoch
    if best_validation_loss is None:
        best_validation_loss = total_loss_validation

    # checking if the directory assets exist or not.
    if not (BASE_PATH / "assets").isdir():
        # if the assets directory is not present then create it.
        p = (BASE_PATH / "assets")
        p.mkdir(parents=True, exist_ok=False)

    # save the best model
    if total_loss_validation < best_validation_loss:
        best_validation_loss = total_loss_validation
        torch.save(model.state_dict(), args.saved_model_dir / "saved_model.pt")
