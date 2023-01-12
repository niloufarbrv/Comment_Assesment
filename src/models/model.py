from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert_config.hidden_size, self.config.num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask=attention_mask)
        # this is the last line that convert the commnet's embedding to a their labels
        classifier_output = self.classifier(output[1])
        return classifier_output

