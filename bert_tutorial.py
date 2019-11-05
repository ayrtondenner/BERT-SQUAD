from pytorch_transformers import BertTokenizer

from pytorch_transformers import BertConfig
from pytorch_transformers import BertModel
from torch.utils.data import Dataset
from pytorch_transformers import 
#from pytorch_transformers import BertDa

from torch import nn

PRETRAINED_MODEL = 'bert-base-multilingual-cased' #'bert-base-uncased'
MAX_LEN = 100 # max is 512 for BERT

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=PRETRAINED_MODEL)
input_text = "this is a piece of feedback"
tokenized_text = tokenizer.tokenize(input_text)
tokenizer.convert_tokens_to_ids(tokenized_text)

class text_dataset(Dataset):
    def __init__(self, X, y):
        print("Inicializando text_dataset")
        self.X = X
        self.y = y
        
    def __getitem__(self,index):

        print("__getitem__")
        
        tokenized = tokenizer.tokenize(self.X[index])
        
        if len(tokenized) > MAX_LEN : tokenized = tokenized[:MAX_LEN]
            
        ids = tokenizer.convert_tokens_to_ids(tokenized)
            
        ids = torch.tensor(ids + [0] * (MAX_LEN - len(ids)))
        
        labels = [torch.from_numpy(np.array(self.y[index]))]
      
        return ids, labels[0]
    
    def __len__(self):
        print("__len__")
        return len(self.X)

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        
class BertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        
num_labels = 11
model = BertForSequenceClassification(num_labels)
model = nn.DataParallel(model)