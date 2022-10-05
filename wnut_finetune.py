from pathlib import Path
import os, time
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import DistilBertForTokenClassification
from pathlib import Path
import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

#######################################
'''
conda create -n nlp python=3.7
conda activate
conda install transformers pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c anaconda scikit-learn 

'''
#######################################

PATH_WNUT17 = r'C:\ai\datasets\transformers\wnut17\wnut17train.conll'
CACHE_DIR = r'C:\ai\datasets\huggingface'

#   ToDo look at masking

LR = 5e-5           # default 5e-5
BATCH_SIZE = 16
WARMUP_RATIO = 0.25
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 1
WEIGHT_DECAY = 0.01

########################################################################


#######################################


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# for computing the metrics during training ToDo think of other useful metrics
def compute_metrics(pred):
    labels = pred.label_ids
    rounded_labels = np.argmax(labels, axis=1)
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(rounded_labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

#   wnut17 project
def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

DIR_OUTPUT_BASE = 'PATH_WNUT17_GRAD_ACC_STEPS_' +  str(GRADIENT_ACCUMULATION_STEPS) + '_LR_' + str(LR) + '_EPOCHS_' + str(EPOCHS) + '_' + str(time.time())
DIR_OUTPUT = os.path.join(os.getcwd(), 'models', DIR_OUTPUT_BASE)      # saved metrics and data_dir

#   define tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#   read texts
texts, tags = read_wnut(PATH_WNUT17)

print(texts[0][10:17], tags[0][10:17], sep='\n')

#   create val
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

#   create tokenizer and encode the texts
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

#   create datasets
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

#   create model
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

training_args = TrainingArguments(
    output_dir=DIR_OUTPUT,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE, #16
    per_device_eval_batch_size=BATCH_SIZE * 4,  #16
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=10,
    evaluation_strategy='epoch',                #
    save_strategy='epoch',                      # Needed to save best model in the end and to continue training from a checkpoint
    bf16=True,
    load_best_model_at_end=True,
    #lr_scheduler_type=lr_scheduler_type,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    #lr_scheduler_type=
    do_predict=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    #data_collator=data_collator,
    compute_metrics=compute_metrics
)


print("Finetune wnut")

print(model.dtype)
print(model.device)

print('Start training')
train_results = trainer.train()

model.eval()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

#metrics = trainer.evaluate(eval_dataset=tokenized_imdb_test)
metrics = trainer.evaluate(val_dataset)
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

model.save_pretrained(r'models/' + 'wnut17_' + str(time.time()) + '.pth')
