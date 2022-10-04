from pathlib import Path
import os, time
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import transformers

#######################################
'''
conda create -n nlp python=3.7
conda activate
conda install transformers pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c anaconda scikit-learn 
conda install -c conda-forge nlp

EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks’ by Jason Wei, Kai Zou.
Synonym Replacement (SR), Random Insertion (RI), Random Swap (RS) and Random Deletion (RD)
https://towardsdatascience.com/augment-your-small-dataset-using-transformers-synonym-replacement-for-sentiment-analysis-part-1-87a838cd0baa

randomize words in sentence, in story
reverse sentence, one
exchange positive with negative, 1,2,3
remove 20% words
remove filler words

'''
#######################################
#   ToDo look at masking

SAMPLES = 75000
TRAIN_WITH_PYTORCH = False
LR_ = 5e-5       # default 5e-5
BATCH_SIZE = 16
WARMUP_RATIO = 0.25
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 5

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


def read_imdb_split(split_dir, max_files = -1):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        file_counter = 0
        for text_file in (split_dir/label_dir).iterdir():
            file_counter +=1
            if max_files > 0:
                if file_counter > max_files:
                    break
            texts.append(text_file.read_text(encoding='utf8'))
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(LR, GRADIENT_ACCUMULATION_STEPS):
    TIME_START = time.time()
    DIR_OUTPUT = os.path.join(os.getcwd(), 'models', str(GRADIENT_ACCUMULATION_STEPS) + '_LR_' + str(LR) + '_SIZE_' + str(SAMPLES))

    data_dir = r'C:\ai\datasets\transformers\aclImdb'
    train_dir = os.path.join(data_dir, r'train')
    test_dir = os.path.join(data_dir, r'test')

    print('Get texts from folders')
    train_texts, train_labels = read_imdb_split(train_dir, SAMPLES)
    test_texts, test_labels = read_imdb_split(test_dir, -1)

    #   validation dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    #   define tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    #   tokenize texts
    print('Tokenize')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    print('Create datasets')
    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    #lr_scheduler_type = transformers.get_cosine_schedule_with_warmup(torch.optim.AdamW(params), 125, 375)

    if not TRAIN_WITH_PYTORCH:
        #   Finetune with Trainer from transformers
        training_args = TrainingArguments(
            output_dir=DIR_OUTPUT,                      # output directory
            num_train_epochs=EPOCHS,                    # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,     # batch size per device during training
            per_device_eval_batch_size=64,              # batch size for evaluation
            #warmup_steps=200,                          # number of warmup steps for learning rate scheduler
            warmup_ratio=WARMUP_RATIO,
            weight_decay=0.01,                          # strength of weight decay
            logging_dir='./logs',                       # directory for storing logs
            logging_steps=10,
            learning_rate=LR,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            bf16=True,
            #fp16=True,
            load_best_model_at_end=True,
            #lr_scheduler_type=lr_scheduler_type,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            #lr_scheduler_type=
            do_predict=True
        )

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics
        )

        print('Start training')
        train_results = trainer.train()
        model.eval()

        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        metrics = trainer.evaluate()
        # some nice to haves:
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        model.save_pretrained(r'models/' + 'imdb_' + str(SAMPLES) + '.pth')


    if TRAIN_WITH_PYTORCH:
        # finetune with pytorch training
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.to(device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optim = AdamW(model.parameters(), lr=LR)

        for epoch in range(3):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()

        print(model.eval())
        model.save_pretrained(r'models/' + 'imdb_' + str(SAMPLES) + '.pth')

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            loss.backward()
            optim.step()



    #   ToDo try test set


    running_time = int(time.time()) - int(TIME_START)
    print('\nRunning time is ' + str(running_time) + ' seconds.')

    ##########################
    '''
    exit(0)
    #   how to prepare a dataset for imdb
    from nlp import load_dataset
    train = load_dataset("imdb", split="train")

    print(train.column_names)

    train = train.map(lambda batch: tokenizer(batch["text"], truncation=True, padding=True), batched=True)
    train.rename_column_("label", "labels")

    train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    #({key: val.shape for key, val in train[0].items()})
    '''

for LR in [2.5e-5]:
    train(LR, GRADIENT_ACCUMULATION_STEPS)
