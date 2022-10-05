'''
Finetune imdb-dataset
conda install -c conda-forge datasets -y

'''

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TFAutoModel
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os, time
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import datasets

########################################################################

CACHE_DIR = r'C:\ai\datasets\huggingface'

#   ToDo look at masking

LR = 9.9e-5           # default 5e-5
BATCH_SIZE = 16
WARMUP_RATIO = 0.25
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 1
WEIGHT_DECAY = 0.01

########################################################################

# for computing the metrics during training ToDo think of other useful metrics
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

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def info(dataset_hugging, print_rows = 0):
    print('Type : ' + str(type(dataset_hugging)))
    #print(dataset_hugging)
    try:
        print('shape : ' + str(dataset_hugging.shape))
        if print_rows > 0:
            print(dataset_hugging[0])
    except:
        pass

#   return the sentiment for one review
def get_sentiment(text: str):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    output = model(**inputs)[0].squeeze()
    return (output.argmax())

def sentiment(reviews):
    LABELS = ["negative", "positive"]
    pretrained = "lannelin/bert-imdb-1hidden"
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained)

    preds = []
    for review in reviews:
        label = get_sentiment(review)
        preds.append((review, LABELS[label]))
    return preds



print("Loading imdb dataset\n")
#imdb_train_base, imdb_test_base = load_dataset("imdb", cache_dir=CACHE_DIR, split=['train', 'test'])
imdb = load_dataset("imdb", cache_dir=CACHE_DIR)
imdb_shuffled = imdb.shuffle()

#imdb_train, imdb_val = imdb_shuffled['train'].train_test_split(train_size=0.9, shuffle=True)

print('Example:')
#print(imdb_shuffled['train']['text'][0])
#print(imdb_train['train']['label'][:50])
#print(imdb_val['train']['label'][:50])

SAMPLES = len(imdb_shuffled['train'])
DIR_OUTPUT = os.path.join(os.getcwd(), 'models', str(GRADIENT_ACCUMULATION_STEPS) + '_LR_' + str(LR) + '_SIZE_' + str(SAMPLES) + '_EPOCHS_' + str(EPOCHS) + '_' + str(time.time()))      # saved metrics and data_dir

print('imdb_train samples ' + str(len(imdb_shuffled['train'])))
print('imdb_test samples ' + str(len(imdb_shuffled['test'])))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Tokenize imdb")
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
#checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_imdb = imdb_shuffled.map(preprocess_function, batched=True)

#model = TFAutoModel.from_pretrained(checkpoint)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)  # do_lower_case=True)
#model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")              # ToDo understand the differences

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
    train_dataset=tokenized_imdb['train'],
    eval_dataset=tokenized_imdb['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


print("Finetune imdb")

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
metrics = trainer.evaluate(eval_dataset=tokenized_imdb['test'])
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

model.save_pretrained(r'models/' + 'imdb_' + str(SAMPLES) + '.pth')