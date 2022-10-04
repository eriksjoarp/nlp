import json, os
from pathlib import Path
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
import torch

##########################################
BATCH_SIZE = 16
LR = 5e-5


##########################################


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


#   get the character position at which the answer ends in the passage
def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


#    convert our character start/end positions to token start/end positions
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})



data_dir = r'C:\ai\datasets\transformers\squad'
path_train = os.path.join(data_dir, r'train-v2.0.json')
path_dev = os.path.join(data_dir, r'dev-v2.0.json')

train_contexts, train_questions, train_answers = read_squad(path_train)
val_contexts, val_questions, val_answers = read_squad(path_dev)

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

#   tokenize
print('tokenize')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#   encodings
print('encode')
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

#   token_positions
print('token positions')
add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

#   datasets
print('create datasets')
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

#   load model
print('load model')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

#   dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
optim = AdamW(model.parameters(), lr=LR)

#   training
print('start training')
for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()