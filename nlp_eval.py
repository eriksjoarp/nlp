import os
from transformers import BertConfig, BertModel

# Find files in subdirs with correct extension
def files_in_dirs(base_dir, extension='.txt'):
    return_files = []
    for path, current_directory, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                return_files.append(os.path.join(path, file))
    return return_files


CURRENT_MODEL = 'imdb_1000.pth'
CONFIG_NAME = 'config.json'
MODEL_NAME = 'pytorch_model.bin'

dir_models = os.path.join(os.getcwd(), 'models')
dir_current_model = os.path.join(dir_models, CURRENT_MODEL)
path_config = os.path.join(dir_current_model, CONFIG_NAME)
path_model = os.path.join(dir_current_model, MODEL_NAME)

# Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
config = BertConfig.from_json_file(path_config)

model = BertModel.from_pretrained(path_model, config=config)

print(model.eval())

dir_positive = r'C:\ai\datasets\transformers\aclImdb\train\pos'

files = files_in_dirs(dir_positive)

for i in range(5):
    with open(files[i]) as f:
        lines = f.readlines()
    print(lines[0])
print('\n')

model




