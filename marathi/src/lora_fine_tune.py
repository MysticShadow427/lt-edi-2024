import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from peft import AutoPeftModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate_model import get_confusion_matrix,get_classification_report,get_peft_predictions,get_scores
from augment_data import random_undersample
import pandas as pd
from preprocess_data import remove_newline_pattern,remove_numbers_and_urls,remove_emojis,remove_pattern
from dataloaders import get_trainer

peft_model_name = 'marathi-bert-lt-edi'
modified_base = 'marathi-bert-modified-lt-edi'
base_model = 'l3cube-pune/marathi-bert-v2'

train_df = pd.read_csv('/content/lt-edi-2024/marathi/data/Marathi_train.csv')
val_df = pd.read_csv('/content/lt-edi-2024/marathi/data/Marathi_dev.csv')
test_df = pd.read_csv('/content/lt-edi-2024/marathi/data/Marathi_test.csv')

train_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)
val_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)
test_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)

train_df = random_undersample(train_df)

train_df['content'] = train_df['content'].apply(remove_numbers_and_urls)
train_df['content'] = train_df['content'].apply(remove_newline_pattern)
train_df['content'] = train_df['content'].apply(remove_pattern)
train_df['content'] = train_df['content'].apply(remove_emojis)

val_df['content'] = val_df['content'].apply(remove_numbers_and_urls)
val_df['content'] = val_df['content'].apply(remove_newline_pattern)
val_df['content'] = val_df['content'].apply(remove_pattern)
val_df['content'] = val_df['content'].apply(remove_emojis)

test_df['content'] = test_df['content'].apply(remove_numbers_and_urls)
test_df['content'] = test_df['content'].apply(remove_newline_pattern)
test_df['content'] = test_df['content'].apply(remove_pattern)
test_df['content'] = test_df['content'].apply(remove_emojis)

train_df.to_csv('/content/lt-edi-2024/marathi/data/Marathi_train.csv',index = False)
val_df.to_csv('/content/lt-edi-2024/marathi/data/Marathi_dev.csv',index = False)
test_df.to_csv('/content/lt-edi-2024/marathi/data/Marathi_test.csv',index = False)
print('\033[96m' + 'Preprocessed CSV  ready'+ '\033[0m')

train_dataset = load_dataset("csv", data_files='/content/lt-edi-2024/marathi/data/Marathi_train.csv')
val_dataset = load_dataset("csv", data_files='/content/lt-edi-2024/marathi/data/Marathi_dev.csv')
test_dataset = load_dataset("csv", data_files='/content/lt-edi-2024/marathi/data/Marathi_test.csv')
tokenizer = AutoTokenizer.from_pretrained(base_model)

num_labels = 3
class_names = ['None of the categories','Homophobia','Transphobia']

id2label = {i: label for i, label in enumerate(class_names)}
label2id = {label: i for i,label in enumerate(class_names)}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def preprocess(examples):
    examples["label"] = [label2id[label] for label in examples["label"]]
    tokenized = tokenizer(examples['content'], truncation=True, padding=True)
    return tokenized

# def preprocess_test(examples):
#     # examples["label"] = [label2id[label] for label in examples["label"]]
#     tokenized = tokenizer(examples['content'], truncation=True, padding=True)
#     return tokenized

tokenized_dataset_train = train_dataset.map(preprocess, batched=True,  remove_columns=["content"])
tokenized_dataset_val = val_dataset.map(preprocess, batched=True,  remove_columns=["content"])
tokenized_dataset_test = test_dataset.map(preprocess, batched=True,  remove_columns=["content"])

train_dataset=tokenized_dataset_train['train']
val_dataset=tokenized_dataset_val['train']
test_dataset = tokenized_dataset_test['train']
print('\033[96m' + 'Datasets ready'+ '\033[0m')
print()

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
)
print('\033[96m' + 'Training arguments set.'+ '\033[0m')
print()


model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label=id2label)
print('\033[96m' + 'Loaded pretrained Marathi BERT'+ '\033[0m')
print()

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,target_modules='all-linear')
peft_model = get_peft_model(model, peft_config)

print('PEFT Model')
peft_model.print_trainable_parameters()

peft_lora_finetuning_trainer = get_trainer(peft_model)
print('\033[96m' + 'Training the peft model...'+ '\033[0m')
print()
peft_lora_finetuning_trainer.train()
peft_lora_finetuning_trainer.evaluate()
print('\033[96m' + 'Saving fine tuned peft model...'+ '\033[0m')
print()
tokenizer.save_pretrained(modified_base)
peft_model.save_pretrained(peft_model_name)
print('\033[96m' + 'Saved the peft model'+ '\033[0m')
print()
inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_name, id2label=id2label).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(modified_base)
print('\033[96m' + 'Loaded Trained Model for inference'+ '\033[0m')
print()


train_data_loader = DataLoader(train_dataset, batch_size=16, collate_fn=data_collator)
val_data_loader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)
test_data_loader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

print('\033[96m' + 'Getting Predictions...'+ '\033[0m')
print()
y_pred_test, y_pred_probs_test, y_test = get_peft_predictions(model,test_data_loader)
y_pred_val, y_pred_probs_val, y_val = get_peft_predictions(model,test_data_loader)
y_pred_train, y_pred_probs_train, y_train = get_peft_predictions(inference_model,train_data_loader)

print('\033[96m' + 'Test Data Classification Report : '+ '\033[0m')
print()
get_classification_report(y_test,y_pred_test)
get_scores(y_test,y_pred_test)
get_confusion_matrix(y_test,y_pred_test,class_names)
print()

print('\033[96m' + 'Val Data Classification Report : '+ '\033[0m')
print()
get_classification_report(y_val,y_pred_val)
get_scores(y_val,y_pred_val)
get_confusion_matrix(y_val,y_pred_val,class_names)
print()

print('\033[96m' + 'Train Data Classification Report : '+ '\033[0m')
print()
get_classification_report(y_train,y_pred_train)
get_scores(y_train,y_pred_train)
get_confusion_matrix(y_train,y_pred_train,class_names)
# generate_submission_lora_track_3(inference_model,test_data_loader,label2id)