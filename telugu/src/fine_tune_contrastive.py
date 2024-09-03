import os
import sys
from collections import defaultdict
import argparse
from model_trainer import train_epoch,eval_model
from dataloaders import create_data_loader
from load_llm import SpanClassifier,LorafiedSpanClassifier
from custom_losses import SupConWithCrossEntropy,SupervisedContrastiveLoss
from utils import plot_accuracy_loss,save_training_history,create_directory
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup,AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluate_model import get_confusion_matrix,get_predictions,get_scores,get_classification_report,visualize_pca_embeddings
from augment_data import random_undersample
from load_llm import SpanClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="number of epochs for training")
    parser.add_argument("--learning_rate",type=float,help="learning rate")
    parser.add_argument("--batch_size",type=int,help="batch size forr training")
    parser.add_argument("--num_views",type=int,help="number of views for the contrastive loss")
    parser.add_argument("--lora",type=int,help="add adapter layers or not")

    args = parser.parse_args()

    EPOCHS = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_views = args.num_views
    lora = args.lora

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\033[96m' + 'Device : ',device + '\033[0m')
    print()

    train_df = pd.read_csv('/content/lt-edi-2024/telugu/data/homo_Telugu_train.csv')
    val_df = pd.read_csv('/content/lt-edi-2024/telugu/data/homo_Telugu_dev.csv')
    test_df = pd.read_csv('/content/lt-edi-2024/telugu/data/homo_Telugu_test.csv')
    # train_df = pd.concat([train_df, val_df])
    train_df.rename(columns={'Text': 'content','Category':'label'}, inplace=True)
    val_df.rename(columns={'Text': 'content','Category':'label'}, inplace=True)
    test_df.rename(columns={'Text': 'content','Category':'label'}, inplace=True)
    print('\033[96m' + 'Loaded Training, validation and test dataframes'+ '\033[0m')
    print()
    # train_df = random_undersample(train_df)

    print('\033[96m' + 'Preprocessing of Data done'+ '\033[0m')
    print()
    directory_path = "/content/lt-edi-2024/telugu/artifacts"
    create_directory(directory_path)
    # Label to index
    tags = train_df.label.unique().tolist()
    num_classes = len(tags)
    class_to_index = {tag: i for i, tag in enumerate(tags)}
    # Encode labels
    train_df["label"] = train_df["label"].map(class_to_index)
    val_df["label"] = val_df["label"].map(class_to_index)
    test_df["label"] = test_df["label"].map(class_to_index)
    class_names = ['None of the categories','Homophobia','Transphobia']
    num_labels = 3

    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i,label in enumerate(class_names)}
    checkpoint = 'l3cube-pune/telugu-bert'

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('\033[96m' + 'Tokenizer Loaded'+ '\033[0m')
    print()
    model = None
    if not lora:
        model = SpanClassifier().to(device)
    else:
        model =  LorafiedSpanClassifier().to(device)
        
    print('\033[96m' + 'Model Telugu BERT Loaded'+ '\033[0m')
    print()

    train_data_loader = create_data_loader(train_df,tokenizer=tokenizer,max_len=35,batch_size=batch_size)
    val_data_loader = create_data_loader(val_df,tokenizer=tokenizer,max_len=35,batch_size=batch_size)
    test_data_loader = create_data_loader(test_df,tokenizer=tokenizer,max_len=35,batch_size=batch_size)
    print('\033[96m' + 'Dataloaders created')
    print()

    total_steps = len(train_data_loader) * EPOCHS

    loss_fn = SupConWithCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
                )
    print('\033[96m' + 'Loss function,Optimizer and Learning Rate Schedule set'+ '\033[0m')
    print()

    history = defaultdict(list)
    best_acc = 0
    print('\033[96m' + 'Starting training...'+ '\033[0m')
    print()

    for epoch in range(EPOCHS):
        print('\033[96m' + f'Epoch {epoch + 1}/{EPOCHS}'+ '\033[0m')
        print('\033[96m' + '-' * 10+ '\033[0m')

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_df),
            num_views,
            epoch
        )

        print('\033[96m' + f'Train loss {train_loss} accuracy {train_acc}'+ '\033[0m')

        val_acc, val_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(test_df),
            num_views
        )

        print('\033[96m' + f'Val   loss {val_loss} accuracy {val_acc}'+ '\033[0m')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_acc:
            torch.save(model.state_dict(), '/content/lt-edi-2024/telugu/artifacts/best_model_state_full_fine_tune.bin')
            torch.save(obj=model.state_dict(),f='/content/lt-edi-2024/telugu/artifacts/best_model_full_fine_tune.pth')
            # model.save_pretrained('/content/lt-edi-2024/marathi/artifacts/marbert',from_pt = True)
            best_acc = val_acc
    print()
    print('\033[96m' + 'Training finished'+ '\033[0m')
    print()
    # plot_accuracy_loss(history)
    # history_csv_file_path = "/content/lt-edi-202/marathi/artifacts/history.csv"
    # save_training_history(history=history,path=history_csv_file_path)
    # model = SpanClassifier()
    # model.load_state_dict(torch.load(f='/content/lt-edi-2024/marathi/artifacts/best_model_full_fine_tune.pth'))
    # model = model.to('cuda')
    print('\033[96m' + 'Training History saved'+ '\033[0m')
    print()

    print('\033[96m' + 'Getting Predictions...'+ '\033[0m')
    print()
    y_review_texts_test, y_pred_test, y_pred_probs_test, y_test, embeddings = get_predictions(model,test_data_loader)
    y_review_texts_val, y_pred_val, y_pred_probs_val, y_val, _ = get_predictions(model,val_data_loader)
    y_review_texts_train, y_pred_train, y_pred_probs_train, y_train, _ = get_predictions(model,train_data_loader)

    visualize_pca_embeddings(embeddings,y_test)

    print('Test Data Classification Report : ')
    print()
    get_classification_report(y_test,y_pred_test)
    get_scores(y_test,y_pred_test)
    get_confusion_matrix(y_test,y_pred_test,class_names)
    print('Val Data Classification Report : ')
    print()
    get_classification_report(y_val,y_pred_val)
    get_scores(y_val,y_pred_val)
    get_confusion_matrix(y_val,y_pred_val,class_names)
    print('\033[96m' + 'Train Data Classification Report : '+ '\033[0m')
    print()
    get_classification_report(y_train,y_pred_train)
    get_scores(y_train,y_pred_train)
    get_confusion_matrix(y_train,y_pred_train,class_names)
    


