import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import torch 
import joblib
from load_llm import SpanClassifier

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{path}': {error}")


def plot_accuracy_loss(history):

    for key in history:
        history[key] = [torch.tensor(value) if not isinstance(value, torch.Tensor) else value for value in history[key]]

# Extracting data from history
    train_acc = [acc.cpu().numpy() for acc in history['train_acc']]
    train_loss = [loss.cpu().numpy() for loss in history['train_loss']]
    val_acc = [acc.cpu().numpy() for acc in history['val_acc']]
    val_loss = [loss.cpu().numpy() for loss in history['val_loss']]



    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy', marker='o')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def save_training_history(history,path):
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch in range(len(history['train_acc'])):
            writer.writerow({'epoch': epoch,
                            'train_acc': history['train_acc'][epoch],
                            'train_loss': history['train_loss'][epoch],
                            'val_acc': history['val_acc'][epoch],
                            'val_loss': history['val_loss'][epoch]})

    print("History saved to", path)


