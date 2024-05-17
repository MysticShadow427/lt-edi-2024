import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))

def get_confusion_matrix(y_test, y_pred,class_names):
  cm = confusion_matrix(y_test, y_pred)
  print('\033[96m' + 'Confusion Matrix : \n'+'\033[0m',cm)
  df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
  hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Labels')
  plt.xlabel('Predicted Labels')
  plt.show()

def get_scores(y_test,y_pred):
    print('Accuracy : ',accuracy_score(y_test,y_pred))
    print()
    print('Precision : ',precision_score(y_test,y_pred,average='macro'))
    print()
    print('Recall : ',recall_score(y_test,y_pred,average='macro'))
    print()
    print('F-1 : ',f1_score(y_test,y_pred,average='macro'))

def get_peft_predictions(model, data_loader):
    model = model.eval()

    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs.logits, dim=1)

            probs = F.softmax(outputs.logits, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, prediction_probs, real_values