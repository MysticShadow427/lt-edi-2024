import torch.nn as nn
import numpy as np 
import torch 
from tqdm import tqdm
from augment_data import augment_embeddings

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples,
  num_views,
  i
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    modules = [*model.bert.encoder.layer[:i]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
    features,outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    aug_features = augment_embeddings(features,num_views)
    _, preds = torch.max(outputs, dim=1)
    # aug_features = aug_features/torch.sqrt(torch.tensor(768, dtype=torch.float32))
    norms = torch.norm(aug_features, p=2, dim=2, keepdim=True)
    aug_features= aug_features/ norms
    loss = loss_fn(aug_features,outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples,num_views):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      features,outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      aug_features = augment_embeddings(features,num_views)
      
    #   aug_features = aug_features/torch.sqrt(torch.tensor(768, dtype=torch.float32))
      norms = torch.norm(aug_features, p=2, dim=2, keepdim=True)
      aug_features= aug_features/ norms
      _, preds = torch.max(outputs, dim=1)
     
      loss = loss_fn(aug_features,outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)