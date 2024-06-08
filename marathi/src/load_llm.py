from transformers import AutoModel,AutoModelForSequenceClassification
from peft import LoraConfig,get_peft_model
import torch
import torch.nn as nn


class SpanClassifier(nn.Module):

  def __init__(self, n_classes=3,checkpoint = 'l3cube-pune/marathi-bert-v2'):
    super(SpanClassifier, self).__init__()
    self.checkpoint = checkpoint
    self.bert = AutoModel.from_pretrained(self.checkpoint,return_dict=False)
    # modules = [self.bert.embeddings, *self.bert.encoder.layer[:5]]
    # for module in modules:
    #     for param in module.parameters():
    #         param.requires_grad = False
    for param in self.bert.parameters():
        param.requires_grad = False
    self.drop = nn.Dropout(p=0.4)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return pooled_output,self.out(output)
  

class LorafiedSpanClassifier(nn.Module):
   def __init__(self,n_classes=3,checkpoint = 'l3cube-pune/marathi-bert-v2'):
      super().__init__()
      self.checkpoint = checkpoint
      self.bert = AutoModel.from_pretrained(self.checkpoint,return_dict=False)
      for param in self.bert.parameters():
        param.requires_grad = False
      self.peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=128, lora_dropout=0.4,target_modules='all-linear')
      self.peft_model = get_peft_model(self.bert, self.peft_config)
      self.drop = nn.Dropout(p=0.4)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
   
   def forward(self, input_ids, attention_mask):
    _, pooled_output = self.peft_model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return pooled_output,self.out(output)

  