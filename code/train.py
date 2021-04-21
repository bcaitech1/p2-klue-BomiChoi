import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertConfig, ElectraConfig, XLMRobertaConfig, BertForSequenceClassification, ElectraForSequenceClassification, XLMRobertaForSequenceClassification
from tokenization_kobert import KoBertTokenizer
from load_data import *
# from catalyst.data.sampler import BalanceClassSampler

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  def get_classes(self):
    return self.labels

def train(args):
  # load model and tokenizer
  MODEL_NAME = args.model_name
  if args.model_type == 'kobert':
    tokenizer = KoBertTokenizer.from_pretrained(MODEL_NAME)
  else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  # train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  train_dataset = load_data("/content/drive/MyDrive/Boostcamp/Stage2_KLUE/input/data/train/train.tsv")
  #dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset['label'].values
  #dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

   # BalanceClassSampler를 정의합니다. 여기선 upsampling 옵션을 주었습니다.
  # sampler = BalanceClassSampler(RE_train_dataset.get_classes(), 'upsampling')
  # RE_train_loader = DataLoader(RE_train_dataset, batch_size=16, sampler=sampler)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  if args.model_type == 'bert':
    bert_config = BertConfig.from_pretrained(MODEL_NAME)
    bert_config.num_labels = 42
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)
  elif args.model_type == 'electra':
    electra_config = ElectraConfig.from_pretrained(MODEL_NAME)
    electra_config.num_labels = 42
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=electra_config)
  elif args.model_type == 'roberta':
    roberta_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    roberta_config.num_labels = 42
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME,config=roberta_config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

  training_args = TrainingArguments(
    output_dir='./results/'+str(args.id),          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.num_train_epochs,              # total number of training epochs
    learning_rate=args.learning_rate,               # learning_rate
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs/'+str(args.id),            # directory for storing logs
    logging_steps=args.logging_steps,              # log saving step.
    #evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    #eval_steps = 500,            # evaluation step.
    # save_strategy='epoch'
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    #eval_dataset=RE_dev_dataset,             # evaluation dataset
    #compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

def main():
  train()

if __name__ == '__main__':
  main()
