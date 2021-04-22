from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForSequenceClassification, ElectraForSequenceClassification, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import pickle as pickle
import numpy as numpy
from load_data import *
from argument import get_args

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      if 'token_type_ids' in data.keys():
          outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              token_type_ids=data['token_type_ids'].to(device)
          )
      else:
          outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device)
          )
    # logits = outputs[0]
    # logits = logits.detach().cpu().numpy()
    # result = np.argmax(logits, axis=-1)
    # output_pred.append(result)

    logits = []
    predictions = []
    _logits = outputs[0].detach().cpu().numpy()      
    _predictions = np.argmax(_logits, axis=-1)
    logits.append(_logits)
    predictions.extend(_predictions.ravel())
  
  # return np.array(output_pred).flatten()
  return np.concatenate(logits), np.array(predictions)

def load_test_dataset(root, tokenizer):
  test_dataset = load_data(root+"/input/data/test/test.tsv", root)
  # test_dataset = load_data(root+"/input/data/test/ner_test_ver2.tsv", root)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = args.model_name
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  model_dir = f'./results/{args.id}/checkpoint-{args.checkpoint}'
  if args.model_type == 'bert':
    model = BertForSequenceClassification.from_pretrained(model_dir)
  elif args.model_type == 'electra':
    model = ElectraForSequenceClassification.from_pretrained(model_dir)
  elif args.model_type == 'roberta':
    model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
  model.parameters
  model.to(device)

  # load test datset
  # root = "/opt/ml"
  # root = "/content/drive/MyDrive/Boostcamp/Stage2_KLUE"
  root = args.root
  test_dataset, test_label = load_test_dataset(root, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  # pred_answer = inference(model, test_dataset, device)
  logits, predictions = inference(model, test_dataset, device)

  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  # output = pd.DataFrame(pred_answer, columns=['pred'])
  output = pd.DataFrame(predictions, columns=['pred'])
  output.to_csv(f'./results/{args.id}/submission{args.id}.csv', index=False)
  np.save(f'./results/{args.id}/logits{args.id}.npy', logits)
  print('File saved')

if __name__ == '__main__':
  args = get_args()
  main(args)
  
