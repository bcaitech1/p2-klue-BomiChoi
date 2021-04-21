from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForSequenceClassification, ElectraForSequenceClassification, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader
<<<<<<< HEAD
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
=======
import pandas as pd
import torch
import pickle as pickle
import numpy as numpy
from load_data import *
from argument import get_args
>>>>>>> b63d148b02f7db386f213180070535e4e675048c

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          # token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

<<<<<<< HEAD
def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
=======
def load_test_dataset(oot, tokenizer):
  test_dataset = load_data(root+"/input/data/test/test.tsv", root)
>>>>>>> b63d148b02f7db386f213180070535e4e675048c
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
  # root = '/opt/ml'
  root = args.root
  test_dataset, test_label = load_test_dataset(root, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer = inference(model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv(f'./results/{args.id}/submission{args.id}.csv', index=False)
  print('File saved')

if __name__ == '__main__':
  args = get_args()
  main(args)
  
