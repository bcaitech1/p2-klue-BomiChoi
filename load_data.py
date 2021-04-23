import pickle as pickle
import os
import pandas as pd
import torch
# from pororo import Pororo


# Dataset 구성.
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


def add_entity_tokens(sentence, a1, a2, b1, b2):
    new_sentence = None
    # ner = Pororo(task="ner", lang="ko")
    
    e1, e2 = sentence[a1:a2+1], sentence[b1:b2+1]
    # n1, n2 = Counter([e[1] for e in ner(e1)]).most_common(1)[0][0], Counter([e[1] for e in ner(e2)]).most_common(1)[0][0]
    # ner1, ner2 = "[T1]" + n1 + "[T1]", "[T2]" + n2 + "[T2]"
    n = 30
    if a1 > b1:
      # if b1 >= n:
      #   head = sentence[b1-n:b1]
      # else:
      #   head = sentence[:b1]
      # if len(sentence) >= a2+1+n:
      #   tail = sentence[a2+1:a2+1+n]
      # else:
      #   tail = sentence[a2+1:]
      head = sentence[:b1]
      tail = sentence[a2+1:]
      # new_sentence = sentence[:b1] + "[E2]" + ner2 + sentence[b1:b2+1] + "[E2]" + sentence[b2+1:a1] + "[E1]" + ner1 + sentence[a1:a2+1] + "[E1]" + sentence[a2+1:]
      new_sentence = head + "[E2]" + e2 + "[/E2]" + sentence[b2+1:a1] + "[E1]" + e1 + "[/E1]" + tail
    else:
      # if a1 >= n:
      #   head = sentence[a1-n:a1]
      # else:
      #   head = sentence[:a1]
      # if len(sentence) >= b2+1+n:
      #   tail = sentence[b2+1:b2+1+n]
      # else:
      #   tail = sentence[b2+1:]
      head = sentence[:a1]
      tail = sentence[b2+1:]
      # new_sentence = sentence[:a1] + "[E1]" + ner1 + sentence[a1:a2+1] + "[E1]" + sentence[a2+1:b1] + "[E2]" + ner2 + sentence[b1:b2+1] + "[E2]" + sentence[b2+1:]
      new_sentence = head + "[E1]" + e1 + "[/E1]" + sentence[a2+1:b1] + "[E2]" + e2 + "[/E2]" + tail
    return new_sentence


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 's1':dataset[3], 'e1':dataset[4], 'entity_02':dataset[5], 's2':dataset[6], 'e2':dataset[7], 'label':label,})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir, root):
  # load label_type, classes
  with open(root+'/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # dataset = pd.read_csv(dataset_dir, delimiter='\t')
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  new_sentences = []
  # tokenizer.add_special_tokens({'additional_special_tokens':["[E1]", "[/E1]", "[E2]", "[/E2]"]})
  # tokenizer.add_special_tokens({'additional_special_tokens':["@", "α", "#", "β"]})

  # for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
  for e01, e02, s1, e1, s2, e2, sen in zip(dataset['entity_01'], dataset['entity_02'], dataset['s1'], dataset['e1'], dataset['s2'], dataset['e2'], dataset['sentence']):
    temp1 = e01 + '[SEP]' + e02
    # temp1 = '[E1]' + e01 + '[/E1]' + '[E2]' + e02 + '[/E2]'
    # temp1 = e01 + ', ' + e02 +' 관계는?'
    temp2 = add_entity_tokens(sen, s1, e1, s2, e2)
    concat_entity.append(temp1)
    new_sentences.append(temp2)
  print(concat_entity[:10])
  print(new_sentences[:10])

  tokenized_sentences = tokenizer(
      concat_entity,
      # list(dataset['sentence']),
      new_sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=160,
      add_special_tokens=True,
      )
  return tokenized_sentences
