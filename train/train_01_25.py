import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<unused0>')

Chatbot_Data = pd.read_csv('../Chatbot_data/ChatbotData.csv')
# Test 용으로 300개 데이터만 처리한다.
Chatbot_Data = Chatbot_Data[:300]
Chatbot_Data.head()

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = '<usr>'
        self.a_token = '<sys>'
        self.sent_token = '<unused1>'
        self.eos = '</s>'
        self.mask = '<unused0>'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "skt/kogpt2-base-v2", bos_token='<s>', eos_token='</s>',
            unk_token='<unk>', pad_token='<pad>', mask_token='<unused0>')

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token + sentiment)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
                     self.mask,
                 ] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return (token_ids, np.array(mask),
                labels_ids)

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

train_set = ChatbotDataset(Chatbot_Data, max_len=40)
train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch,)

# print("start")
# for batch_idx, samples in enumerate(train_dataloader):
#     token_ids, mask, label = samples
#     print("token_ids ====> ", token_ids)
#     print("mask =====> ", mask)
#     print("label =====> ", label)
# print("end")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 1
Sneg = -1e18


########################################### 학습 모델 로딩  ##############################
save_ckpt_path=('../checkpoint/test.pth')
save_step = 10

pre_epoch, pre_loss, train_step = 0, 0, 0

if os.path.isfile(save_ckpt_path):
    checkpoint = torch.load(save_ckpt_path, map_location=device)
    pre_epoch = checkpoint['epoch']
    pre_loss = checkpoint['loss']
    train_step = checkpoint['train_no']
    total_train_step =  checkpoint['total_train_step']

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}")  #, loss={pre_loss}\n")
########################################### 학습 모델 로딩  ##############################
count=0
print ("start")
for epoch in range(epoch):
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits      #Returns a new tensor with the logit of the elements of input
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        # 학습 끝
        optimizer.step()

        #################################### 중간 저장     ########################################
        if (count > 0 and count % save_step == 0):
            torch.save({
                'epoch': epoch,
                'train_no': count,
                'model_state_dict': model.state_dict(),
                'total_train_step': len(train_dataloader),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_ckpt_path)
        count += 1
torch.save({
    'epoch': epoch,
    'train_no': count,
    'model_state_dict': model.state_dict(),
    'total_train_step': len(train_dataloader),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, save_ckpt_path)

#######################################################################################
print ("end")
