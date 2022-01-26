import os
import pandas as pd
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

from model.attraction import attraction_recommend
# from model.attraction.attraction_recommend import attraction_recommend
############################## 모델 프리딕트 ##############################

save_ckpt_path=('../checkpoint/test.pth')
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<unused0>')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
##########################################################################################
stopwords = ['씨발','18놈','꺼져']
sent = '0'
with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        ####################### 비속어 검출  #############################
        if q in stopwords :
            print('이쁜 말 써주세요.')
            continue
        ####################### 여행지 추천  ##################################
        if '여행지 추천' in q :
            print('어떤 여행지를 원하시나요?')
            keyword = input('키워드를 입력해주세요 >').strip()
            recommendation = attraction_recommend.attraction_recommend_1()
            recommendation = recommendation.searchKeyword(keyword)
            print(recommendation)
            continue
        #####################################################################
        a = ""
        while 1:
            input_ids = torch.LongTensor(tokenizer.encode("<usr>" + q + '<unused1>' + sent + "<sys>" + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == '</s>':
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))