import pandas as pd
import torch.nn as nn
from scipy.io import mmwrite, mmread
from sklearn.metrics.pairwise import linear_kernel
import pickle
from gensim.models import Word2Vec


class attraction_recommend_1():
    def __init__(self):
        self.keyword = ""
        self.df_reviews = pd.read_csv('../model/attraction/drop_cleaned_merged_reviews_trip_naver.csv')
        self.Tfidf_matrics = mmread('../model/attraction/Tfidf_attraction_review.mtx').tocsr()
        self.embedding_model = Word2Vec.load('../model/attraction/word2VecModel_attraction_reviews.model')
        with open('../model/attraction/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))  # 리스트로 묶여있으니 0이나 -1이나 같다.
        # ex) [(0, 0.18777633855487236), (1, 0.22096990858660473),  .... (917, 0.15855474152191032), (918, 0.14953229333043086)]

        # sorted 전에 content에 인덱스를 주자.
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)  # 내림차순 정리
        simScore = simScore[1:11]  # 0번은 자기 자신이니까 제외. 유사도 1이므로.
        content_idx = [i[0] for i in simScore]  # i[0]는 Content의 인덱스 // 유사한 10개 영화의 인덱스 받음.
        recContentList = self.df_reviews.iloc[content_idx]
        return recContentList

    def searchKeyword(self, sentence):
        self.key_word = sentence
        sentence = [self.key_word] * 11  # 토르가 10번 들어있는 리스트
        sim_word = self.embedding_model.wv.most_similar(self.key_word, topn=10)  # sim_word에 리스트로 받음.
        words = []
        for word, _ in sim_word:  # word는 단어 / _ 는 유사도
            words.append(word)

        for i, word in enumerate(words):
            sentence += [word] * (10 - i)  # 토르 10번 라그나로크 9번 갤럭시8번
        #
        sentence = ' '.join(sentence)
        #
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrics)
        recommendation = self.getRecommendation(cosine_sim)
        return recommendation['content']

