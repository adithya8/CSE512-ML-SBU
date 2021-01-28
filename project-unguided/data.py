
import numpy as np
import pandas as pd
import re
from collections import Counter
import functools

DATA_PATH_PREFIX = "/data/avirinchipur/SST/stanfordSentimentTreebank/"
DATA_TEXT_PATH = "datasetSentences.txt"
DATA_SPLIT_PATH = "datasetSplit.txt"
DATA_LABEL_PATH = "sentiment_labels.txt"

class dataReader:
    def __init__(self):
        texts = open(f"{DATA_PATH_PREFIX+DATA_TEXT_PATH}", encoding="latin1").readlines()[1:]
        texts = list(map(lambda t: t.strip().split("\t"), texts))

        splits = open(f"{DATA_PATH_PREFIX+DATA_SPLIT_PATH}").readlines()[1:]
        splits = list(map(lambda s: s.strip().split(","), splits))
        splits = np.array(splits, dtype=int)

        labels = open(f"{DATA_PATH_PREFIX+DATA_LABEL_PATH}").readlines()[1:]
        labels = list(map(lambda l: l.strip().split("|"), labels))
        labels_soft = np.array(labels, dtype=float)
        labels_hard = []
        for i, label in labels_soft:
            if label <= 0.4:
                labels_hard.append([i, 0])
            elif label > 0.6:
                labels_hard.append([i, 2])
            else:
                labels_hard.append([i, 1])

        labels_hard = np.array(labels_hard)
        
        self.train, self.eval, self.test = [], [], []
        for i, j in splits:
            temp = [self.__clean_str(texts[i-1][1]), labels_soft[i-1, 1], labels_hard[i-1, 1]]
            if j == 1:
                self.train.append(temp)
            elif j == 2:
                self.test.append(temp)
            else:
                self.eval.append(temp)
        
    def __clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        #https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/blob/23c72429de5fb3d2f3da87bf2efb5559c15b78f7/preprocess.py#L15
        """
        string = re.sub(r"\. \. \.", "\.", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    
    def collect_text_meta(self):

        wordlist = []
        self.sents_word_freq = []
        for record in self.train:
            temp_list = record[0].split(" ")
            self.sents_word_freq.append(Counter(temp_list))
            wordlist.extend(temp_list)
        self.word_freq = Counter(wordlist)
    
    def to_csv(self, path_prefix:str):
        texts = open(f"{DATA_PATH_PREFIX+DATA_TEXT_PATH}", encoding="latin1").readlines()[1:]
        texts = list(map(lambda t: t.strip().split("\t"), texts))


        splits = open(f"{DATA_PATH_PREFIX+DATA_SPLIT_PATH}").readlines()[1:]
        splits = list(map(lambda s: s.strip().split(","), splits))
        splits = np.array(splits, dtype=int)

        labels = open(f"{DATA_PATH_PREFIX+DATA_LABEL_PATH}").readlines()[1:]
        labels = list(map(lambda l: l.strip().split("|"), labels))
        labels_soft = np.array(labels, dtype=float)
        labels_hard = []
        for i, label in labels_soft:
            if 0 <= label <= 0.2:
                labels_hard.append([i, 0, 0])
            elif 0.2 < label <= 0.4:
                labels_hard.append([i, 0, 1])
            elif 0.4 < label <= 0.6:
                labels_hard.append([i, 1, 2])
            elif 0.6 < label <= 0.8:
                labels_hard.append([i, 2, 3])
            elif 0.8 < label <= 1.0:
                labels_hard.append([i, 2, 4])

        labels_hard = np.array(labels_hard, dtype=int)
        
        self.texts_df = pd.DataFrame(texts, columns=["message_id", "message"])
        splits_df = pd.DataFrame(splits, columns=["message_id", "train_test_eval"])
        labels_df = pd.DataFrame(labels_hard, columns=["message_id", "coarse_label", "fine_label"])
        self.outcomes_df = pd.merge(labels_df, splits_df, on=["message_id"])

        print (f"Text df shape: {self.texts_df.shape}")
        print (f"Splits df shape: {splits_df.shape}")
        print (f"Labels df shape: {labels_df.shape}")
        print (f"Outcomes df shape: {self.outcomes_df.shape}")

        self.texts_df.to_csv(f"{path_prefix}/text.csv", index=False)
        print (f"Texts dumped to {path_prefix}/text.csv")
        self.outcomes_df.to_csv(f"{path_prefix}/outcomes.csv", index=False)
        print (f"Outcomes dumped to {path_prefix}/outcomes.csv")


