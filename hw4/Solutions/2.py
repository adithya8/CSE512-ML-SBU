###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw4-Naive-bayes-2
###########################################
###########################################
#Imports
import pickle
import numpy as np
###########################################
###########################################

def load_meta(file_path:str='../alice_release/alice_parsed.pkl'):
    
    data = pickle.load(open(file_path,'rb'), encoding="utf-8")
    return data

###########################################
###########################################

def word_prior_proba(word:str, count:dict):

    if word not in count: return 0
    return count[word]/sum(list(count.values()))

def condn_proba_nxt_word(x:str, y:str, next_word_count:dict, count:dict):

    if y not in next_word_count: return 0
    dr = count[y]
    nr = next_word_count[x][y]
    return nr/dr

def pred_next_word(x:str, next_word_count:dict):

    next_words = sorted(next_word_count[x].items(), key = lambda a: a[1])
    max_count = next_words[-1][-1]

    return [i[0] for i in next_words if i[1] == max_count]

###########################################
###########################################

if __name__ == '__main__':

    count, next_word_count = load_meta()

    print (word_prior_proba("the", count))

    print (condn_proba_nxt_word("the", "rabbit", next_word_count, count))

    words = ["a", "the", "splendidly", "exclaimed"]
    for word in words:
        print (word, pred_next_word(word, next_word_count))
    
###########################################
###########################################
###########################################
###########################################
