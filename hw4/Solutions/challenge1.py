###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw4-Naive-bayes-challenge-1
###########################################
###########################################
#Imports
import pickle
import numpy as np
np.random.seed(42)
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

def pred_next_word(x:str, next_word_count:dict, top_k:int=1):

    #[(y, nxt_wc[x][y]),...] -> Sorted in the increasing order of nxt_wc[x][y]
    next_words = sorted(next_word_count[x].items(), key = lambda a: a[1])
    max_count = np.unique(np.array(next_words)[:, 1])[-top_k:].astype(int)
    
    #return np.array(next_words)[-top_k:, 0]
    return [i[0] for i in next_words if i[1] in max_count]

###########################################
###########################################

if __name__ == '__main__':

    count, next_word_count = load_meta()

    '''
    print (word_prior_proba("the", count))

    print (condn_proba_nxt_word("the", "rabbit", next_word_count, count))
    
    words = ["a", "the", "splendidly", "exclaimed"]
    #words = "its a bit of that they had never heard a the".split()
    for word in words:
        print (word, pred_next_word(word, next_word_count))
    '''

    np.random.seed(420)
    seed_word = np.random.choice(list(count.keys()), size = 1)[0]
    print (f"seed: {seed_word}")

    len_para = 200
    i = 0
    para = [seed_word]
    while i<=len_para:
        next_word = np.random.choice([pred_next_word(seed_word, next_word_count, 1)[0],], size=1)[0]
        para.append(next_word)
        seed_word = para[-1]
        i+=1

    print (f"para with highest proba word: {' '.join(para)}")

    np.random.seed(420)
    seed_word = np.random.choice(list(count.keys()), size = 1)[0]
    print (f"seed: {seed_word}")

    len_para = 200
    i = 0
    para = [seed_word]
    while i<=len_para:
        next_word = np.random.choice(pred_next_word(seed_word, next_word_count, 1), size=1)[0]
        para.append(next_word)
        seed_word = para[-1]
        i+=1

    print (f"para with highest proba word (random): {' '.join(para)}")

    np.random.seed(420)
    seed_word = np.random.choice(list(count.keys()), size = 1)[0]
    print (f"seed: {seed_word}")

    len_para = 200
    i = 0
    para = [seed_word]
    while i<=len_para:
        next_word = np.random.choice(pred_next_word(seed_word, next_word_count, 2), size=1)[0]
        para.append(next_word)
        seed_word = para[-1]
        i+=1

    print (f"para with 2 highest proba word: {' '.join(para)}")    

###########################################
###########################################
###########################################
###########################################
