import numpy as np
from scipy.spatial.distance import jensenshannon

lang2file = {
    'en' : 'en-train',
    'fi' : 'fi-train',
    'ar' : 'ar-train',
    'bn' : 'bn-train',
    'id' : 'id-train',
    'ko' : 'ko-train',
    'ru' : 'ru-train',
    'sw' : 'sw-train',
    'te' : 'te-train',
}
path = "./download/panx/"

def read_data(path):  
    sentences = []
    tags = []
    answers = []
    sentence=[]
    sent_tag=[]

    with open(path, 'r') as f:
        for line in f:
            if line.strip()=="":
                sentences.append(" ".join(sentence))
                sentence=[]
                continue
            words = line.strip().split()
            sentence.append(words[0].split(":")[1])

    return sentences

def calculate_njsd(p, q):
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]
    # Calculate Jensen-Shannon Divergence
    jsd = jensenshannon(p, q)

    # Normalize the Jensen-Shannon Divergence
    njsd = jsd / np.sqrt(2)

    return njsd

def convert_ngram_prob(text,n):
    ngrams = [tuple(text[i:i+n]) for i in range(len(text)-n+1)]
    ngram_counts = {}
    for gram in ngrams:
        ngram_counts[gram] = ngram_counts.get(gram, 0) + 1
    prob = np.array(list(ngram_counts.values())) / sum(ngram_counts.values())
    return prob


def main():
    lexical_sim = {} # {(S,T), sim}
    lang2prob={}
    for lang in lang2file.keys():
        lang_path = path + lang2file[lang]
        text_corpus = read_data(lang_path)
        lang2prob[lang]=convert_ngram_prob(text_corpus,3)
    print("ngrams (3) calculated, computing Normalised Jensen-Shannon Divergence")
    for S in lang2file.keys():
        for T in lang2file.keys():
            if(S==T):
                continue
            njsd_value = calculate_njsd(lang2prob[S], lang2prob[T])
            lexical_sim[(S,T)]=njsd_value

    print(lexical_sim)

main()