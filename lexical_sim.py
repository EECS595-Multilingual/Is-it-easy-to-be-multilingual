import numpy as np
from scipy.spatial.distance import jensenshannon

lang2file = {
    'en' : 'en-train',
    'fi' : 'fi-train',
    'ar' : 'ar-train',
    'bn' : 'bn-train',
    'id' : 'in-train',
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
                sentences+=sentence
                sentence=""
                continue
            words = line.strip().split()
            sentence+=words[0].split(":")[1]

    return sentences

def calculate_njsd(p, q):
    # Calculate Jensen-Shannon Divergence
    jsd = jensenshannon(p, q)

    # Calculate entropy of the average distribution
    m = 0.5 * (p + q)
    entropy_m = -np.sum(m * np.log2(m))

    # Normalize the Jensen-Shannon Divergence
    njsd = jsd / np.sqrt(2 * (1 - entropy_m))

    return njsd

def convert_ngram_prob(text,n):
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
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
        text_corpus = read_data(path)
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