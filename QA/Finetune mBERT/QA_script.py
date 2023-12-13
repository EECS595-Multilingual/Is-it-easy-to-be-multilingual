import json
import random
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
from tqdm.auto import tqdm
import numpy as np
import evaluate
import collections
from collections import Counter
from nltk.util import ngrams
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-uncased")


S_lang2file = {
    'en' : 'tydiqa.en.train.json',
    'fi' : 'tydiqa.fi.train.json',
    'ar' : 'tydiqa.ar.train.json',
    'bn' : 'tydiqa.bn.train.json',
    'id' : 'tydiqa.id.train.json',
    'ko' : 'tydiqa.ko.train.json',
    'ru' : 'tydiqa.ru.train.json',
    'sw' : 'tydiqa.sw.train.json',
    'te' : 'tydiqa.te.train.json',
}

T_lang2file = {
    'en' : 'tydiqa.en.dev.json',
    'fi' : 'tydiqa.fi.dev.json',
    'ar' : 'tydiqa.ar.dev.json',
    'bn' : 'tydiqa.bn.dev.json',
    'id' : 'tydiqa.id.dev.json',
    'ko' : 'tydiqa.ko.dev.json',
    'ru' : 'tydiqa.ru.dev.json',
    'sw' : 'tydiqa.sw.dev.json',
    'te' : 'tydiqa.te.dev.json',
}

accuracy_dict = {} # for storing all the test accuracies in the form { (S,T,SHOT) , Acc }
init_acc_dict = {}

# All Languages: en, fi, ar, bn, id, ko, ru, sw, te = 9
# Total Language pairs = 9*9 = 81

SHOT = 0

max_length = 384
stride = 128

path = ""

metric = evaluate.load("squad")


def read_data(path):  
    with open(path, 'rb') as f:
        squad = json.load(f)
    contexts = []
    questions = []
    answers = []
    id = []
    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    id.append(qa['id'])
    return contexts, questions, answers, id

def get_s_data(S,T,SHOT):
    s_path = path + 'tydiqa-goldp-v1.1-train/' + S_lang2file[S]
    s_context, s_q, s_a, s_i = read_data(s_path)
    s_tydi = []
    for _ in range(len(s_a)):
        s_tydi.append({})
        s_tydi[_]['answers'] = s_a[_]
        s_tydi[_]['context'] = s_context[_]
        s_tydi[_]['question'] = s_q[_]
        s_tydi[_]['id'] = s_i[_]
    if SHOT>0:
        few_shot_path = path + 'tydiqa-goldp-v1.1-train/' + S_lang2file[T]
        fs_context, fs_q, fs_a, fs_i = read_data(few_shot_path)
        for _ in range(SHOT):
            s_tydi.append({})
            s_tydi[len(s_tydi) - 1]['answers'] = fs_a[_]
            s_tydi[len(s_tydi) - 1]['context'] = fs_context[_]
            s_tydi[len(s_tydi) - 1]['question'] = fs_q[_]
            s_tydi[len(s_tydi) - 1]['id'] = fs_i[_]
    s_data = Dataset.from_list(s_tydi)
    return s_data

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def get_t_data(T):
    t_path = path + 'tydiqa-goldp-v1.1-dev/' + T_lang2file[T]
    t_context, t_q, t_a, t_i = read_data(t_path)
    t_tydi = []
    for _ in range(len(t_a)):
        t_tydi.append({})
        t_tydi[_]['answers'] = t_a[_]
        t_tydi[_]['context'] = t_context[_]
        t_tydi[_]['question'] = t_q[_] 
        t_tydi[_]['id'] = t_i[_] 
    t_data = Dataset.from_list(t_tydi)
    return t_data


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logit)[-1 : -20 - 1 : -1].tolist() #n_best
            end_indexes = np.argsort(end_logit)[-1 : -20 - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > 60 #max_answer_len
                    ):
                        continue
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    theoretical_answers = [{"id": ex["id"], "answers": {"text":[ex["answers"]["text"]], "answer_start":[ex["answers"]["answer_start"]]}} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



def model_train(tr_data, te_data):
    data_collator = DefaultDataCollator()
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-uncased")
    training_args = TrainingArguments(
        output_dir='QA_OP',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_data,
        eval_dataset=te_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer


### FOR LEXICAL SIMILARITY

def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence.lower(), return_tensors='pt')
        tokenized_sentences.extend([token for token in tokenized_sentence])
    return tokenized_sentences

def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def compute_ngram_freq(ngram_list):
    ngram_freq = Counter(ngram_list)
    total_ngrams = sum(ngram_freq.values())
    ngram_prob = {ngram: freq / total_ngrams for ngram, freq in ngram_freq.items()}
    return ngram_prob

jsd_dict = {}

for S in S_lang2file.keys(): # S_lang2file.keys()
    for T in T_lang2file.keys(): # T_lang2file.keys()

        # if S==T:
        #     continue
        
        s_data = get_s_data(S,T,SHOT)
        t_data = get_s_data(T,T,SHOT)

        corpus1_sentences = s_data['context']
        corpus2_sentences = t_data['context']

        # Tokenize sentences in each corpus into tokens using the BERT tokenizer
        corpus1_tokens = tokenize_sentences(corpus1_sentences)
        corpus2_tokens = tokenize_sentences(corpus2_sentences)

        # Generate 3-grams for both corpora
        n = 3  # Change n to the desired n-gram size
        corpus1_3grams = generate_ngrams(corpus1_tokens, n)
        corpus2_3grams = generate_ngrams(corpus2_tokens, n)

        # Compute n-gram probabilities for each corpus
        corpus1_ngram_prob = compute_ngram_freq(corpus1_3grams)
        corpus2_ngram_prob = compute_ngram_freq(corpus2_3grams)

        # Convert dictionaries to lists of probabilities
        corpus1_prob_list = list(corpus1_ngram_prob.values())
        corpus2_prob_list = list(corpus2_ngram_prob.values())

        # Calculate Jensen-Shannon Divergence using scipy's jensenshannon function
        min_len = min(len(corpus1_prob_list), len(corpus2_prob_list))
        s = corpus1_prob_list[:min_len]
        t = corpus2_prob_list[:min_len]
        jsd_result = jensenshannon(s, t) / np.sqrt(2)
        print(f"Jensen-Shannon Divergence between {S} , {T} : {jsd_result}")        
        jsd_dict[(S,T)] = jsd_result

jsd_dict
with open("tydiqa_lex.txt", "w") as fp:
    print(jsd_dict, file=fp)

### LEXICAL SIMILARITY ENDS

### FOR LM feature


for T in T_lang2file.keys(): # T_lang2file.keys(T)
   t_data = get_t_data(T)
   validation_dataset = t_data.map(preprocess_validation_examples, batched=True, remove_columns=t_data.column_names)
   eval_set_for_model = validation_dataset.remove_columns(["example_id", "offset_mapping"])
   eval_set_for_model.set_format('torch')
   batch = {k: eval_set_for_model[k] for k in eval_set_for_model.column_names}
   model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-uncased")
   with torch.no_grad():
       outputs = model(**batch)
   start_logits = outputs.start_logits.cpu().numpy()
   end_logits = outputs.end_logits.cpu().numpy()
   f = compute_metrics(start_logits, end_logits, validation_dataset, t_data)
   init_acc_dict[T] = f
   print(init_acc_dict)

with open("Init_Acc_QA.txt", "w") as fp:
   json.dump(init_acc_dict, fp)


### LM Feature ENDS

### MODEL TRAINING

for S in S_lang2file.keys(): # S_lang2file.keys()
    train_counter = 1
    for T in T_lang2file.keys(): # T_lang2file.keys()
        s_data = get_s_data(S,T,SHOT)
        train_dataset = s_data.map(preprocess_training_examples, batched=True, remove_columns=s_data.column_names)
        t_data = get_t_data(T)
        validation_dataset = t_data.map(preprocess_validation_examples, batched=True, remove_columns=t_data.column_names)
        if train_counter == 1:
            trainer = model_train(train_dataset, validation_dataset)
        train_counter = train_counter + 1
        predictions, _, _ = trainer.predict(validation_dataset)
        start_logits, end_logits = predictions
        f1 = compute_metrics(start_logits, end_logits, validation_dataset, t_data)
        accuracy_dict[(S,T,SHOT)] = f1

with open("Acc_QA.txt", "w") as fp:
    print(accuracy_dict, file=fp)

###