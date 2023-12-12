import json
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-uncased")

S_lang2file = {
    'en' : 'tydiqa.en.train.json',
    'fi' : 'tydiqa.fi.train.json',
    'ar' : 'tydiqa.ar.train.json',
    'bn' : 'tydiqa.bn.train.json',
    'id' : 'tydiqa.in.train.json',
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
    'id' : 'tydiqa.in.dev.json',
    'ko' : 'tydiqa.ko.dev.json',
    'ru' : 'tydiqa.ru.dev.json',
    'sw' : 'tydiqa.sw.dev.json',
    'te' : 'tydiqa.te.dev.json',
}

accuracy_dict = {} # for storing all the test accuracies in the form { (S,T,SHOT) , Acc }


path = "/Users/rishikesh/Desktop/Project/download/tydiqa/"

def read_data(path):  
    with open(path, 'rb') as f:
        squad = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


# All Languages: en, fi, ar, bn, id, ko, ru, sw, te = 9
# Total Language pairs = 9*9 = 81

SHOT = 0 # 0-shot or few-shot

# S = 'en'
# T = 'fi'  

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=400,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
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

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
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





### To run it all at once  -

for S in S_lang2file.keys():
    for T in T_lang2file.keys():

        s_path = path + 'tydiqa-goldp-v1.1-train/' + S_lang2file[S]
        t_path = path + 'tydiqa-goldp-v1.1-dev/' + T_lang2file[T]
        s_context, s_q, s_a = read_data(s_path)
        t_context, t_q, t_a = read_data(t_path)
        if SHOT>0:
            few_shot_path = path + 'tydiqa-goldp-v1.1-train/' + S_lang2file[T]
            fs_context, fs_q, fs_a = read_data(few_shot_path)

        s_tydi = []
        for _ in range(len(s_a)):
            s_tydi.append({})
            s_tydi[_]['answers'] = s_a[_]
            s_tydi[_]['context'] = s_context[_]
            s_tydi[_]['question'] = s_q[_]
        if SHOT>0:
            for _ in range(SHOT):
                s_tydi.append({})
                s_tydi[len(s_tydi) - 1]['answers'] = fs_a[_]
                s_tydi[len(s_tydi) - 1]['context'] = fs_context[_]
                s_tydi[len(s_tydi) - 1]['question'] = fs_q[_]
        s_data = Dataset.from_list(s_tydi)

        t_tydi = []
        for _ in range(len(t_a)):
            t_tydi.append({})
            t_tydi[_]['answers'] = t_a[_]
            t_tydi[_]['context'] = t_context[_]
            t_tydi[_]['question'] = t_q[_]  
        t_data = Dataset.from_list(t_tydi)


        tokenized_s_data = s_data.map(preprocess_function, batched=True, batch_size=32, remove_columns=s_data.column_names)
        tokenized_t_data = t_data.map(preprocess_function, batched=True, batch_size=32, remove_columns=t_data.column_names)
        data_collator = DefaultDataCollator()

        # TRAIN
        tokenized_s_data.set_format("torch")
        train_dataloader = DataLoader(tokenized_s_data, batch_size=32, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-uncased")

        # Training loop
        num_epochs = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for batch in tqdm(train_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss
                total_loss += loss.item()
                # Get predicted start and end positions
                pred_start_positions = torch.argmax(outputs.start_logits, dim=1)
                pred_end_positions = torch.argmax(outputs.end_logits, dim=1)
                # Calculate accuracy
                correct_start = (pred_start_positions == start_positions).sum().item()
                correct_end = (pred_end_positions == end_positions).sum().item()
                total_correct += correct_start + correct_end
                total_samples += start_positions.size(0) * 2  # Multiply by 2 as we have start and end positions
                loss.backward()
                optimizer.step()
            average_loss = total_loss / len(train_dataloader)
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Training Accuracy: {accuracy}")


        # TEST
        tokenized_t_data.set_format("torch")
        test_dataloader = DataLoader(tokenized_t_data, batch_size=32)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                # Get predicted start and end positions
                pred_start_positions = torch.argmax(outputs.start_logits, dim=1)
                pred_end_positions = torch.argmax(outputs.end_logits, dim=1)
                # Calculate accuracy
                correct_start = (pred_start_positions == start_positions).sum().item()
                correct_end = (pred_end_positions == end_positions).sum().item()
                total_correct += correct_start + correct_end
                total_samples += start_positions.size(0) * 2  # Multiply by 2 as we have start and end positions
        accuracy = total_correct / total_samples
        print(f"Testing Accuracy: {accuracy}")


        accuracy_dict[(S,T,SHOT)] = accuracy