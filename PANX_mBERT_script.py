import json
from datasets import Dataset
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification, DefaultDataCollator, TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-uncased")
SHOT = 0 # 0-shot or few-shot
S_lang2file = {
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

T_lang2file = {
    'en' : 'en-dev',
    'fi' : 'fi-dev',
    'ar' : 'ar-dev',
    'bn' : 'bn-dev',
    'id' : 'in-dev',
    'ko' : 'ko-dev',
    'ru' : 'ru-dev',
    'sw' : 'sw-dev',
    'te' : 'te-dev',
}
accuracy_dict = {} # for storing all the test accuracies in the form { (S,T,SHOT) , Acc }

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
                sentences.append(sentence)
                sentence=[]
                tags.append(sent_tag)
                sent_tag=[]
            words = line.strip().split()
            sent_tag.append(words[1])
            sentence.append(words[0].split(":")[1])

    return sentences,tags




def preprocess_function(sentences,tags):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=200,
        is_split_into_words=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask=inputs["attention_mask"]

    label_map = {}
    for sentence in tags:
        for tag in sentence:
            if tag not in label_map:
                label_map[tag]=len(label_map)
    numerical_labels = [[label_map[tag] for tag in sentence] for sentence in tags]

    padded_labels = [torch.nn.functional.pad(torch.tensor(labels),(0,200-len(labels)),value=-100) for labels in numerical_labels]

    return TensorDataset(input_ids, attention_mask, torch.stack(padded_labels)), len(label_map)+1





### To run it all at once  -

for S in S_lang2file.keys():
    for T in T_lang2file.keys():

        s_path = path + S_lang2file[S]
        t_path = path +  T_lang2file[T]
        s_sent,s_tag = read_data(s_path)
        t_sent,t_tag = read_data(t_path)

        tokenized_s_data = preprocess_function(s_sent,s_tag)
        tokenized_t_data = preprocess_function(t_sent,t_tag)
        data_collator = DefaultDataCollator()
        model.config.num_labels=num_class
        model.classifier = torch.nn.Linear(model.config.hidden_size,num_class)
        # TRAIN
        train_dataloader = DataLoader(tokenized_s_data, batch_size=32, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=1e-5)

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
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                # Get predicted labels
                pred_labels = torch.argmax(outputs.logits, dim=1)

                # Calculate accuracy
                correct = (pred_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.numel() - (labels== -100).sum().item() 
                loss.backward()
                optimizer.step()
            average_loss = total_loss / len(train_dataloader)
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Training Accuracy: {accuracy}")


        # TEST
        test_dataloader = DataLoader(tokenized_t_data, batch_size=32)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # Get predicted labels
                pred_labels = torch.argmax(outputs.logits, dim=1)

                # Calculate accuracy
                correct = (pred_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.numel() - (labels== -100).sum().item() 
        accuracy = total_correct / total_samples
        print(f"Testing Accuracy: {accuracy}")


        accuracy_dict[(S,T,SHOT)] = accuracy