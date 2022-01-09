
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Models
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Training
import torch.optim as optim

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


device = torch.device("cuda:0")


# In[4]:


data = pd.read_csv("DATA_PATH")


# In[7]:


text_train, text_test, y_train, y_test = train_test_split(data['text'],
                                                          data['label'],
                                                          random_state = 42,
                                                          test_size=0.2
                                                         )


# In[8]:


tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")


# In[9]:


# Model parameter
MAX_SEQ_LEN = 128
BATCH_SIZE = 16

PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


# In[10]:


source_folder = "./JaBERT_Data"
destination_folder = "./JaBERT_Result"

def build_data(text, label) :
    tmp_df = pd.DataFrame( columns = ['label', 'text'])
    tmp_df['label'] = label
    tmp_df['text'] = text

    return tmp_df

build_data(text_train, y_train).to_csv(source_folder+"/train.csv", index=False)
build_data(text_test, y_test).to_csv(source_folder+"/test.csv", index=False)


# In[11]:


label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label', label_field), ('text', text_field)]


train, test = TabularDataset.splits(path=source_folder, train='train.csv',test='test.csv', 
                                    format='CSV', fields=fields, skip_header=True)


train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)


# In[12]:


class BERT(nn.Module):
    def __init__(self,
                 hidden_size = 768,
                 num_classes=1,
                 dr_rate=None,
                 params=None):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        self.dr_rate = dr_rate
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size , num_classes),
            nn.Sigmoid()
                                       )
            
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids,token_type_ids, attention_mask):
        
        _, pooler = self.bert(input_ids = input_ids, 
                              token_type_ids = token_type_ids, 
                              attention_mask = attention_mask)
        
        if self.dr_rate:
            out = self.dropout(pooler)
        else :
            out = pooler
        
        out = self.classifier(out)
        
        return out.squeeze(dim = 1)


# In[13]:


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# In[14]:


# Training Function

def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = test_iter,
          num_epochs = 5,
          eval_every = len(train_iter) // 5,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    tmp_loss = 0
    log_interval = 10000
    
    # training loop
    print("========= train start!! =========")
    model.train()
    for epoch in range(num_epochs):
        for (labels, text), _ in tqdm(train_loader):
            labels = labels.type(torch.FloatTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            
            #make token_type_ids, attention mask
            token_type_ids = list()
            attention_mask = list()

            for tmp in text:
                nonzero = np.nonzero(tmp).shape[0]
                token_type_ids.append([0 for x in range(MAX_SEQ_LEN)])
                attn_tmp = [1 for x in range(nonzero)]
                attn_tmp.extend([0 for x in range(MAX_SEQ_LEN-nonzero)])
                attention_mask.append(attn_tmp)            
                
            token_type_ids = torch.LongTensor(token_type_ids).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device)
            
            out = model(text, token_type_ids, attention_mask)
            
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            tmp_loss += loss.item()
            global_step += 1
            if global_step % log_interval == 0 :
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader), tmp_loss / 100))
                tmp_loss = 0
                
            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (labels, text), _ in valid_loader:
                        labels = labels.type(torch.FloatTensor)           
                        labels = labels.to(device)
                        
                        text = text.type(torch.LongTensor)  
                        text = text.to(device)
                        
                        #make token_type_ids, attention mask
                        token_type_ids = list()
                        attention_mask = list()

                        for tmp in text:
                            nonzero = np.nonzero(tmp).shape[0]
                            token_type_ids.append([0 for x in range(MAX_SEQ_LEN)])
                            attn_tmp = [1 for x in range(nonzero)]
                            attn_tmp.extend([0 for x in range(MAX_SEQ_LEN-nonzero)])
                            attention_mask.append(attn_tmp)            

                        token_type_ids = torch.LongTensor(token_type_ids).to(device)
                        attention_mask = torch.LongTensor(attention_mask).to(device)
                        
                        out = model(text, token_type_ids, attention_mask)                        

                        loss = criterion(out, labels)
                                                
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'MODEL_SAVE.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
            
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


# In[37]:


model = BERT(dr_rate=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# In[16]:


train(model=model, optimizer=optimizer)


# In[52]:


# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels,text), _ in test_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                
                #make token_type_ids, attention mask
                token_type_ids = list()
                attention_mask = list()

                for tmp in text:
                    nonzero = np.nonzero(tmp).shape[0]
                    token_type_ids.append([0 for x in range(MAX_SEQ_LEN)])
                    attn_tmp = [1 for x in range(nonzero)]
                    attn_tmp.extend([0 for x in range(MAX_SEQ_LEN-nonzero)])
                    attention_mask.append(attn_tmp)            

                token_type_ids = torch.LongTensor(token_type_ids).to(device)
                attention_mask = torch.LongTensor(attention_mask).to(device)
            
                out = model(text, token_type_ids, attention_mask)                        
                
                y_pred.extend([1 if x >= 0.5 else 0 for x in out])
                y_true.extend(labels.tolist())
    return y_true, y_pred    


# In[53]:


best_model = BERT().to(device)
load_checkpoint(destination_folder + '/MODEL_SAVE.pt', best_model)
y_true, y_pred = evaluate(best_model, test_iter)


# In[ ]:


print('Classification Report:')
print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['non-depression', 'depression'])
ax.yaxis.set_ticklabels(['non-depression', 'depression'])


# In[ ]:


print(classification_report(y_true, y_pred, labels=[0,1], digits=4))


# In[67]:


torch.save(best_model, "MODEL_SAVE.pt")

