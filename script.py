import sys
import os
import re
from pathlib import Path
import pathlib
import glob
from PyPDF2 import PdfReader
import shutil
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch
import torch.nn as nn
import pickle

embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 750 # max number of words in a question to use

embedding_matrix_file = os.path.join(os.getcwd(), "embedding_matrix.pickle")
with open(embedding_matrix_file, 'rb') as handle:
    embedding_matrix = pickle.load(handle)
le_file = os.path.join(os.getcwd(), "le.pickle")
with open(le_file, 'rb') as handle:
    le = pickle.load(handle)

class BiLSTM(nn.Module):
    
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 128
        drp = 0.5
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

file_path = os.path.join(os.getcwd(), "BiLSTM")

model = torch.load(file_path,map_location=torch.device('cpu'))

token_file = os.path.join(os.getcwd(), "tokenizer.pickle")
with open(token_file, 'rb') as handle:
    tokenizer = pickle.load(handle)


def cleanResume(resumeText):
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # remove non-ascii characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = re.sub(r'[0-9]+', '', resumeText)  #remove numbers
    return resumeText.lower()

def predict_single(x):    
    x = x.lower() # lower the text
    x = cleanResume(x) # Clean the text
    x = tokenizer.texts_to_sequences([x]) # tokenize
    x = pad_sequences(x, maxlen=maxlen) # pad
    x = torch.tensor(x, dtype=torch.long)#.cuda() # create dataset
    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()
    pred = pred.argmax(axis=1)
    pred = le.classes_[pred]
    return pred[0]


if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        print('path exist')
        path = os.path.join(os.getcwd(), 'Resume')
        if os.path.exists(path):
            print(path)
        else:
            os.mkdir(path)
        categorized_dic = {'filename':[],
                            'category':[]
                            }
        ext = ('.pdf')
        for root, dirs, files in os.walk(sys.argv[1]):
            for files in os.listdir(root):
                if files.endswith(ext):
                    # file_list.append(os.path.join(root,files))
                    reader = PdfReader(os.path.join(root,files))
                    # print(reader.pages[0].extract_text())
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    prdiction = predict_single(text)
                    # print(text)
                    # print(prdiction)
                    # break
                    categorized_dic['filename'].append(files)
                    categorized_dic['category'].append(prdiction)
                    if os.path.exists(os.path.join(path,prdiction)):
                        # print(os.path.join(root,files), os.path.join(path,prdiction,files))
                        shutil.copyfile(os.path.join(root,files), os.path.join(path,prdiction,files))
                    else:
                        os.mkdir(os.path.join(path,prdiction))
                        shutil.copyfile(os.path.join(root,files), os.path.join(path,prdiction,files))
                    # print(predict_single(reader.pages[0].extract_text()))
        df = pd.DataFrame.from_dict(categorized_dic)
        csv_file_path = os.path.join(os.getcwd(),'categorized_resumes.csv.')
        df.to_csv(csv_file_path, index=False)     
    else:
        print(sys.argv[1])