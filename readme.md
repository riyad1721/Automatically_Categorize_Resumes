# Instruction about the folder structure and run the script file<br />
## Overview folder structure
<br />
directory sturcture is 
<br />
=> Training Notebook : <br />
=>=>Resume_Classification.ipynb <br />
        under Training Notebook folder Resume Classification .ipynb file
        which trains the model based on training dataset and all code explanations have under the notebook markdown text.<br />
=> BILSTM : <br />
    This is trained model which is trained and mimic the data on model training time <br />
=>embedding_matrix.pickle: <br />
    Save the embadding matrix which is used to training the model <br />
=>le.pickle: <br />
    save the label encoder in a pickle format which need to testing and predict the the classes <br />
=> tokenizer.pickle: <br />
    Here I am using keras.preprocessing.text.Tokenizer tokenizer (splits) a text into tokens in a train session and save it and use it test data to split text
     into token <br />

## Overview of scripts.py file <br />
Under the script.py file I am testing the model and giving a prediction based on the model learned.here i am using all necessary files which i am saving in training time and also used the trained model class. the file will test only the .pdf file and also summarize the file based on prediction and copy in a folder. and produce a .csv file also
<br />

## Run Script Instructions- <br />
The script should be executable from the command line as follows: python script.py path/to/dir
Note: Make sure the directory path do no have any space