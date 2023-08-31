README.md

## APPROACH 1 : LM-ONLY QA MODEL

- Run LMQA_final.ipynb

## APPROACH 2 : QAGNN [LM+KG QA MODEL]

## Step 1 : COMPUTE USED ##
2 RTX-6000 GPUs [Cuda 11.7.0]

## Step 2 : CREATE ENVIRONMENT ##
conda create -n qagnn python=3.7
source activate qagnn
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.4.0
pip install nltk spacy==2.1.6
python -m spacy download en
pip install torch-sparse torch-geometric torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch.__version__}.html

## Step 4 : DOWNLOAD RAW DATA ##
./ download_raw_data.sh

## Step 3 : CREATE PREPROCESSED DATA ##
python preprocess.py

## Step 4 :  RUN BASH FILE ##
./ run_qagnn__obqa.sh

- Check saved_model folder for epoch-wise predictions on test and metric csv
- Check logs/ for training status.

