import _pickle as pickle
import pandas as pd
import numpy as np
import sys
import json
from evaluation import evaluate_predictions
from read_data import read_data
from preprocess import create_vocabulary
from collections import Counter
from generate_sparse_vector import sparse_vector
from tqdm import tqdm
from perceptron import Perceptron, step_fn, store_model, load_model, store_vocab, load_vocab
if __name__=="__main__":
    arg_list=sys.argv
    epochs=200
    model_update=False
    if len(sys.argv) >1:
        for i in arg_list[1:]:
            if 'model' in i:
                model_file=i
                model_update=True
            if 'epochs' in i:
                epochs=int(i[7:])


    if not model_update:
        model_file='model_latest.pkl'
        print(f"Model name: {model_file}\n Using default settings (epochs=200) .\n To update a model please enter model file location as arg ex: train_model.py model4.pkl epochs=100")
    vocab_file='vocab.pkl'
    train_data='train.json'
    train_labels='train_labels.lst'
    test_data="test.json"
    test_labels='test_labels.lst'
    dev_data='dev.jsonl'
    dev_labels='dev-labels.lst'
    stopwords=['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','a','an','the']
    punc=''.join(['.',',',"'","'",":","?",'!','@','/','&'])
    
    #read train data
    train_df=read_data(train_data,train_labels)

    #create and store vocabulary
    vocab=create_vocabulary(train_df,stopwords,punc)
    store_vocab(vocab)

    vocab_size=len(vocab)

    # generate sparse vector for training data
    X_train, no_of_feat = sparse_vector(train_df, vocab)

    Y_train= train_df['Y'].values 

    # Read and generate sparse vector for Dev data
    dev_df=read_data(dev_data,dev_labels)
    X_dev,_=sparse_vector(dev_df,vocab)
    Y_dev=dev_df['Y'].values
    
    # training from scratch
    # model = Perceptron(l_r=0.3,n_iters=1000)

    #training update and testing
    
    
    #retraining the model for weight updation
    if model_update:
        try:
            model=load_model(model_file)
        except Exception as e:
            print("Error: Please check model file is properly entered",e)
            exit()
        model.epochs=epochs
        print("updating Model")
        model.partial_fit(X_train, Y_train, X_dev,Y_dev)
        print(f"Storing Model {model_file}")
        store_model(model,model_file)
        
    else:
        model=Perceptron(epochs=epochs,no_of_feat=no_of_feat)
        print('Training model..........')
        model.partial_fit(X_train,Y_train, X_dev,Y_dev)
        print(f"Storing Model {model_file}")
        store_model(model,model_file)
        
    

    # testing model performance using dev data
    best=True
    y_pred=[model.pred(x) for x in X_dev]
    y_pred_best=[model.pred(x,best)for x in X_dev]
    accuracy_dev=evaluate_predictions(y_pred,Y_dev)
    acc_dev_best=evaluate_predictions(y_pred_best,Y_dev)
    print(f'Accuracy Dev data (normal weights): {accuracy_dev}\nAccuracy dev best weights:{acc_dev_best}')
    #testing accuracy on test data
    test_df=read_data(test_data,test_labels)
    X_test,_=sparse_vector(test_df,vocab)
    Y_test=test_df['Y'].values
    print('Model accuracy for test data')
    y_pred=[model.pred(x) for x in X_test]
  
    # use best weights
    y_pred_best=[model.pred(x,best) for x in X_test]
    accuracy_test=evaluate_predictions(y_pred,Y_test)
    accuracy_test_best=evaluate_predictions(y_pred_best,Y_test)
    print(f"Accuracy for test data normal weights:{accuracy_test}\nAccuracy test data best weights:{accuracy_test_best}")
    # Accuracy=0.5094647519582245
