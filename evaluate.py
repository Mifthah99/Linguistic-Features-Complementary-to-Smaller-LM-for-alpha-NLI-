import _pickle as pickle
import pandas as pd
import numpy as np
import sys
import json
from evaluation import evaluate_predictions
from read_data import read_data
from preprocess import remove_punctuations,create_vocabulary
from collections import Counter
from generate_sparse_vector import sparse_vector
from tqdm import tqdm
from perceptron import Perceptron, step_fn, store_model, load_model, store_vocab, load_vocab
if __name__=="__main__":
    arg_list=sys.argv
    if len(arg_list) >1:
        model_file=arg_list[1]
    else:
        model_file='model_new.pkl'
        print(f"Using default model {model_file}\nTo run with other model please enter model file location as arg")
    
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
    #print(type(train_df))

    #create and store vocabulary
    vocab=create_vocabulary(train_df,stopwords,punc)
    store_vocab(vocab)
    #print(vocab)

    vocab_size=len(vocab)

    # generate sparse vector for training data
    X_gen,_ = sparse_vector(train_df, vocab)
    #print(X_gen[0]==X_gen[1002])
    Y = train_df['Y'].values 

    # Read and generate sparse vector for Dev data
    dev_df=read_data(dev_data,dev_labels)
    X_dev,_=sparse_vector(dev_df,vocab)
    Y_dev=dev_df['Y'].values
    
    # training from scratch
    # model = Perceptron(l_r=0.3,n_iters=1000)

    #training update and testing
    try:
        model=load_model(model_file)
    except Exception as e:
        print("Error: Please check model file is properly entered",e)
        exit()
    
    #retraining the model for weight updation
    # model.n_iters=100
    # model.weights=model.best_weights.copy()
    # print("Training Model")
    # model.partial_fit(X_gen, Y, X_dev,Y_dev)
    # model_fname="model4.pkl"
    # print(f"Storing Model {model_fname}")
    # store_model(model,model_fname)

    

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
