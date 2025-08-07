from collections import Counter
from generate_sparse_vector import sparse_vector # type: ignore
from tqdm import tqdm
import _pickle as pickle
import pandas as pd
import numpy as np
from evaluation import evaluate_predictions


def step_fn(x):
    return np.where(x > 0, 1, -1)

class Perceptron:
    def __init__(self, l_r=0.40, epochs=1000,decay_rate=0.99,best=False,no_of_feat=106868):
        self.lr = l_r
        self.best=best
        self.epochs = epochs
        self.decay_rate = decay_rate  # decay factor for exponential decay
        self.activation_fn = step_fn
        self.weights = None
        self.bias = None
        self.best_weights=None
        self.best_bias=None
        self.max_acc=0
        self.n_features=no_of_feat

    def decay_lr(self, iteration):
        """Apply exponential learning rate decay."""
        return self.lr * (self.decay_rate ** iteration)
    
    def partial_fit(self, X_train, Y, X_dev,Y_dev):
        '''
        Inputs: 
          For training:
            X_train         list of features represented in the form of a sparse vector dictionary for each row of training data
            Y               list of actual predictions for the training data
          To check change in accuracy and save best weights
            X_dev           list of features represented in the form of a sparse vector dictionary for each row of dev data
            Y_dev           list of actual predictions for the dev data

        Outputs:
            Prints accuracy for each iteration
            saves latest weights and bias and best weights for model  
        '''
        n_samples = len(Y)
        n_features = self.n_features
        self.weights = np.random.rand(n_features) if self.weights is None else self.weights
        self.bias = 0 if self.bias is None else self.bias
        pbar=tqdm(total=self.epochs,desc="Training iterations")
        bar=0#update progress bar
        for i in range(self.epochs):
            
            # Shuffle data at the start of each epoch to improve generalization
            indices = np.random.permutation(n_samples)
            X_train = [X_train[idx] for idx in indices]
            Y = [Y[idx] for idx in indices]

            #set current l_r
            lr=self.decay_lr(i)
            for x_list, y_true in zip(X_train, Y):
                score=0 # initialize value to 0 for each row of training data
                for index, value in x_list:
                    score+= self.weights[index]*value
                y_pred = self.activation_fn(score+self.bias)
                if y_pred != y_true:
                    update = lr * (y_true - y_pred)
                    for index, value in x_list:
                        self.weights[index]+= update * value 
                    self.bias += update

            y_pred=[self.pred(x,self.best) for x in X_dev]
            accuracy=evaluate_predictions(y_pred,Y_dev)
            bar+=1
            if (i + 1) % 100 == 0:
                pbar.update(bar)
                bar=0
                print(f' Epoch {i + 1}: Accuracy = {accuracy:.5f}, Max Accuracy = {self.max_acc:.5f}')
                
            if accuracy>self.max_acc:
                pbar.update(bar)
                bar=0
                print (f' Epoch {i+1} of {self.epochs} : Accuracy = {accuracy}')
                self.best_weights=self.weights.copy()
                self.best_bias=self.bias
                self.max_acc=accuracy
            if self.best:
                self.weights=self.best_weights
                self.bias=self.best_bias

        pbar.close()
    def pred(self, X,best=False):
        weights=self.weights
        bias=self.bias
        if best:
            weights=self.best_weights
            bias=self.best_bias
        op=0
        for index, value in X:
                op+= weights[index]*value
        y_pred = self.activation_fn(op+bias)
        return y_pred
   
            
# Pickle to store and load model
def store_model(model, filename):
    with open(filename,'wb') as f:
        pickle.dump(model,f)
    return
def load_model(filename):
    with open(filename,'rb')as f:
        model=pickle.load(f)
    return model
# pickle store and load vocabulary
def store_vocab(vocab,filename='vocab.pkl'):
    with open(filename,'wb') as f:
        pickle.dump(vocab,f)

def load_vocab(filename='vocab.pkl'):
    with open(filename,'rb') as f:
        vocab=pickle.load(f)
        return vocab
    
# if __name__=="__main__":

#     import pandas as pd
#     import json

#     from evaluation import evaluate_predictions
#     from read_data import read_data
#     from preprocess import remove_punctuations,create_vocabulary
    
#     batch_size=2000

#     train_data='train.json'
#     train_labels='train_labels.lst'

#     test_data="test.json"
#     test_labels='test_labels.lst'
#     dev_data='dev.jsonl'
#     dev_labels='dev-labels.lst'
#     stopwords=['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','a','an','the']
#     punc=''.join(['.',',',"'","'",":","?",'!','@','/','&'])
#     #read train data
#     train_df=read_data(train_data,train_labels)
#     #print(type(train_df))
#     vocab=create_vocabulary(train_df,stopwords,punc)
#     store_vocab(vocab)
#     #print(vocab)
#     vocab_size=len(vocab)
#     X_gen = sparse_vector(train_df, vocab)
#     #print(X_gen[0]==X_gen[1002])
#     Y = train_df['Y'].values 
#     dev_df=read_data(dev_data,dev_labels)
#     X_dev=sparse_vector(dev_df,vocab)
#     Y_dev=dev_df['Y'].values
    
#     # training from scratch
#     # model = Perceptron(l_r=0.3,n_iters=1000)

#     #training update and testing
#     model=load_model('model4.pkl')
#     #retraining the model for weight updation
#     # model.n_iters=100
#     # model.weights=model.best_weights.copy()
#     # print("Training Model")
#     # model.partial_fit(X_gen, Y, X_dev,Y_dev)
#     # model_fname="model4.pkl"
#     # print(f"Storing Model {model_fname}")
#     # store_model(model,model_fname)

    

#     # testing model performance using dev data
#     best=True
#     y_pred=[model.pred(x) for x in X_dev]
#     y_pred_best=[model.pred(x,best)for x in X_dev]
#     accuracy_dev=evaluate_predictions(y_pred,Y_dev)
#     acc_dev_best=evaluate_predictions(y_pred_best,Y_dev)
#     print(f'Accuracy Dev data (normal weights): {accuracy_dev}\nAccuracy dev best weights:{acc_dev_best}')
#     #testing accuracy on test data
#     test_df=read_data(test_data,test_labels)
#     X_test=sparse_vector(test_df,vocab)
#     Y_test=test_df['Y'].values
#     print('Model accuracy for test data')
#     y_pred=[model.pred(x) for x in X_test]
 
#     y_pred_best=[model.pred(x,best) for x in X_test]
#     accuracy_test=evaluate_predictions(y_pred,Y_test)
#     accuracy_test_best=evaluate_predictions(y_pred_best,Y_test)
#     print(f"Accuracy for test data normal weights:{accuracy_test}\nAccuracy test data best weights:{accuracy_test_best}")
#     # Accuracy=0.5094647519582245


    

    

    
    