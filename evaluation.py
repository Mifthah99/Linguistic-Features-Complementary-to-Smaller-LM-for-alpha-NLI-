#def evaluate_predictions(preds, trues):
    # tp = sum((p == 1 and t == 1) for p, t in zip(preds, trues))
    # tn = sum((p == 0 and t == 0) for p, t in zip(preds, trues))
    # fp = sum((p == 1 and t == 0) for p, t in zip(preds, trues))
    # fn = sum((p == 0 and t == 1) for p, t in zip(preds, trues))

    # precision= tp/(tp+fp)
    # recall= tp/(tp+fn)
    
    #accuracy=sum((p == t) for p, t in zip(preds, trues))/len(preds)
    
    # f1 = 2*precision*recall/(precision+recall)
    
    #print(f"Accuracy=%.5f"%accuracy) #, F1 Score={f1}")

    #return accuracy
import numpy as np

def evaluate_predictions(preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    return np.mean(preds == trues)