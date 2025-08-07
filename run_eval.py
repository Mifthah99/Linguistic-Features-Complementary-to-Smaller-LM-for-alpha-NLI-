import sys
from evaluation import evaluate_predictions
if __name__==__main__:
  if len(sys.argv)<3:
    print("Enter files for trues and prediction")
    sys.exit(0)
  actual=sys.argv[1]
  pred=sys.argv[2]
  
  with open(actual) as act:
      trues=[line.strip()  for line in act.readlines()]
  with open(pred ) as pred:
      preds=[line.strip() for line in pred.readlines()]
  tp, tn, fp, fn,precision,recall,accuracy, f1 = evaluate_predictions(preds, trues)
  
  print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
  print(f"F1 Score: {f1:.4f}")
  print(f"Accuracy: {accuracy:.4f}")
