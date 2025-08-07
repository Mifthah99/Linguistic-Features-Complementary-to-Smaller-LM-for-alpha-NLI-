import random
import json
import sys

def split_data(input_file='train.jsonl', labels_file='train-labels.lst', train_file='train.json', test_file='test.json',train_labels_file='train_labels.lst',test_labels_file='test_labels.lst', train_ratio=0.8,seed=33):
    
    # set random seed specified by arg
    random.seed(seed)

    # Read data and labels
    with open(input_file, 'r') as file:
        data=[line.strip() for line in file.readlines()]
    with open(labels_file, 'r') as file:
        labels=[line.strip() for line in file.readlines()]
    
    #Error if length doesnt match
    if len(data)!=len(labels):
        raise ValueError(f"Number of data items ({len(data)}) does not match number of labels ({len(labels)})")

    # Making use of indices to ensure same label and data match
    indices=list(range(len(labels)))

    # Shuffle indices
    random.shuffle(indices)

    #split shuffled indices based on train ratio
    split_idx=int(len(indices)*train_ratio)

    # set train and test indices
    train_indices=indices[:split_idx]
    test_indices=indices[split_idx:]

    # split data and labels based on indices
    train_data=[data[i] for i in train_indices]
    test_data=[data[i] for i in test_indices]

    train_labels=[labels[i] for i in train_indices]
    test_labels=[labels[i] for i in test_indices]

    # Write training and testing data and labels to file
    with open(train_file, 'w') as f:
        json.dumps(train_data)
        
    
    with open(test_file, 'w') as f:
        json.dumps(test_data)

    with open(train_labels_file, 'w') as f:
        json.dumps(train_labels)
    
    with open(test_labels_file, 'w') as f:
        json.dumps(test_labels)

    print(f"Data split complete:")
    print(f"Total records: {len(data)}")
    print(f"Training set: {len(train_data)} records ({train_ratio*100}%)")
    print(f"Testing set: {len(test_data)} records ({(1-train_ratio)*100}%)")

def print_usage():
    """Print usage information"""
    print("Usage: python split_data.py <data_file> <labels_file> <train_data_file> <train_labels_file> <test_data_file> <test_labels_file> [train_ratio] [seed]")
    print("\nArguments:")
    print("  data_file         : Path to the input JSON data file (default: 'train.jsonl')")
    print("  labels_file       : Path to the input labels file (default: 'train-labels.lst')")
    print("  train_data_file   : Path to save the training data JSON (default: 'train.json')" )
    print("  test_data_file    : Path to save the testing data JSON (default: 'test.json')")
    print("  train_labels_file : Path to save the training labels (default: 'train_labels.lst')")
    print("  test_labels_file  : Path to save the testing labels (default: 'test_labels.lst')")
    print("  train_ratio       : Proportion of data for training (default: 0.8)")
    print("  seed              : Random seed (default: 42)")
    print("\nExample:")
    print("  python split_data.py dataset.json labels.lst train.json train_labels.lst test.json test_labels.lst 0.8 42")

if __name__=='__main__':
    if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        print_usage()
        sys.exit(0)
    else:
        args = sys.argv[1:]
        split_data(*args)
