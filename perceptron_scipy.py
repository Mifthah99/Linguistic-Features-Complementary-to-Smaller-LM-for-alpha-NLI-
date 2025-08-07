import numpy as np

class Perceptron:
    def __init__(self, l_r=0.02, n_iters=1000):
        self.lr = l_r
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0
        self.max_acc = 0
        self.best_weights = None

    def activation_fn(self, x):
        return np.where(x > 0, 1, -1)

    def partial_fit(self, X_train, Y_train, X_dev, Y_dev):
        n_features = X_train.shape[1]
        self.weights = np.random.rand(n_features) if self.weights is None else self.weights

        for i in range(self.n_iters):
            # Compute predictions
            scores = X_train.dot(self.weights) + self.bias
            preds = self.activation_fn(scores)

            # Update weights where incorrect
            for idx in np.where(preds != Y_train)[0]:
                update = self.lr * (Y_train[idx] - preds[idx])
                self.weights += update * X_train[idx].toarray().ravel()
                self.bias += update

            # Dev accuracy check
            dev_preds = self.pred_batch(X_dev)
            acc = evaluate_predictions(dev_preds, Y_dev)
            if (i + 1) % 100 == 0 or acc > self.max_acc:
                print(f"Iter {i+1}/{self.n_iters}: Accuracy = {acc:.4f}, Max Accuracy = {self.max_acc:.4f}")
            if acc > self.max_acc:
                self.max_acc = acc
                self.best_weights = self.weights.copy()

    def pred_batch(self, X, best=False):
        weights = self.best_weights if best and self.best_weights is not None else self.weights
        scores = X.dot(weights) + self.bias
        return self.activation_fn(scores)

from scipy.sparse import csr_matrix
from collections import Counter

def generate_bow_vectors_sparse(df, vocab, word_to_index):
    vocab_size = len(vocab)
    segments = ['cleaned_text', 'hyp', 'obs1', 'obs2']
    
    data, rows, cols = [], [], []
    for row_idx, row in df.iterrows():
        for seg_idx, seg in enumerate(segments):
            offset = seg_idx * vocab_size
            words = row[seg].lower().split()
            counts = Counter(w for w in words if w in vocab)
            for word, count in counts.items():
                col_idx = word_to_index[word] + offset
                data.append(count)
                rows.append(row_idx)
                cols.append(col_idx)
    
    mat = csr_matrix((data, (rows, cols)), shape=(len(df), vocab_size * 4))
    return mat

if __name__ == "__main__":
    from read_data import read_data
    from preprocess import create_vocabulary
    from evaluation import evaluate_predictions

    train_df = read_data('train.json', 'train_labels.lst')
    dev_df = read_data('dev.jsonl', 'dev-labels.lst')

    stopwords = [...]  # your stopwords
    punc = ''.join(['.', ',', "'", ":", "?", '!', '@', '/', '&'])
    vocab = create_vocabulary(train_df, stopwords, punc)
    word_to_index = {word: i for i, word in enumerate(sorted(vocab))}
    
    X_train = generate_bow_vectors_sparse(train_df, vocab, word_to_index)
    Y_train = train_df['Y'].values
    X_dev = generate_bow_vectors_sparse(dev_df, vocab, word_to_index)
    Y_dev = dev_df['Y'].values

    model = Perceptron(n_iters=1000)
    print("Training...")
    model.partial_fit(X_train, Y_train, X_dev, Y_dev)
