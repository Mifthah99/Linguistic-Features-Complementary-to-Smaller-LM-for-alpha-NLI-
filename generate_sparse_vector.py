from collections import Counter
from extract_features import extract_features # type: ignore

def sparse_vector(df, vocab,):
    vocab_size = len(vocab)
    word_index = {word: i for i, word in enumerate(sorted(vocab))}  # local reference
    segments = ['cleaned_text', 'hyp', 'obs1', 'obs2']
    X = [[] for _ in range(len(df))]

    for i, row in enumerate(df.itertuples(index=False)):
        for segment_num, segment in enumerate(segments):
            offset = segment_num * vocab_size
            words = getattr(row, segment).split()
            counts = Counter(w for w in words if w in vocab)
            for word, count in counts.items():
                X[i].append((word_index[word] + offset, count))

        features = extract_features(row._asdict())  # extract_features takes a dict as input
        for j, value in enumerate(features):
            if value > 0:
                X[i].append((4 * vocab_size + j, value))
        j=len(features)
        feature_count=j+(len(segments)*vocab_size)

    return X,feature_count
