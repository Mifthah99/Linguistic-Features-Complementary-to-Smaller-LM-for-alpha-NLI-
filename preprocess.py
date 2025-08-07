import pandas as pd
def remove_punctuations(full_text, punc):
    if type(full_text)==list:
      full_text = ' '.join(full_text)
    full_text = full_text.translate(str.maketrans('', '', ''.join(punc)))  # Faster punctuation removal
    full_text = full_text.lower()
    return full_text

def create_vocabulary(df, stopwords, punc):
    full_text = ' '.join(df['cleaned_text'].astype(str))
    full_text = full_text.translate(str.maketrans('', '', punc))  # Faster punctuation removal
    words = full_text.lower()
    words=words.split()
    vocab = set(word for word in words if word not in stopwords)
    return vocab

