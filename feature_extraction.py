import pandas as pd
import re
import json

NEGATIONS = {"no", "not", "never", "none", "nothing", "neither", "nobody", "nowhere"}
POSITIVE_WORDS = {"good", "happy", "love", "excellent", "great", "positive", "joy"}
NEGATIVE_WORDS = {"bad", "sad", "hate", "terrible", "awful", "negative", "pain"}

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def word_overlap(set1, set2):
    return len(set1 & set2)

def sentiment_score(tokens):
    pos = len([w for w in tokens if w in POSITIVE_WORDS])
    neg = len([w for w in tokens if w in NEGATIVE_WORDS])
    return pos - neg

def contains_negation(tokens):
    return any(w in NEGATIONS for w in tokens)

def get_overlap_bin(overlap):
    if overlap <= 10:
        return "1_10"
    elif overlap <= 20:
        return "11_20"
    elif overlap <= 30:
        return "21_30"
    else:
        return "31_plus"

def get_sentiment_bin(score):
    if score < 0:
        return "neg"
    elif score == 0:
        return "neutral"
    else:
        return "pos"

def extract_features(row):
    obs1_tokens = set(tokenize(row['obs1']))
    obs2_tokens = set(tokenize(row['obs2']))
    hyp_tokens = set(tokenize(row['hyp']))

    all_obs_tokens = obs1_tokens | obs2_tokens

    jaccard_obs_hyp = jaccard_similarity(all_obs_tokens, hyp_tokens)
    overlap_all = word_overlap(all_obs_tokens, hyp_tokens)
    overlap_obs1 = word_overlap(obs1_tokens, hyp_tokens)
    overlap_obs2 = word_overlap(obs2_tokens, hyp_tokens)

    sentiment_obs = sentiment_score(list(all_obs_tokens))
    sentiment_hyp = sentiment_score(list(hyp_tokens))

    negation_obs = contains_negation(all_obs_tokens)
    negation_hyp = contains_negation(hyp_tokens)

    features = {
        "jaccard_obs_hyp": jaccard_obs_hyp,
        "overlap_all": overlap_all,
        "overlap_obs1": overlap_obs1,
        "overlap_obs2": overlap_obs2,
        "overlap_bin_" + get_overlap_bin(overlap_all): 1,
        "sentiment_obs": sentiment_obs,
        "sentiment_hyp": sentiment_hyp,
        "sentiment_bin_obs_" + get_sentiment_bin(sentiment_obs): 1,
        "sentiment_bin_hyp_" + get_sentiment_bin(sentiment_hyp): 1,
        "negation_obs": int(negation_obs),
        "negation_hyp": int(negation_hyp)
    }

    return features

def read_data(input, labels):
    with open(input) as f:
        data = f.readlines()
    with open(labels) as f:
        labels = f.readlines()
    datap = [json.loads(i) for i in data]
    d = pd.DataFrame(datap)
    y = [i.strip() for i in labels]
    d['Y'] = y
    d1 = d[['story_id', 'obs1', 'obs2', 'hyp1', 'Y']].rename(columns={'hyp1': 'hyp'})
    d2 = d[['story_id', 'obs1', 'obs2', 'hyp2', 'Y']].rename(columns={'hyp2': 'hyp'})
    d1['ord'] = 1
    d2['ord'] = 2
    df = pd.concat([d1, d2], ignore_index=True).sort_values(['story_id', 'ord'])
    df['temp'] = df.apply(lambda row: 1 if int(row['Y']) == int(row['ord']) else 0, axis=1)
    df['Y'] = df['temp']
    df.drop(['ord', 'temp'], axis=1, inplace=True)
    return df

if __name__ == "__main__":
    json_path = "train.json"
    label_path = "train_labels.lst"

    df = read_data(json_path, label_path)
    features = df.apply(extract_features, axis=1, result_type='expand')
    df_features = pd.concat([df[['story_id', 'Y']], features], axis=1)

    df_features.to_csv("train_features.csv", index=False)
    print(df_features.head())
