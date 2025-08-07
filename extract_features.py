from read_data import read_data
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

def joh_bins(jaccard_obs_hyp):
    if jaccard_obs_hyp<0.06:
        return 1,0,0,0  
    elif jaccard_obs_hyp<0.1:
        return 0,1,0,0
    elif jaccard_obs_hyp<0.16:
        return 0,0,1,0
    else:
        return 0,0,0,1


def word_overlap(set1, set2):
    return len(set1 & set2)

def sentiment_score(tokens):
    pos = len([w for w in tokens if w in POSITIVE_WORDS])
    neg = len([w for w in tokens if w in NEGATIVE_WORDS])
    return pos - neg

def contains_negation(tokens):
    return any(w in NEGATIONS for w in tokens)

def get_overlap_bin(overlap):
    if overlap == 1:
        return 1,0,0,0,0,0
    elif overlap == 2:
        return 0,1,0,0,0,0
    elif overlap ==3:
        return 0,0,1,0,0,0
    elif overlap<=5:
        return 0,0,0,1,0,0
    elif overlap<=10:
        return 0,0,0,0,1,0
    else:
        return 0,0,0,0,0,1
    

def get_sentiment_bin(score):
    if score < 0:
        return 0,0,1
    elif score == 0:
        return 0,1,0
    else:
        return 1,0,0

def extract_features(row):
    obs1_tokens = tokenize(row['obs1'])
    obs2_tokens = tokenize(row['obs2'])
    hyp_tokens = tokenize(row['hyp'])
    obs1_set = set(tokenize(row['obs1']))
    obs2_set = set(tokenize(row['obs2']))
    hyp_set = set(tokenize(row['hyp']))

    all_obs_set = obs1_set | obs2_set

    jaccard_obs_hyp = jaccard_similarity(all_obs_set, hyp_set)
    joh1,joh2,joh3,joh4=joh_bins(jaccard_obs_hyp) #get bins for jaccard similarity of obs and hyp
    jaccard_obs1_hyp = jaccard_similarity(obs1_set, hyp_set)
    jo1h1,jo1h2,jo1h3,jo1h4=joh_bins(jaccard_obs1_hyp)#get bins for jaccard similarity of obs1 and hyp
    jaccard_obs2_hyp = jaccard_similarity(obs2_set, hyp_set)
    jo2h1,jo2h2,jo2h3,jo2h4=joh_bins(jaccard_obs2_hyp)#get bins for jaccard similarity of obs2 and hyp
    
    overlap_all = word_overlap(all_obs_set, hyp_set)
    oa1,oa2,oa3,oa5,oa10,oa15=get_overlap_bin(overlap_all) # get bins for word overlap btw obs_all and hyp
    overlap_obs1 = word_overlap(obs1_set, hyp_set)
    oobs1_1,oobs1_2,oobs1_3,oobs1_5,oobs1_10,oobs1_15=get_overlap_bin(overlap_obs1) # get bins for word overlap btw obs1 and hyp
    overlap_obs2 = word_overlap(obs2_set, hyp_set)
    oobs2_1,oobs2_2,oobs2_3,oobs2_5,oobs2_10,oobs2_15=get_overlap_bin(overlap_obs2)# get bins for word overlap btw obs2 and hyp
    sentiment_obs = sentiment_score(list(all_obs_set))
    sent_obs_pos,sent_obs_neutral,sent_obs_neg=get_sentiment_bin(sentiment_obs) # get bins for sentiment of obs
    sentiment_obs1=sentiment_score(list(obs1_tokens))
    sent_obs1_pos,sent_obs1_neutral,sent_obs1_neg=get_sentiment_bin(sentiment_obs1)# get bins for sentiment of obs1
    sentiment_obs2=sentiment_score(list(obs2_tokens))
    sent_obs2_pos,sent_obs2_neutral,sent_obs2_neg=get_sentiment_bin(sentiment_obs2)# get bins for sentiment of obs2
    sentiment_hyp = sentiment_score(list(hyp_tokens))
    sent_hyp_pos,sent_hyp_neutral,sent_hyp_neg=get_sentiment_bin(sentiment_hyp)# get bins for sentiment of hyp

    negation_obs = contains_negation(all_obs_set)
    negation_obs1=contains_negation(obs1_set)
    negation_obs2=contains_negation(obs2_set)
    negation_hyp = contains_negation(hyp_set)

    features = [jaccard_obs1_hyp,jo1h1,jo1h2,jo1h3,jo1h4,                       # jaccard obs1 and bins 4               total 
                jaccard_obs2_hyp,jo2h1,jo2h2,jo2h3,jo2h4,                       # jaccard obs2 and bins 4                    
                jaccard_obs_hyp,joh1,joh2,joh3,joh4,                            # jaccard obs_all and bins 4               15
                overlap_all,oa1,oa2,oa3,oa5,oa10,oa15,                          # overlap obs_all with hyp and bins 6
                overlap_obs1,oobs1_1,oobs1_2,oobs1_3,oobs1_5,oobs1_10,oobs1_15, # overlap obs_1 with hyp and bins 6
                overlap_obs2,oobs2_1,oobs2_2,oobs2_3,oobs2_5,oobs2_10,oobs2_15, # overlap obs_2 with hyp and bins 6         21
                sentiment_obs,sent_obs_pos,sent_obs_neutral,sent_obs_neg,       # sentiment of obs_all and bins 3
                sentiment_obs1,sent_obs1_pos,sent_obs1_neutral,sent_obs1_neg,   # sentiment of obs_1 and bins 3
                sentiment_obs2,sent_obs2_pos,sent_obs2_neutral,sent_obs2_neg,   # sentiment of obs_2 and bins 3
                sentiment_hyp,sent_hyp_pos,sent_hyp_neutral,sent_hyp_neg,       # sentiment of hyp and bins 3               16
                int(negation_obs),int(negation_obs1),int(negation_obs2),int(negation_hyp)]# negation check of all texts 4
    '''uncomment below for dictionary'''
    # st='jaccard_obs1_hyp,jaccard_obs1_bin1,jaccard_obs1_bin2,jaccard_obs1_bin3,jaccard_obs1_bin4,' \
    # 'jaccard_obs2_hyp,jaccard_obs2_bin1,jaccard_obs2_bin2,jaccard_obs2_bin3,jaccard_obs2_bin4,'
    # 'jaccard_obs_hyp,jaccard_obs_bin1,jaccard_obs_bin2,jaccard_obs_bin3,jaccard_obs_bin4,'
    # 'overlap_all,overlap_all_bin1,overlap_all_bin2,overlap_all_bin3,overlap_all_bin4,overlap_all_bin5,overlap_all_bin6,'
    # 'overlap_obs1,overlap_obs1_bin1,overlap_obs1_bin2,overlap_obs1_bin3,overlap_obs1_bin4,overlap_obs1_bin5,overlap_obs1_bin6,'
    # 'overlap_obs2,overlap_obs2_bin1,overlap_obs2_bin2,overlap_obs2_bin3,overlap_obs2_bin4,overlap_obs2_bin5,overlap_obs2_bin6,'
    # 'sentiment_obs,sentiment_obs_positive,sentiment_obs_neutral,sentiment_obs_negative,'
    # 'sentiment_obs1,sentiment_obs1_positive,sentiment_obs1_neutral,sentiment_obs1_negative,'
    # 'sentiment_obs2,sentiment_obs2_positive,sentiment_obs2_neutral,sentiment_obs2_negative,'
    # 'sentiment_hyp,sentiment_hyp_positive,sentiment_hyp_neutral,sentiment_hyp_negative,'
    # 'negation_obs,negation_obs1,negation_obs2,negation_hyp'
    # st=st.split(',')
    # features_dict={}
    # for i,v in enumerate(features):
    #     features_dict[st[i]]=v
    
    return features #features_dict if you want to get dictionary


'''Code for testing script'''

# if __name__ == "__main__":
#     json_path = "train.json"
#     label_path = "train_labels.lst"

#     df = read_data(json_path, label_path)
#     features = pd.json_normalize(df.apply(extract_features, axis=1))
#     df_features = pd.concat([df[['story_id', 'Y']], features], axis=1)

#     df_features.to_excel("train_features.xlsx", index=False)
#     print(df_features.head())