import pandas as pd,json
from preprocess import remove_punctuations
stopwords=['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','a','an','the']
punc=''.join(['.',',',"'","'",":","?",'!','@','/','&'])
def read_data(input,labels):
    with open(input) as f:
        data=f.readlines()
    with open(labels)as f:
        labels=f.readlines()
    
    datap=[]
    for i in data:
        datap.append(json.loads(i))
    d=pd.DataFrame(datap)
    d.reset_index()
    d['Index']=d.index
    if len(labels)==0:
        labels=['1' for i in range(len(d))]
    y=[i.strip()for i in labels]
    d['Y']=y
    d1=d[['Index','story_id', 'obs1', 'obs2', 'hyp1', 'Y']].rename(columns={'hyp1':'hyp'})
    d2=d[['Index','story_id', 'obs1', 'obs2', 'hyp2', 'Y']].rename(columns={'hyp2':'hyp'})
    d1['ord']=1
    d2['ord']=2
    
    df=pd.concat([d1,d2],ignore_index=True).sort_values(['Index','ord'])
    df['temp']=df.apply(lambda row: 1 if int(row['Y'])==int(row['ord']) else -1, axis=1)
    df['Y']=df['temp']
    df.drop(['temp'],axis=1, inplace=True)
    
    #clean text
    df['obs1'] = df['obs1'].apply(lambda x: remove_punctuations(x, punc))
    df['obs2'] = df['obs2'].apply(lambda x: remove_punctuations(x, punc))
    df['hyp']  = df['hyp'].apply(lambda x: remove_punctuations(x, punc))

    df['cleaned_text'] = df['obs1'] + " " + df['obs2'] + " " + df['hyp']

    return df

