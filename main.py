#Python Libraries Used; Spacy and Pandas are Open Source
#Addison Reminder: CITATION NEEDED FOR ALL MODULES + TRAINING DATA SETS
import spacy
import pandas as pd
import gensim
import re
from collections import Counter

#Load Core Libaries
nlpen = spacy.load("en_core_web_trf")
nlpes = spacy.load("es_dep_news_trf")

#Load English/Spanish Data
eng = spacy.lang.en.English()
esp = spacy.lang.es.Spanish()

#Preprocessing Function: Turns Pandas DataFrame Object into a String for NLP
def d2s(df):
    return df.to_string(columns=None, buf=None, index=False, na_rep="NaN")

def frequency(txt):
    nouns = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN")]
    adjectives = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "ADJ")]
    verbs = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "VERB")]
    
    #The Bottom Print Section is for Beta Testing; Will be Replaced with an Export-To-CSV Function
    print("Most Common Nouns")
    print(Counter(nouns).most_common(20))
    print("Most Common Adjectives")
    print(Counter(adjectives).most_common(20))
    print("Most Common Verbs")
    print(Counter(verbs).most_common(20))

rawdata = nlpes(re.sub(" +", " ", d2s(pd.read_csv("INSERT DATA HERE", usecols=['INSERT COLS HERE'])).replace("\n"," ").replace("NaN", "")))
