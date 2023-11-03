#Python Libraries Used; Spacy and Pandas and Gensim are Open Source
#Addison Reminder: CITATION NEEDED FOR ALL MODULES + TRAINING DATA SETS
import spacy
import pandas as pd
import gensim
import re
from collections import Counter
import spacy_fastlang
import plot-likert
import wordcloud
import matplotlib.pyplot as plt

#Load Core Libaries
nlpen = spacy.load("en_core_web_trf")
nlpes = spacy.load("es_dep_news_trf")

#Load English/Spanish Data
eng = spacy.lang.en.English()
esp = spacy.lang.es.Spanish()

engdata = []
espdata = []

#Preprocessing Functions: Turns Pandas DataFrame Object into a String for NLP
def d2s(txt):
    string = " ".join(txt)
    return string

# pand = re.sub(" +", " ", d2s(pd.read_csv("[CSVDATA]", usecols=[COLUMNS])).replace("\n"," ").replace("NaN", "")).replace("COLUMNS", "")
# wcwc = nlpes(pand)

def langseparate(df):
    nlpen.add_pipe("language_detector", name='language_detector', last=True)
    nlpes.add_pipe("language_detector", name='language_detector', last=True)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if nlpen(df.iat[i, j])._.language == "en":
                engdata.append(df.iat[i, j])
                print(engdata)
            else:
                if nlpes(df.iat[i, j])._.language == "es":
                    espdata.append(df.iat[i, j])
                    print(espdata)
                else:
                    print("Error")
    print("English Data")
    print(engdata)
    print("Spanish Data")
    print(espdata)

def frequency(txt):
    nouns = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN") or (token.ent_iob_ == "B") or (token.ent_iob_ == "I")]
    adjectives = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "ADJ")]
    verbs = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "VERB")]
    nphrases = [chunk.text for chunk in txt.noun_chunks]
    print("Most Common Nouns")
    print(Counter(nouns).most_common(20))
    print("Most Common Adjectives")
    print(Counter(adjectives).most_common(20))
    print("Most Common Verbs")
    print(Counter(verbs).most_common(20))
    print("Commonly Clustered Terms (Noun Phrases)")
    print(Counter(nphrases).most_common(freq))
    usefulwords = nouns + adjectives + verbs
    return usefulwords

def likert(df, scale, width, height):
    picture = picture = plot_likert.plot_likert(df, scale, figsize=(width, height))
    picture.get_figure().savefig('likertplot.png', dpi=200, bbox_inches='tight')

def wordscloud(txt):
    wc = wordcloud.WordCloud(max_font_size=80).generate(" ".join(frequency(txt, 20)))
    image = wc.to_image()
    image.show()
