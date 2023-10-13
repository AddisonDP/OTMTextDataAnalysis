#Python Libraries Used; Spacy and Pandas are Open Source
#Addison Reminder: CITATION NEEDED FOR ALL MODULES + TRAINING DATA SETS
import spacy
import pandas as pd

#Load Core Libaries
nlpen = spacy.load("en_core_web_trf")
nlpes = spacy.load("es_dep_news_trf")

#Load English/Spanish Data
eng = spacy.lang.en.English()
esp = spacy.lang.es.Spanish()

#Preprocessing Function: Turns Pandas DataFrame Object into a String for NLP
def dataframetostring(df):
    return df.to_string(columns=None, buf=None, index=False, na_rep="NaN")

# Commented Currently for Functionality
# pand = pd.read_csv("[INSERT DATA LINK HERE]", usecols=[INSERT COLUMNS HERE])
# string = dataframetostring(pand)
