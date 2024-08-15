#Python Libraries Used; Spacy and Pandas and Gensim are Open Source
#Addison Reminder: CITATION NEEDED FOR ALL MODULES + TRAINING DATA SETS
import spacy
import math
import pandas as pd
from collections import Counter
from collections import defaultdict
import string
import langid
import pyLDAvis
import pyLDAvis.gensim_models as gentopic
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess as preprocess
from gensim import corpora

#Main starting proccess; Only works on English and Spanish, but can be expanded.
if __name__ == '__main__':
    #Language Detection set languages it can choose from
    langid.set_languages(langs=['en', 'es'])

    #Load Core Libaries; Set Arrays for language seperation
    nlpen = spacy.load("en_core_web_trf")
    nlpes = spacy.load("es_dep_news_trf")
    engdata = []
    espdata = []
    #Error array in Language separation for testing
    error = []

    #Data import; Note: May need to be modified for various data situations; Update with regex at some point
    #data = pd.read_csv([ADD DATA HERE], dtype="str", encoding="utf-8").replace("\"", "")

    #Modify for alternate languages; 
    def langseparate(df):
        #df.shape[0,1] are the dataframe rows/columns
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                #if/else for null values
                if str(df.iat[i, j]) != "nan":
                    #langid Classification; Encoding is for the use of Spanish diacritics
                    match langid.classify(str(df.iat[i, j]))[0]:
                        case "en":
                            engdata.append(df.iat[i, j].encode('utf-8'))
                        case "es":
                            espdata.append(df.iat[i, j].encode('utf-8'))
                        case default:
                            error.append(df.iat[i, j].encode('utf-8'))
                else:
                    pass
        #Bug Testing; Determine what the bot is seeing
        print(engdata)
        print(espdata)
        print(error)

    def frequency(txt):
        #token (add nlp model); is_stop implies stop word (English/Spanish), token.pos_ = "Part of Speech", ent_iob_ is added for nouns for specific noun forms not recognized by "NOUN"
        nouns = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN") or (token.ent_iob_ == "B") or (token.ent_iob_ == "I")]
        adjectives = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "ADJ")]
        verbs = [token.text for token in txt if (not token.is_stop and not token.is_punct and token.pos_ == "VERB")]
        print("Most Common Nouns")
        print(Counter(nouns).most_common(20))
        print("Most Common Adjectives")
        print(Counter(adjectives).most_common(20))
        print("Most Common Verbs")
        print(Counter(verbs).most_common(20))

    def bigram(txt, lang):
        #Identify input lanuage
        match lang:
            case "en":
                #Cluster bigrams, find largest 20 in text
                nphrases = CountVectorizer(ngram_range=(2,2), max_features=20, stop_words=list(nlpen.Defaults.stop_words)).fit(txt)
                dataphrases = nphrases.transform(txt)
                frequency = sorted([(word, dataphrases.sum(axis=0)[0, i]) for word, i in nphrases.vocabulary_.items()], key = lambda x: x[1], reverse=True)
            case "es":
                nphrases = CountVectorizer(ngram_range=(2,2), max_features=20, stop_words=list(nlpes.Defaults.stop_words)).fit(txt)
                dataphrases = nphrases.transform(txt)
                frequency = sorted([(word, dataphrases.sum(axis=0)[0, i]) for word, i in nphrases.vocabulary_.items()], key = lambda x: x[1], reverse=True)
            case default:
                pass
        bigrams = pd.DataFrame(frequency)
        bigrams.columns = ["Bigrams", "Frequency"]
        print(bigrams)
    

    def topicmodel(array):
        #Find every word in formatted array that isn't a stopword and count frequencies
        preproc = [[item for item in entry.split() if entry not in nlpen.Defaults.stop_words and nlpes.Defaults.stop_words] for entry in array]
        freq = defaultdict(int)
        for item in preproc:
            for word in item:
                freq[word] += 1
        #Establish dictionary (total body of words to select from), and word2vec corpora
        dcorp = corpora.Dictionary(preproc)
        print(len(dcorp))
        wordvec = [dcorp.doc2bow(text) for text in preproc]
        #Prepare topic modeling scores; Lowest coherence score corresponds to the desired "optimal number of topics" value
        scores = []
        #Generate topics with at least 20 words per topic; Hence division by 20 is utilized for finding the ideal number of topics.
        for numtop in range(1, math.ceil(len(dcorp)/20)):
            #assess the viability of LDA at every 
            lda = LdaModel(wordvec, num_topics=numtop, id2word=dcorp)
            coherencemod = CoherenceModel(model=lda, corpus=wordvec, coherence='u_mass')
            coherence = coherencemod.get_coherence()
            scores.append(coherence)
            print(scores.index(min(scores)) + 1)
        #idlda = ideal lda; pyLDAvis is utilized for visualization.
        idlda = LdaModel(wordvec, num_topics=(scores.index(min(scores))+1), id2word=dcorp)
        finalview = gentopic.prepare(idlda, wordvec, dcorp)
        pyLDAvis.display(finalview)

    langseparate(data)

    espfinal = [x.decode('utf-8').lower() for x in espdata]
    engfinal = [x.decode('utf-8').lower() for x in engdata]

    #bigram(espfinal, "es")
    #bigram(engfinal, "en")

    #frequency(nlpes(" ".join(espfinal)))
    #frequency(nlpen(" ".join(engfinal)))

    topicmodel(engfinal)
