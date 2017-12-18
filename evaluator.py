import csv
from summarizer import Summarizer
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


myList = []
with open('news_summary.csv','r',encoding='utf-8', errors='ignore') as csvfile:
    reader = csv.DictReader(csvfile)
    try:
        for row in reader:
            try:
                myList.append({'text':row['text'],'ctext':row['ctext']})
            except:
                continue
    except:
        for row in reader:
            try:
                myList.append({'text':row['text'],'ctext':row['ctext']})
            except:
                continue


text_rank_summarizer = Summarizer()
lsa_summarizer = Summarizer(method="lsa")

print('tr,lsa')
lsa_score = 0
tr_score = 0
ct = 0
for dat in myList:
    ct += 1
    tr = text_rank_summarizer.summarize(language='english', size=3, text=dat['ctext'])
    lsa = lsa_summarizer.summarize(language='english', size=3, text=dat['ctext'])
    sim_lsa = cosine_sim(dat['text'], lsa)
    sim_tr = cosine_sim(dat['text'], tr)
    tr_score += sim_tr
    lsa_score += sim_lsa
    print(str(sim_tr) + "," +  str(sim_lsa))

print()
print("average lsa : " + str(float(lsa_score/ct)))
print("average tr : " + str(float(tr_score/ct)))
