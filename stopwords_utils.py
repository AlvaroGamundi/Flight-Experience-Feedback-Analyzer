import nltk
import pandas as pd
import re
from nltk.corpus import stopwords as nltk_stopwords

nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text

def get_clean_stopwords():
    stopwords_list = nltk_stopwords.words('english')
    stopwords_df = pd.DataFrame(stopwords_list, columns=['stopword'])
    stopwords_df['limpias'] = stopwords_df['stopword'].transform(preprocess_text)
    stopwords_df = stopwords_df.drop(columns=['stopword']).drop_duplicates()
    return stopwords_df
