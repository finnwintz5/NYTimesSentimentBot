import tweepy
from keep_alive import keep_alive
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
import re, string,random
from bs4 import BeautifulSoup
import requests
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from math import ceil
from datetime import date


# *********************************************************
# --------------------HELPERS FUNCTIONS--------------------
# *********************************************************

# ---------------------TWEEPY HELPERS---------------------
def getClient():
    client = tweepy.Client(bearer_token=BEARER_TOKEN,
                           consumer_key=API_KEY,
                           consumer_secret=API_KEY_SECRET,
                           access_token=ACCESS_TOKEN,
                           access_token_secret=ACCESS_TOKEN_SECRET)
    return client

# ---------------------BAYES ANALYSIS HELPERS---------------------
def remove_noise(tweet_tokens,stop_words=()):
    cleaned_tokens=[]
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer=WordNetLemmatizer()
        token=lemmatizer.lemmatize(token,pos)
        if len(token)>0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
# GETTING ALL WORDS
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# PREP FOR BAYES ANALYSIS
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token,True] for token in tweet_tokens)

# ---------------------WEB SCRAPING HELPER---------------------
def scrape_nytimes():
    url="https://www.nytimes.com/"
    web_stuff = requests.get(url)
    html = web_stuff.text
    soup = BeautifulSoup(html, "html.parser")
    nytimes_content = soup.find_all("h3", class_="indicate-hover")
    all_entries=''
    for entry in nytimes_content:
        html_block = str(entry)
        temp_entry = html_block.split('>', 1)
        headline = temp_entry[1].split('</h3')[0]
        all_entries+=", "
        all_entries+=headline
    return all_entries

# ---------------------OTHER HELPERS---------------------
def get_score_graphic(neg,neu,pos):
    result=""
    pos_num=int(ceil(pos*10))
    neg_num=int(ceil(neg*10))
    neu_num=10-pos_num-neg_num
    for i in range(pos_num):
        result+="🟩"
    for i in range(neu_num):
        result+="⬜️"
    for i in range(neg_num):
        result+="🟥"
    return result




# *********************************************************
# ----------------------ONE TIME CODE----------------------
# *********************************************************

# ---------------------TWEEPY CREDENTIALS---------------------
# INITIALIZE ACCESS
API_KEY = 'ax1RHUVJZkOiO3HXwR9pr2flR'
API_KEY_SECRET = 'S8QCnidAXBLJRU6vRgrNzOcC9mmTpriIPDlAOnAHDqNTWAqyhP'

BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAO%2F0fAEAAAAAHp4vlANlpJRr1AAmB8Z6iuG9ddE%3DZIBDWk1OBFscdjxkyyaxMF0lx95fchSSRv4PrYodAbdHH9FbpL'

ACCESS_TOKEN = '1551716767071977474-62rKoifpF95wU32gJgH6d3gk7Vi4va'
ACCESS_TOKEN_SECRET = 'vfgMtSO3sg2tCmTZNtj3cinvOIZHwVB3bNEuxS51EF8XI'

client=getClient()

# ---------------------BAYES ANALYSIS PREP---------------------
# https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
# TOKENIZE WORDS FROM TWEET DATASET
positive_tweets_tokenized = twitter_samples.tokenized('positive_tweets.json')
negative_tweets_tokenized = twitter_samples.tokenized('negative_tweets.json')
stop_words=stopwords.words('english')

# CLEAN TOKENS BY REMOVING NOISE
positive_cleaned_tokenized_list=[remove_noise(tokens,stop_words) for tokens in positive_tweets_tokenized]
negative_cleaned_tokenized_list=[remove_noise(tokens,stop_words) for tokens in negative_tweets_tokenized]

# CONVERT TOKENS TO DICTIONARY
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokenized_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokenized_list)

# CREATE DATASET FOR ANALYSIS
positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]
dataset=positive_dataset+negative_dataset

# SPLIT DATASET TO TRAINING AND TESTING
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]
classifier = NaiveBayesClassifier.train(train_data)

# GET ACCURACY OF TRAINING
print("Accuracy of training is:", classify.accuracy(classifier, test_data))
# print(classifier.show_most_informative_features(10))




# **********************************************************
# ------------------------DAILY CODE------------------------
# **********************************************************
keep_alive()
time.sleep(31,959)
while True:
    # TOKENIZE NYTIMES
    nytimes_text=scrape_nytimes()
    nytimes_tokens=remove_noise(word_tokenize(nytimes_text))

    # BAYES: DETERMINE SENTIMENT OF NYTIMES
    naive_bayes_classifier=classifier.classify(dict([token,True] for token in nytimes_tokens))

    # SCORE: DETERMINE SCORE OF NYTIMES
    analyzer = SentimentIntensityAnalyzer()
    output=analyzer.polarity_scores(nytimes_text)
    score_graphic=get_score_graphic(output["neg"],output["neu"],output["pos"])

    if output["compound"]>=.05:
        sentiment_type_result="Positive"
    elif output["compound"] <=-.05:
        sentiment_type_result="Negative"
    else:
        sentiment_type_result="Neutral"
    
    sentiment_type_result+=" ("+str(output["compound"])+")"
    today=date.today()
    final_tweet="Today’s nytimes headlines have a "+sentiment_type_result+" sentiment.\n\n"+score_graphic+"\n\n"+"Naive Bayes classifier for " +today.strftime("%b-%d-%Y")+ ": "+naive_bayes_classifier
    client.create_tweet(text=final_tweet)
    time.sleep(86395)