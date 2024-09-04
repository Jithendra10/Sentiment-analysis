from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')


app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('mdl.pkl', 'rb') as f:
    mdl = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)


def preprocessing(tweet): 
    tweet = re.sub(r'[^a-zA-Z]',' ',tweet)    
    tweet = tweet.lower()
    tweet = [word for word in tweet.split(' ') if not word in stopwords.words('english')]
    tweet = [stemmer.stem(word) for word in tweet]
    tweet = [word for word in tweet if len(word) != 0]
    tweet = ' '.join(tweet)
    return tweet

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = mdl.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)