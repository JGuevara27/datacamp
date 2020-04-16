# Natural Language Processing Fundamentals in Python
## Katharine Jarmul
## Hugo Bowne-Anderson
## Yashas Roy

# Regular Expressions & Word Tokenization
- Massive Field of study focused on making sense of language
- Going over basics NLP:
  1. Topic Identification
  2. Text Classification
- Some topics are:
  * Chatbots
  * Translation
  * Sentiment Analysis
  * + more.
- **Regular Expressions** are strings with special context.
- They allow us to match patterns in other strings.
- You can use them to:
  * Find links in a webpage.
  * Parse email addresses.
  * Remove/Replace unwanted characters.
- Python will be using the library *re*.
- We can match a substring using the `re.match()` function; matches pattern with a string.
- This does **not** return a result, but instead returns a *match object*.
- There are also special patterns that are accepted.
- `\w+` will match a word.
- `\d` will match numbers.
- `\s` will match a space.
- `.*` is a wildcard.
- `+` or `*` cause the match to be *greedy* or match as many characters as it can.
- If you capitalize the letters, then it negates the argument.
- You can also create a group of characters using `[asdlkjflk]`.
- The function `split()` will split a string on the match.
- The function `findall()` will match all patterns.
- The function `search()` will return the index of the match.
- You always pass the pattern first, and then the string.
- Make sure your pattern starts with `r""` first.
- **Tokenization** is when you turn a string document into *tokens*.
- There are many different theories and rules around it.
- A commonly used library for this is *nltk*.
- Tokenizing text can:
  * make it easier to map out parts of speech.
  * Match common words.
  * Remove unwanted tokens.
- The function `sent_tokenize()` will tokenize a document into sentences.
- The function `regexp_tokenize()` will tokenize a string or document based on a regex pattern.
- The function `TweetTokenizer` will allow you to token tweets into separate hashtags, mentions, etc.
- The difference between `match()` and `search()` is that match always starts at the beginning while search doesn't.
- You can print the starting and ending indexes of searches/matches using `x.start()` and `x.end()`.
- The or symbol `|` is very useful.
- You can define a group using `()`.
- The pattern `('(\d+|\w+))'` will match all words and digits but ignore punctuation.
- Groups mean explicitly *only* match what is in between.
- The backslash is called the *escape character*.
- An example to remember is `all_tokens = [tknzr.tokenize(t) for t in tweets]`.
- Unicode looks like : `'\U0001F300-\U0001F5FF'`.
- The library `matplotlib` is used by most everyone for Python.
- It is normal to write the import as `from matplotlib import pyplot as plt`.
- You can pass a small array to the `plt.hist()` function to draw a histogram.
- Then, you use `plt.show()` to have it plot the graph.
- You can substitute characters or patterns using `re.sub()`.

# Simple Topic Identification
- **Bag-of-Words** is a basic method for finding topics in a text.
- First tokenize, then count the tokens.
- We'lll be using `from collections import Counter`.
- The *Counter Object* has a similar structure to a dictionary.
- The object also has a method called `.most_common(<n>)`; n for return that number of results.
- **Text Preprocessing** is to help make better input data.
- Other techniques are:
  * Lemmatization
  * Stemming
  * Removing stop words.
- To handle stop words `from nltk.corpus import stopwords`.
- To use it, you'd use `if t not in stopwords.words('english')`.
- You use `x.isalpha()` will tell you if all characters are from the alphabet.
```python
# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
```
- We will be using `gensim` to process next.
- The library is open source and can be used by anyone.
- It uses top academic models to perform complex tasks.
- A **Word Vector** is a multidimensional representation of a word.
- **LDA Visualization**; where LDA stands for **Latent Dirichlet Allocation**
- To import and use it, you'd `from gensim.corpora.dictionary import Dictionary`.
- Pass the tokens into the funtion call: `dictionary = Dictionary(tokenized_docs)`.
- You can see a list of tokens and their ids using `dictionary.token2id`.
- Then, you create a **corpus** using `corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]`.
- And, this model can be easily saved, updated and reused.
- Since it's treated as a dictionary, you can use `computer_id = dictionary.token2id.get("computer")`.
- **Term-Frequency - Inverse Document Frequency** or *tf-idf* which assists us in understanding the most important words.
- The assumption in the model is that the corpus might have more shared words than just the stopwords.
- These common words are *down-weighted* due to how common they are.
- You can build a model using the previous corpus.
- First, you will need to import `from gensim.models.tfidfmodel import TfidfModel`.
- Then, pass the corpus into the function: `tfidf = TfidfModel(corpus)`.
- You can calculate the weights using `tfidf_weights = tfidf[ doc ]`.
```python
# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
```

# Named-Entity Recognition
- **Named-Entity Recognition** is used to identify important named entities in text.
- There is also the **Stanford CoreNLP Library** we can use.
- To use it, you'll need to install and setup Java first.
- You can tag the tokens using `nltk.pos_tag( tokens )`.
- This will categorize the different parts of speech.
- Then, after tagging, you pass those into the function `nltk.ne_chunk()`.
- You can also apply this to sentences using `nltk.ne_chunk_sents( pos_sentences )`.
```python
# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)
```
- **SpaCy** is an NLP library similar to gensim but with different implementation.
- It has a focus on creating pipelines to generate models and corpora.
- To import it, use `import spacy`.
- Make sure to load `nlp = spacy.load('en')`.
- To create a new one, you pass the string into the `nlp()` instance.
```python
# Create a new document: doc
doc = nlp( article )
```
- To get what spacy thinks are the important entities, use `nlp.ents`.
- You can access the entity as well as the label: `doc.ents[0], doc.ents[0].label`.
```python
# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
```
- **polyglot** is another library for making word vectors.
- The advantage of this package is that it supports lots of languages: 130+ .
- You can use it to swap one set of characters from a language into another.
- You import it using `from polyglot.text import Text`.
- To access the entities, you use `ptext.entities`.
- To print the list of all entities, use `txt.entities`
- You can access the tag using `txt.tag`.
-

# Building a "Fake News" Classifier
- Supervised learning is a form of machine learning.
- Attributes:
  * Problem has predefined training data.
  * Data has a label/outcome.
- The goal is to classify to the outcome correctly.
- To do this, we're going to be using the library scikit-learn.
- Steps:
  1. Collect and preprocess data.
  2. Determine a label.
  3. Split data into testing/training.
  4. Extract features to help predict label.
  5. Evaluate model on test data.
- You can explore the dataframe using `.head()`.
- You can get the attributes using `count_train.A`
- You can get the features using `count_vectorizer.get_feature_names()`
- You can access the columns using `count_df.columns`
- You can check for equality between DataFrames using `.equals()`.
- **Naive Bayes** is used a lot in NLP.
- It's not always the most effective, but it simple and usually works.
- To import it, use `from sklearn.naive_bayes import MultinominalNB`.
- Initialize class `nb_classifier = MultinominalNB()`.
- Build model: `nb_classifier.fit( count_train, y_train)`.
- To then make predictions, we use `nb_classifier.predict( count_test )`.
- You can import metrics library with `from sklearn import MultinomialNB`.
- We test the acccuracy using `metrics.accuracy_score(y_test, pred)`.
- You can also print out a Confusion Matrix using `metrics.confusion_matrix( y_test, pred, labels=[0,1])`.
- Sentiment Analysis is the next hard problem in Language Processing.
- You can access the classes in the model you've built using `nb_classifier.classes_`.
- You can access the model coefficients using `nb_classifier.coef_[0]`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklean.feature_extraction.text import CountVectorizer

# ..

y = df ['Outcome']

x_train, x_text, y_train, y_test = train_test_split(
    df['plot'],
    y,
    text_size=.33,
    random_state=53
)

count_vectorizer = CountVectorizer( stop_words='english')
count_train = count_vectorizer.fit_transform( x_train.values)
count_test = count_vectorizer.transform( x_test.values )
```

# Research:
- `defaultdict()`?
- `itertools.chain.from_iterable()`
- `sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)``
- The Stanford CoreNLP Library?
- `chunked_sentences = nltk.ne_chunk_sents( pos_sentences, binary=True )`
- SpaCy Entity Recognition Visualizer?

# Reference:
