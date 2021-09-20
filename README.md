# Predicting The Sentiment of Financial new articles

In the finance field, the stock market and its trends are highly volatile. It attracts researchers to capture the volatility and predicting its next moves. Investors and market analysts study the market behavior and plan their buy or sell strategies accordingly. 

As the stock market  produces a large amount of data every day, it isn't straightforward for an individual to consider all the current and past information for predicting the future trend of a stock.

This project attempts to build a model that predicts news polarity that may affect changes in stock trends. In other words, check the impact of news articles on stock prices by using supervised machine learning as classification and other text mining techniques to check news polarity.


# Technologies

### Languages

Project is created with Python 3.6.9.

### Dependencies

-   [NumPy](https://numpy.org/)
-   [Matplotlib](https://matplotlib.org/)
-   [pandas](https://pandas.pydata.org/)
-   [Wordcloud](https://github.com/amueller/word_cloud)
-   [NLTK](https://www.nltk.org/)
-   [re](https://docs.python.org/3/library/re.html)
-   [collections](https://docs.python.org/2/library/collections.html)
[Text-classification-flow-based-on-improved-TF-IDF-text-representation-method-5_Q320.jpg (320×320) (researchgate.net)]
## Methodology
![Methodology](https://www.researchgate.net/publication/340734048/figure/fig2/AS:881690708832256@1587222854072/Text-classification-flow-based-on-improved-TF-IDF-text-representation-method-5_Q320.jpg)
We use different feature sets and machine learning classifiers to determine the best combination for sentiment analysis of twitter. We also experiment with various pre-processing steps like - punctuations, emoticons,specific terms and stemming. We investigated the following features - unigrams, bigrams, trigrams and negation detection. We finally train our classifier using various machine-learning algorithms - Naive Bayes, Decision Trees and Maximum Entropy.

### Pre Processing

User-generated content on the web is seldom present in a form usable for learning. It becomes important to normalize the text by applying a series of pre-processing steps. We have applied an extensive set of pre-processing steps to decrease the size of the feature set to make it suitable for learning algorithms. 

Although not all Punctuations are important from the point of view of classification but some of these, like question mark, exclamation mark can also provide information about the sentiments of the text. 

All stemming algorithms are of the following major types – affix removing, statistical and mixed. The first kind, Affix removal stemmer, is the most basic one. These apply a set of transformation rules to each word in an attempt to cut off commonly known prefixes and / or suffixes [8]. A trivial stemming algorithm would be to truncate words at N-th symbol. But this obviously is not well suited for practical purposes.

Lemmatization is the process of normalizing a word rather than just finding its stem. In the process, a suffix may not only be removed, but may also be substituted with a different one. It may also involve first determining the part-of-speech for a word and then applying normalization rules. It might also involve dictionary look-up. For example, verb ‘saw’ would be lemmatized to ‘see’ and the noun ‘saw’ will remain ‘saw’. For our purpose of classifying text, stemming should suffice.

#### Term frequency

It increases the weight of the terms (words) that occur more frequently in the document. It can be defined as tf(t,d) = F(t,d) where F(t,d) is number of occurrences of term ‘t’ in document ‘d’. But practically, it seems unlikely that thirty occurrences of a term in a document truly carry thirty times the significance of a single occurrence. So, in order to make it more pragmatic, we scale tf in logarithmic way so that as the frequency of terms increases exponentially, we will be increasing the weights of terms in additive manner.

      tf(t,d) = log(F(t,d))
      
  ####              Inverse document frequency

It diminishes the weight of the terms that occur in all the documents of corpus and similarly increases the weight of the terms that occur in rare documents across the corpus. Basically, the rare keywords get special treatment and stop words/non-distinguishing words get punishment. We define idf as:

      idf(t,D) = log(N/Nt ∈ d)

Here, ‘N’ is the total number of files in the corpus ‘D’ and ‘Nt ∈ d‘ is number of files in which term ‘t’ is present. By now, we can agree to the fact that tf is a intra-document factor which depends on individual document and idf is a per corpus factor which is constant for a corpus. 
Finally, We calculate tf-idf as:

      tf-idf(t,d,D) = tf(t,d) . idf(t,D)


### Input/Output ScreenShot
![Methodology](https://raw.githubusercontent.com/pranshu1229/sentiment/main/Screenshot.png)