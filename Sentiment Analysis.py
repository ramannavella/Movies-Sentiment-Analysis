import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
def word_features(words):
return dict([(word, True) for word in words])
print("movie review categories :",movie_reviews.categories())
negativeids = movie_reviews.fileids('neg')
positiveids = movie_reviews.fileids('pos')
neg_features = [(word_features(movie_reviews.words(fileids=[f])), 'neg') for f in negativeids]
pos_features = [(word_features(movie_reviews.words(fileids=[f])), 'pos') for f in positiveids]
negcutoff = len(neg_features)*3/4
negcutoff=int(negcutoff)
poscutoff = len(pos_features)*3/4
poscutoff=int(poscutoff)
trainfeatures = neg_features[:negcutoff] + pos_features[:poscutoff]
testfeatures = neg_features[negcutoff:] + pos_features[poscutoff:]
print ('train on %d instances' % len(trainfeatures))
print('test on %d instances' % len(testfeatures))
print("\n")
#Using Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(trainfeatures)
accuracy1= nltk.classify.util.accuracy(classifier, testfeatures)
print("accuracy :",format(accuracy1*100,'.2f'),"%");
print("\n")
#most informative features
classifier.show_most_informative_features(100)
print("\n")
#Checking individual word classification
print("Magnificient word classification:", classifier.classify(word_features(['magnificient'])))
print("Ludicrous word classification:", classifier.classify(word_features(['ludicrous'])))
print("\n")

24

#Test for own sentence
test_sentence = "Waste movie"
test_sentence_features = word_features(test_sentence)
print('opinion:',test_sentence)
print('review:',classifier.classify(test_sentence_features))
print("\n")
test_sentence = "I loved every part of the movie."
test_sentence_features = word_features(test_sentence)
print('opinion:',test_sentence)
print('review:',classifier.classify(test_sentence_features))
print("\n")
test_sentence = "It was stellar and the performance was spectacular."
test_sentence_features = word_features(test_sentence)
print('opinion:',test_sentence)
print('review:',classifier.classify(test_sentence_features))
print("\n")
test_sentence = "The acting was pale and the plot looked copied."
test_sentence_features = word_features(test_sentence)
print('opinion:',test_sentence)
print('review:',classifier.classify(test_sentence_features))
print("\n")
test_sentence = "The movie was awesome."
test_sentence_features = word_features(test_sentence)
print('opinion:',test_sentence)
print('review:',classifier.classify(test_sentence_features))
print("\n")
test_sentence = "The movie was so boring I almost walked out of the theater"
test_sentence_features = word_features(test_sentence)
print('opinion:',test_sentence)
print('review:',classifier.classify(test_sentence_features))
print("\n")
test_sentence = input("Enter opinion: ")
test_sentence_features = word_features(test_sentence)
print('review:',classifier.classify(test_sentence_features))