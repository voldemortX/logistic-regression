# logistic-regression
--voldemort's logistic regression

--tested on my son francis's homework assignment

--20000 half-positive-half-negative movie reviews classification

--this is a very simple model without regularizations


# how to use this model?
1. use tests.py to call calculateWordList().
2. use tests.py to call reviewModel() with pre().
3. use tests.py to call reviewModel() with fastPre() if you wish to
tune some hyperparameters regarding the cross-validation set.
4. repeat 1-3 if you changed your data.
5. add your own code if you wish to track exactly which reviews you classified wrong.

--this exact model should give an accuracy around 92%.


# detailed description
1. this model uses text cleaning provided by packet re.
2. this model uses word stemming provided by packet nltk.
3. this model then eliminates words(features) that appears less than 0.25% after stemming.
4. this model thus gives value 1 to a feature if that word exists and gives 0 otherwise.
5. divides the data set to 3 parts: 60% train; 20% cross-validation; 20% test
6. this model finally uses numpy to perform a fully vectorized implementation of 
batch gradient descent on the training set to fit a simple logistic regression model.
