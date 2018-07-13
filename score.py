# from deepwalk

import numpy
import sys

from collections import defaultdict
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy import sparse
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def score (embeddings, group, V) :
    # Load labels
    label = [group[i] for i in V] #
    labels_matrix = sparse.csc_matrix((numpy.ones_like(label),(V,label))) #
    labels_count = labels_matrix.shape[1] #
    mlb = MultiLabelBinarizer(range(labels_count)) #

    # Load Embeddings
    features_matrix = numpy.asarray([embeddings[i] for i in V]) #
    #features_matrix = preprocessing.normalize(features_matrix)

    # Shuffle, to create train/test groups
    shuffles = []
    for x in range(5):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # to score each train/test group
    all_results = defaultdict(list)

    #training_percents = numpy.asarray(range(1, 10)) * .1
    training_percents = numpy.asarray(range(5,6)) * .1


    for train_percent in training_percents:
        for shuf in shuffles:
            X, y = shuf
            training_size = int(train_percent * X.shape[0])
            X_train = X[:training_size, :]
            y_train_ = y[:training_size]
            y_train = [[] for x in range(y_train_.shape[0])]
            cy =  y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_train[i].append(j)
            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]
            y_test = [[] for _ in range(y_test_.shape[0])]
            cy =  y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)
            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
            all_results[train_percent].append(results)

    print('Results, using embeddings of dimensionality', X.shape[1])
    print('-------------------')
    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        for index, result in enumerate(all_results[train_percent]):
            print('Shuffle #%d:   ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print('Average score:', dict(avg_score))
        print('-------------------')
        #return dict(avg_score)