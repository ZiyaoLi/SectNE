import numpy as np
import random as rd
from six import iteritems
from collections import defaultdict

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

SHUFFLE = 5
TRAIN_PERCT = [0.1 * t for t in range(10)]


def score(model, label, Vset):
    # load labels
    labels = np.array([label[v] for v in Vset])
    # load embeddings
    if isinstance(model, dict):
        features = np.array([model[v] for v in Vset])
    else:
        features = np.array(model.embed(Vset))
    
    all_results = defaultdict(list)
    # train percent
    for percent in np.array(range(5, 6)) * 0.1 :
        for shuffle in range(SHUFFLE):
            # split
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=percent)
            model = OneVsRestClassifier(SVC())
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            result = {}
            averages = ['micro', 'macro']
            for average in averages :
                result[average] = f1_score(y_test, pred, average=average)
            all_results[percent].append(result)
    
    print('Results, using embeddings of dimensionality', features.shape[1])
    print('-------------------')
    for percent in sorted(all_results.keys()):
        print('Train percent:', percent)
        for index, result in enumerate(all_results[percent]):
            print('Shuffle #%d:   ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[percent])
        print('Average score:', dict(avg_score))
        print('-------------------')
        
        return dict(avg_score)


def read_pairs(filename, sep='\t'):
    pairs = []
    f = open(filename, 'r')
    s = f.readline()
    while len(s):
        pair = s.strip().split(sep)
        left = int(pair[0])
        right = int(pair[1])
        pairs.append((left, right))
        s = f.readline()
    return pairs


def sample_id_mapping(samples, id_map):
    new_samples = []
    for oldVid, label in samples:
        t = id_map[oldVid]
        if isinstance(t, int):
            new_samples.append((t, label))
    return id_map


def multi_class_classification(optimizer, sample_filename, cv=None,
                               train_percentage=TRAIN_PERCT, n_shuffle=SHUFFLE):
    graph = optimizer.graph
    samples = read_pairs(sample_filename)
    samples = sample_id_mapping(samples, graph.vid2newVid_mapping)

    x_matrix = np.stack(optimizer.embed([t[0] for t in samples]))
    y_list = [t[1] for t in samples]

    all_results = defaultdict(list)

    if cv is None:
        for percentage in train_percentage:
            for ite in range(n_shuffle):
                x_train, x_test, y_train, y_test = train_test_split(
                    x_matrix, y_list,
                    test_size=percentage)
                model = OneVsRestClassifier(SVC())
                model.fit(x_train, y_train)
                pred = model.predict(x_test)

                result = {}
                averages = ['micro', 'macro']
                for average in averages:
                    result[average] = f1_score(y_test, pred, average=average)
                all_results[percent].append(result)



