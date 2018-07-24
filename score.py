import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

SHUFFLE = 5
TRAIN_PERCT = [0.1 * t for t in range(1, 10)]


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
    return new_samples


def show_results(optimizer, micro, macro, train_percentage):
    n_shuffle = micro.shape[1]
    print('Results')
    print('-------------------------------------')
    print('Setting:')
    print('LAMBDA:      %.2f' % optimizer.lam)
    print('ETA:         %.2f' % optimizer.eta)
    print('MAX_ITER:    %d' % optimizer.max_iter)
    print('EPSILON:     %.1e' % optimizer.eps)
    print('GROUPING:    %s' % optimizer.grouping_strategy)
    print('K SAMPLING:  %s' % optimizer.sample_strategy)
    print('-------------------------------------')
    for i, percentage in enumerate(train_percentage):
        print('Train percentage:  %.2f' % (1 - percentage))
        print('Iter           Micro       Macro')
        print('-------------------------------------')
        for ite in range(n_shuffle):
            print('Shuffle #%d:   %.4f      %.4f'
                  % (ite + 1, micro[i, ite], macro[i, ite]))
        print('-------------------------------------')
        print('Average:      %.4f      %.4f'
              % (micro.mean(1)[i], macro.mean(1)[i]))


def multi_class_classification(optimizer, sample_filename, cv=None,
                               train_percentage=TRAIN_PERCT, n_shuffle=SHUFFLE):
    graph = optimizer.graph
    samples = read_pairs(sample_filename)
    samples = sample_id_mapping(samples, graph.vid2newVid_mapping)

    x_matrix = np.stack(optimizer.embed([t[0] for t in samples]))
    y_list = [t[1] for t in samples]

    if cv is None:
        micro_results = np.zeros((len(train_percentage), n_shuffle))
        macro_results = np.zeros((len(train_percentage), n_shuffle))

        for i, percentage in enumerate(train_percentage):
            for ite in range(n_shuffle):
                x_train, x_test, y_train, y_test = train_test_split(
                    x_matrix, y_list,
                    test_size=percentage)
                model = OneVsRestClassifier(SVC())
                model.fit(x_train, y_train)
                pred = model.predict(x_test)

                micro_results[i, ite] = f1_score(y_test, pred, average='micro')
                macro_results[i, ite] = f1_score(y_test, pred, average='macro')

        show_results(optimizer, micro_results, macro_results, train_percentage)

