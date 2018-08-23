import numpy as np
import random
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier

SHUFFLE = 5
TEST_PERCT = [0.1 * t for t in range(1, 10)]
CROSS_VAL_FOLD = [2, 3, 5, 8, 10]


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


def read_multi_labels(filename, sep='\t'):
    labels = []
    f = open(filename, 'r')
    s = f.readline()
    while len(s):
        pair = s.strip().split(sep)
        vid = int(pair[0])
        lab = [int(t) for t in pair[1:]]
        labels.append((vid, lab))
        s = f.readline()
    return labels


def read_embeddings(filename, sep=' '):
    embeddings = {}
    f = open(filename, 'r')
    f.readline()  # read the first line
    s = f.readline()
    while len(s):
        embed = s.strip().split(sep)
        idx = int(embed[0])
        vector = np.array([float(t) for t in embed[1:]])
        embeddings[idx] = vector
        s = f.readline()
    return embeddings


def sample_id_mapping(samples, id_map):
    new_samples = []
    for oldVid, label in samples:
        t = id_map[oldVid]
        if isinstance(t, int):
            new_samples.append((t, label))
    return new_samples


def show_results_shuffle(optimizer, micro, macro, train_percentage):
    n_shuffle = micro.shape[1]
    print('Results')
    print('-------------------------------------')
    if optimizer is not None:
        print('Setting:')
        print('K SIZE:      %d' % optimizer.k_size)
        print('DIMENSION:   %d' % optimizer.dim)
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


def show_results_cv(optimizer, micro, macro):
    print('Results')
    print('-------------------------------------')
    if optimizer is not None:
        print('Setting:')
        print('K SIZE:      %d' % optimizer.k_size)
        print('DIMENSION:   %d' % optimizer.dim)
        print('LAMBDA:      %.2f' % optimizer.lam)
        print('ETA:         %.2f' % optimizer.eta)
        print('MAX_ITER:    %d' % optimizer.max_iter)
        print('EPSILON:     %.1e' % optimizer.eps)
        print('GROUPING:    %s' % optimizer.grouping_strategy)
        print('K SAMPLING:  %s' % optimizer.sample_strategy)
        print('-------------------------------------')
    for i in range(len(micro)):
        print('Train percentage:  %.2f' % (1 - 1 / len(micro[i])))
        print('Fold           Micro       Macro')
        print('-------------------------------------')
        for ite in range(len(micro[i])):
            print('Fold #%d:      %.4f      %.4f'
                  % (ite + 1, micro[i][ite], macro[i][ite]))
        print('-------------------------------------')
        print('Average:      %.4f      %.4f'
              % (np.mean(micro[i]), np.mean(macro[i])))


def multi_class_classification(optimizer, sample_filename, cv=True,
                               test_percentage=TEST_PERCT,
                               cross_val_fold=CROSS_VAL_FOLD,
                               n_shuffle=SHUFFLE):
    graph = optimizer.graph
    samples = read_pairs(sample_filename)
    samples = sample_id_mapping(samples, graph.vid2newVid_mapping)
    random.shuffle(samples)
    x_list = optimizer.embed([t[0] for t in samples])
    x_matrix = np.concatenate(x_list, axis=1).T
    y_list = [t[1] for t in samples]

    if not cv:
        micro_results = np.zeros((len(test_percentage), n_shuffle))
        macro_results = np.zeros((len(test_percentage), n_shuffle))

        for i, percentage in enumerate(test_percentage):
            for ite in range(n_shuffle):
                x_train, x_test, y_train, y_test = train_test_split(
                    x_matrix, y_list,
                    test_size=percentage)
                model = OneVsRestClassifier(LinearSVC())
                model.fit(x_train, y_train)
                pred = model.predict(x_test)

                micro_results[i, ite] = f1_score(y_test, pred, average='micro')
                macro_results[i, ite] = f1_score(y_test, pred, average='macro')

        show_results_shuffle(optimizer, micro_results, macro_results, test_percentage)
    else:
        micro_results = []
        macro_results = []

        for i, k_fold in enumerate(cross_val_fold):
            model = OneVsRestClassifier(LinearSVC())
            micro_results.append(cross_val_score(model, x_matrix, y_list, scoring='f1_micro', cv=k_fold))
            macro_results.append(cross_val_score(model, x_matrix, y_list, scoring='f1_macro', cv=k_fold))

        show_results_cv(optimizer, micro_results, macro_results)


def multi_class_classification_dw(embedding_filename, sample_filename, cv=True,
                                  test_percentage=TEST_PERCT,
                                  cross_val_fold=CROSS_VAL_FOLD,
                                  n_shuffle=SHUFFLE):
    embeddings = read_embeddings(embedding_filename)
    samples = read_pairs(sample_filename)
    random.shuffle(samples)
    x_list = [embeddings[t[0]] for t in samples]
    x_matrix = np.stack(x_list)
    del x_list
    y_list = [t[1] for t in samples]

    if not cv:
        micro_results = np.zeros((len(test_percentage), n_shuffle))
        macro_results = np.zeros((len(test_percentage), n_shuffle))

        for i, percentage in enumerate(test_percentage):
            for ite in range(n_shuffle):
                x_train, x_test, y_train, y_test = train_test_split(
                    x_matrix, y_list,
                    test_size=percentage)
                model = OneVsRestClassifier(LinearSVC())
                model.fit(x_train, y_train)
                pred = model.predict(x_test)

                micro_results[i, ite] = f1_score(y_test, pred, average='micro')
                macro_results[i, ite] = f1_score(y_test, pred, average='macro')

        show_results_shuffle(None, micro_results, macro_results, test_percentage)
    else:
        micro_results = []
        macro_results = []

        for i, k_fold in enumerate(cross_val_fold):
            model = OneVsRestClassifier(LinearSVC())
            micro_results.append(cross_val_score(model, x_matrix, y_list, scoring='f1_micro', cv=k_fold))
            macro_results.append(cross_val_score(model, x_matrix, y_list, scoring='f1_macro', cv=k_fold))

        show_results_cv(None, micro_results, macro_results)


def multi_label_classification(optimizer, sample_filename, test_percentage=TEST_PERCT,
                               n_shuffle=SHUFFLE, verbose=True):
    samples = read_multi_labels(sample_filename)
    if verbose:
        print('Samples Read.')
    random.shuffle(samples)
    x_list = optimizer.embed([t[0] for t in samples])
    if verbose:
        print('Vertices Embedded.')
    x_matrix = np.concatenate(x_list, axis=1).T
    del x_list
    y_list = [t[1] for t in samples]
    y_matrix = np.array(y_list)

    micro_results = np.zeros((len(test_percentage), n_shuffle))
    macro_results = np.zeros((len(test_percentage), n_shuffle))

    for i, percentage in enumerate(test_percentage):
        if verbose:
            print('Training classifier with percentage %.2f' % percentage)
        for ite in range(n_shuffle):
            x_train, x_test, y_train, y_test = train_test_split(
                x_matrix, y_matrix,
                test_size=percentage)
            model = OneVsRestClassifier(LinearSVC())
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

            micro_results[i, ite] = f1_score(y_test, pred, average='micro')
            macro_results[i, ite] = f1_score(y_test, pred, average='macro')

    show_results_shuffle(None, micro_results, macro_results, test_percentage)


def multi_label_classification_dw(embedding_filename, sample_filename, test_percentage=TEST_PERCT,
                                  n_shuffle=SHUFFLE, verbose=True):
    embeddings = read_embeddings(embedding_filename)
    if verbose:
        print('Embeddings Read.')
    samples = read_multi_labels(sample_filename)
    if verbose:
        print('Samples Read.')
    random.shuffle(samples)
    x_list = [embeddings[t[0]] for t in samples]
    x_matrix = np.stack(x_list)
    del x_list
    y_list = [t[1] for t in samples]
    y_matrix = np.array(y_list)

    micro_results = np.zeros((len(test_percentage), n_shuffle))
    macro_results = np.zeros((len(test_percentage), n_shuffle))

    for i, percentage in enumerate(test_percentage):
        if verbose:
            print('Training classifier with percentage %.2f' % percentage)
        for ite in range(n_shuffle):
            if verbose:
                print('Training for iteration %d/%d' % (ite + 1, n_shuffle))
            x_train, x_test, y_train, y_test = train_test_split(
                x_matrix, y_matrix,
                test_size=percentage)
            model = OneVsRestClassifier(LinearSVC(tol=1e-2, max_iter=50, verbose=1))
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

            micro = f1_score(y_test, pred, average='micro')
            macro = f1_score(y_test, pred, average='macro')

            micro_results[i, ite] = micro
            macro_results[i, ite] = macro

            print('%.4f; %.4f' % (micro, macro))

    show_results_shuffle(None, micro_results, macro_results, test_percentage)


if __name__ == '__main__':
    multi_label_classification_dw('data\\flickr\\dw10.txt', 'data\\flickr\\samples.txt')
    #multi_label_classification_dw('data\\del3.txt', 'data\\del2.txt', test_percentage=[0.2, 0.5])
