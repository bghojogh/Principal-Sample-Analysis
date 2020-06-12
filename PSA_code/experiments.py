from PSA.PSA_main import *
import numpy as np
from sklearn.svm import SVC   # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA   # http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
from sklearn.ensemble import RandomForestClassifier as RF    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA   # http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
from sklearn.linear_model import LogisticRegression as LR   # http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976   AND   http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.naive_bayes import GaussianNB   # http://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.model_selection import StratifiedShuffleSplit   # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit


class Experiments:

    def __init__(self):
        pass

    def multi_class_demo(self):
        psa = PSA(number_of_selected_samples_in_group=10, number_of_iterations_of_RANSAC=20, save_ranks=True)
        psa.set_demo_settings(number_of_classes=3, dimension_of_data=2, number_of_samples_of_each_class=[30,50,40])
        ranks = psa.rank_samples(X=None, y=None, demo_mode=True, report_steps=True, HavingClasses=True)

    def no_class_demo(self):
        psa = PSA(number_of_selected_samples_in_group=10, number_of_iterations_of_RANSAC=20, save_ranks=True)
        psa.set_demo_settings(number_of_classes=1, dimension_of_data=2, number_of_samples_of_each_class=[30])
        ranks = psa.rank_samples(X=None, y=None, demo_mode=True, report_steps=True, HavingClasses=False)

    def cross_validation(self, X, y, n_splits=10, test_size=0.3):
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)
        train_indices_in_folds = []; test_indices_in_folds = []
        X_train_in_folds = []; X_test_in_folds = []
        y_train_in_folds = []; y_test_in_folds = []
        for train_index, test_index in sss.split(X, y):
            train_indices_in_folds.append(train_index)
            test_indices_in_folds.append(test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]
            X_train_in_folds.append(X_train)
            X_test_in_folds.append(X_test)
            y_train_in_folds.append(y_train)
            y_test_in_folds.append(y_test)
        return train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

    def classify_using_the_entire_dataset(self, X_train, X_test, y_train, y_test, classifiers_for_experiments):
        recognition_rate_LIST = np.zeros((len(classifiers_for_experiments), 1))
        classifier_index = 0
        for classifier in classifiers_for_experiments:
            print('############### Classifier: ' + classifier)
            # ---- classify without PSA:
            if classifier == 'SVM':
                # --------- train:
                clf = SVC(kernel='linear')
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'LDA':
                # --------- train:
                clf = LDA()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'QDA':
                # --------- train:
                clf = QDA()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'Random Forest':
                # --------- train:
                clf = RF(max_depth=2, random_state=0)
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'Logistic Regression':
                # --------- train:
                clf = LR()
                clf.fit(X=X_train, y=y_train)
            elif classifier == 'Gaussian Naive Bayes':
                # --------- train:
                clf = GaussianNB()
                clf.fit(X=X_train, y=y_train)
            # --------- test:
            labels_predicted = clf.predict(X_test)
            recognition_rate_entireDataset = (sum(labels_predicted == y_test) / len(labels_predicted)) * 100
            print('The recognition rate using ' + classifier + ' without data number reduction: ' + str(recognition_rate_entireDataset))
            recognition_rate_LIST[classifier_index] = recognition_rate_entireDataset
            classifier_index += 1
        return recognition_rate_LIST

    def classify_using_PSA(self, X_train, X_test, y_train, y_test, portion_of_sampled_dataset_vector, classifiers_for_experiments, fold_index, find_ranks_in_PSA_again=True):
        # ---- Applying PSA:
        psa = PSA(number_of_selected_samples_in_group=50, number_of_iterations_of_RANSAC=20, save_ranks=False)
        path_of_ranks = './PSA_outputs/fold_' + str(fold_index+1) + '/'
        if find_ranks_in_PSA_again:
            # psa.set_demo_settings(number_of_classes=2, dimension_of_data=2, number_of_samples_of_each_class=[30, 50])
            ranks = psa.rank_samples(X=X_train, y=y_train, demo_mode=False, report_steps=True, HavingClasses=True)
            self.save_variable(ranks, 'ranks', path_to_save=path_of_ranks)
        else:
            file = open(path_of_ranks,'rb')
            ranks = pickle.load(file)
            file.close()
        # ---- sort samples of classes according to their ranks:
        sorted_samples = psa.sort_samples_according_to_ranks(X=X_train, y=y_train, ranks=ranks)
        # ---- Experimenting:
        recognition_rate_LIST = np.zeros((len(classifiers_for_experiments), len(portion_of_sampled_dataset_vector)))
        classifier_index = 0
        for classifier in classifiers_for_experiments:
            print('############### Classifier: ' + classifier)
            portion_index = 0
            for portion_of_sampled_dataset in portion_of_sampled_dataset_vector:
                print('###### Portion of sampled dataset: ' + str(portion_of_sampled_dataset * 100) + '%')
                # ---- data reduction with PSA:
                number_of_classes = len(sorted_samples)
                n_samples = []
                for class_index in range(number_of_classes):
                    number_of_samples_of_class = sorted_samples[class_index].shape[0]
                    n_samples.append(int(number_of_samples_of_class * portion_of_sampled_dataset))
                X, y = psa.reduce_data(sorted_samples=sorted_samples, n_samples=n_samples)
                # ---- report number of sampled data after PSA:
                print('number of sampled data in classes, after PSA: ' + str(n_samples))
                # ---- classify with PSA:
                if classifier == 'SVM':
                    # --------- train:
                    clf = SVC(kernel='linear')
                    clf.fit(X=X, y=y)
                elif classifier == 'LDA':
                    # --------- train:
                    clf = LDA()
                    clf.fit(X=X, y=y)
                elif classifier == 'QDA':
                    # --------- train:
                    clf = QDA()
                    clf.fit(X=X, y=y)
                elif classifier == 'Random Forest':
                    # --------- train:
                    clf = RF(max_depth=2, random_state=0)
                    clf.fit(X=X, y=y)
                elif classifier == 'Logistic Regression':
                    # --------- train:
                    clf = LR()
                    clf.fit(X=X, y=y)
                elif classifier == 'Gaussian Naive Bayes':
                    # --------- train:
                    clf = GaussianNB()
                    clf.fit(X=X, y=y)
                # --------- test:
                labels_predicted = clf.predict(X_test)
                recognition_rate_PSA = (sum(labels_predicted == y_test) / len(labels_predicted)) * 100
                print('The recognition rate using ' + classifier + ' with data number reduction (PSA): ' + str(recognition_rate_PSA))
                recognition_rate_LIST[classifier_index, portion_index] = recognition_rate_PSA
                portion_index += 1
            classifier_index += 1
        return recognition_rate_LIST

    def classify_using_random_sampling(self, X_train, X_test, y_train, y_test, portion_of_sampled_dataset_vector, classifiers_for_experiments):
        psa = PSA()
        # ---- settings:
        number_of_runs_for_random_sampling = 20
        # ---- Experimenting:
        recognition_rate_LIST = np.zeros((len(classifiers_for_experiments), len(portion_of_sampled_dataset_vector)))
        classifier_index = 0
        for classifier in classifiers_for_experiments:
            print('############### Classifier: ' + classifier)
            portion_index = 0
            for portion_of_sampled_dataset in portion_of_sampled_dataset_vector:
                print('###### Portion of sampled dataset: ' + str(portion_of_sampled_dataset * 100) + '%')
                # ---- data reduction with random sampling:
                recognition_rate_with_random_sampling = [None] * number_of_runs_for_random_sampling
                for run_index in range(number_of_runs_for_random_sampling):
                    shuffled_samples = self.shuffle_samples_randomly(X=X_train, y=y_train)  # shuffle samples of classes randomly
                    # ---- data reduction:
                    number_of_classes = len(shuffled_samples)
                    n_samples = []
                    for class_index in range(number_of_classes):
                        number_of_samples_of_class = shuffled_samples[class_index].shape[0]
                        n_samples.append(int(number_of_samples_of_class * portion_of_sampled_dataset))
                    X, y = psa.reduce_data(sorted_samples=shuffled_samples, n_samples=n_samples)
                    # ---- report number of sampled data after PSA:
                    if run_index == 0:  # only report once in the multiple runs
                        print('number of sampled data in classes, after random sampling: ' + str(n_samples))
                    # ---- classify with random sampling:
                    if classifier == 'SVM':
                        # --------- train:
                        clf = SVC(kernel='linear')
                        clf.fit(X=X, y=y)
                    elif classifier == 'LDA':
                        # --------- train:
                        clf = LDA()
                        clf.fit(X=X, y=y)
                    elif classifier == 'QDA':
                        # --------- train:
                        clf = QDA()
                        clf.fit(X=X, y=y)
                    elif classifier == 'Random Forest':
                        # --------- train:
                        clf = RF(max_depth=2, random_state=0)
                        clf.fit(X=X, y=y)
                    elif classifier == 'Logistic Regression':
                        # --------- train:
                        clf = LR()
                        clf.fit(X=X, y=y)
                    elif classifier == 'Gaussian Naive Bayes':
                        # --------- train:
                        clf = GaussianNB()
                        clf.fit(X=X, y=y)
                    # --------- test:
                    labels_predicted = clf.predict(X_test)
                    recognition_rate_with_random_sampling[run_index] = (sum(labels_predicted == y_test) / len(labels_predicted)) * 100
                recognition_rate_with_random_sampling_average = np.mean(recognition_rate_with_random_sampling)
                print('The recognition rate using ' + classifier + ' with data number reduction (random sampling): ' + str(recognition_rate_with_random_sampling_average))
                recognition_rate_LIST[classifier_index, portion_index] = recognition_rate_with_random_sampling_average
                portion_index += 1
            classifier_index += 1
        return recognition_rate_LIST

    def classify_using_sortingByDistanceToMean(self, X_train, X_test, y_train, y_test, portion_of_sampled_dataset_vector, classifiers_for_experiments):
        psa = PSA()
        # ---- sort samples of classes according to their ranks:
        sorted_samples, ranks = self.sort_samples_by_distance_from_mean(X=X_train, y=y_train)
        # ---- Experimenting:
        recognition_rate_LIST = np.zeros((len(classifiers_for_experiments), len(portion_of_sampled_dataset_vector)))
        classifier_index = 0
        for classifier in classifiers_for_experiments:
            print('############### Classifier: ' + classifier)
            portion_index = 0
            for portion_of_sampled_dataset in portion_of_sampled_dataset_vector:
                print('###### Portion of sampled dataset: ' + str(portion_of_sampled_dataset * 100) + '%')
                # ---- data reduction with PSA:
                number_of_classes = len(sorted_samples)
                n_samples = []
                for class_index in range(number_of_classes):
                    number_of_samples_of_class = sorted_samples[class_index].shape[0]
                    n_samples.append(int(number_of_samples_of_class * portion_of_sampled_dataset))
                X, y = psa.reduce_data(sorted_samples=sorted_samples, n_samples=n_samples)
                # ---- report number of sampled data after PSA:
                print('number of sampled data in classes, after PSA: ' + str(n_samples))
                # ---- classify with PSA:
                if classifier == 'SVM':
                    # --------- train:
                    clf = SVC(kernel='linear')
                    clf.fit(X=X, y=y)
                elif classifier == 'LDA':
                    # --------- train:
                    clf = LDA()
                    clf.fit(X=X, y=y)
                elif classifier == 'QDA':
                    # --------- train:
                    clf = QDA()
                    clf.fit(X=X, y=y)
                elif classifier == 'Random Forest':
                    # --------- train:
                    clf = RF(max_depth=2, random_state=0)
                    clf.fit(X=X, y=y)
                elif classifier == 'Logistic Regression':
                    # --------- train:
                    clf = LR()
                    clf.fit(X=X, y=y)
                elif classifier == 'Gaussian Naive Bayes':
                    # --------- train:
                    clf = GaussianNB()
                    clf.fit(X=X, y=y)
                # --------- test:
                labels_predicted = clf.predict(X_test)
                recognition_rate_PSA = (sum(labels_predicted == y_test) / len(labels_predicted)) * 100
                print('The recognition rate using ' + classifier + ' with data number reduction (PSA): ' + str(recognition_rate_PSA))
                recognition_rate_LIST[classifier_index, portion_index] = recognition_rate_PSA
                portion_index += 1
            classifier_index += 1
        return recognition_rate_LIST

    ####### functions:

    def shuffle_samples_randomly(self, X, y):
        labels = list(set(y))  # set() removes redundant numbers in y and sorts them
        number_of_classes = len(labels)
        total_number_of_samples = len(y)
        dimension_of_data = X.shape[1]
        # creating class_samples_list out of X and y:
        class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
        for class_index in range(number_of_classes):
            class_samples_list[class_index] = np.empty([0, dimension_of_data])
        for sample_index in range(total_number_of_samples):
            for class_index in range(number_of_classes):
                if y[sample_index] == labels[class_index]:
                    class_samples_list[class_index] = np.vstack([class_samples_list[class_index], X[sample_index, :]])
        # shuffling randomly:
        shuffled_samples = [None] * number_of_classes   # sorted_samples: a list whose every element contains sorted training samples of a class. In every element of the list, sorted training samples are stacked row-wise.
        for class_index in range(number_of_classes):
            shuffled_samples[class_index] = np.zeros(class_samples_list[class_index].shape)
        for class_index in range(number_of_classes):
            samples_of_class = class_samples_list[class_index]
            number_of_samples_of_class = samples_of_class.shape[0]
            # creating random ranks (for the sake of shuffling):
            ranks_random = np.random.permutation(number_of_samples_of_class)
            for sample_index in range(number_of_samples_of_class):
                rank_among_all_samples_of_class = int(np.where(ranks_random[:] == sample_index)[0])  # find place of sample_index in ranks[class_index][:]
                shuffled_samples[class_index][rank_among_all_samples_of_class, :] = class_samples_list[class_index][sample_index, :]
        return shuffled_samples

    def sort_samples_by_distance_from_mean(self, X, y):
        labels = list(set(y))  # set() removes redundant numbers in y and sorts them
        number_of_classes = len(labels)
        total_number_of_samples = len(y)
        dimension_of_data = X.shape[1]
        # creating class_samples_list out of X and y:
        class_samples_list = [None] * number_of_classes   # class_samples_list: a list whose every element is training samples of a class. In every element of the list, training samples are stacked row-wise.
        for class_index in range(number_of_classes):
            class_samples_list[class_index] = np.empty([0, dimension_of_data])
        for sample_index in range(total_number_of_samples):
            for class_index in range(number_of_classes):
                if y[sample_index] == labels[class_index]:
                    class_samples_list[class_index] = np.vstack([class_samples_list[class_index], X[sample_index, :]])
        # shuffling randomly:
        sorted_samples = [None] * number_of_classes   # sorted_samples: a list whose every element contains sorted training samples of a class. In every element of the list, sorted training samples are stacked row-wise.
        for class_index in range(number_of_classes):
            sorted_samples[class_index] = np.zeros(class_samples_list[class_index].shape)
        ranks_list = []
        for class_index in range(number_of_classes):
            samples_of_class = class_samples_list[class_index]
            number_of_samples_of_class = samples_of_class.shape[0]
            mean_of_class = samples_of_class.mean(axis=0)
            distance = []
            for sample_index in range(number_of_samples_of_class):
                sample = class_samples_list[class_index][sample_index, :]
                distance.append(self.Euclidean_distance(vector1=sample, vector2=mean_of_class))
            ranks = np.argsort(distance)
            for sample_index in range(number_of_samples_of_class):
                rank_among_all_samples_of_class = int(np.where(ranks[:] == sample_index)[0])  # find place of sample_index in ranks[class_index][:]
                sorted_samples[class_index][rank_among_all_samples_of_class, :] = class_samples_list[class_index][sample_index, :]
            ranks_list.append(ranks)
        return sorted_samples, ranks_list

    def Euclidean_distance(self, vector1, vector2):
        return (np.sum((vector1 - vector2)**2))**0.5

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()