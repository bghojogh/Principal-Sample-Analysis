from experiments import Experiments
import pandas as pd
import os
import pickle
import numpy as np


def main():
    # ----- settings:
    experiment_type = 1
    split_in_cross_validation_again = False
    find_ranks_in_PSA_again = False
    portion_of_test_in_dataset = 0.3
    number_of_folds = 10
    portion_of_sampled_dataset_vector = [0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    classifiers_for_experiments = ['SVM', 'LDA', 'QDA', 'Random Forest', 'Logistic Regression', 'Gaussian Naive Bayes']
    path_to_save = './PSA_outputs/'

    # ---- path of dataset:
    path_dataset = './dataset/Breast_cancer_dataset/wdbc_data.txt'
    # ---- read the dataset:
    print('############################## Reading dataset and splitting it to K-fold train and test sets')
    data = pd.read_csv(path_dataset, sep=",", header=None)  # read text file using pandas dataFrame: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
    labels_of_classes = ['M', 'B']
    X, y = read_dataset(data=data, labels_of_classes=labels_of_classes)
    experiments = Experiments()
    # # --- saving/loading split dataset in/from folder:
    # if split_in_cross_validation_again:
    #     train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = experiments.cross_validation(X=X, y=y, n_splits=number_of_folds, test_size=portion_of_test_in_dataset)
    #     save_variable(train_indices_in_folds, 'train_indices_in_folds', path_to_save=path_to_save)
    #     save_variable(test_indices_in_folds, 'test_indices_in_folds', path_to_save=path_to_save)
    #     save_variable(X_train_in_folds, 'X_train_in_folds', path_to_save=path_to_save)
    #     save_variable(X_test_in_folds, 'X_test_in_folds', path_to_save=path_to_save)
    #     save_variable(y_train_in_folds, 'y_train_in_folds', path_to_save=path_to_save)
    #     save_variable(y_test_in_folds, 'y_test_in_folds', path_to_save=path_to_save)
    # else:
    #     file = open(path_to_save+'train_indices_in_folds.pckl','rb')
    #     train_indices_in_folds = pickle.load(file); file.close()
    #     file = open(path_to_save+'test_indices_in_folds.pckl','rb')
    #     test_indices_in_folds = pickle.load(file); file.close()
    #     file = open(path_to_save+'X_train_in_folds.pckl','rb')
    #     X_train_in_folds = pickle.load(file); file.close()
    #     file = open(path_to_save+'X_test_in_folds.pckl','rb')
    #     X_test_in_folds = pickle.load(file); file.close()
    #     file = open(path_to_save+'y_train_in_folds.pckl','rb')
    #     y_train_in_folds = pickle.load(file); file.close()
    #     file = open(path_to_save+'y_test_in_folds.pckl','rb')
    #     y_test_in_folds = pickle.load(file); file.close()

    # ----- experiments:
    if experiment_type == 1:
        experiments.multi_class_demo()

    # recognition_rate_in_folds = []
    # for fold_index in range(number_of_folds):
    #     print('############################## Cross validation: fold number ' + str(fold_index+1) + ' out of ' + str(number_of_folds) + ' folds:')
    #     # --- taking the X and y of train and test sets for classification:
    #     X_train = X_train_in_folds[fold_index]
    #     X_test = X_test_in_folds[fold_index]
    #     y_train = y_train_in_folds[fold_index]
    #     y_test = y_test_in_folds[fold_index]
    #     # --- classification:
    #     if experiment_type == 1:
    #         experiments.multi_class_demo()
    #     elif experiment_type == 2:
    #         experiments.no_class_demo()
    #     elif experiment_type == 3:
    #         recognition_rate = experiments.classify_using_the_entire_dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, classifiers_for_experiments=classifiers_for_experiments)
    #     elif experiment_type == 4:
    #         recognition_rate = experiments.classify_using_PSA(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, portion_of_sampled_dataset_vector=portion_of_sampled_dataset_vector, classifiers_for_experiments=classifiers_for_experiments, fold_index=fold_index, find_ranks_in_PSA_again=find_ranks_in_PSA_again)
    #     elif experiment_type == 5:
    #         recognition_rate = experiments.classify_using_random_sampling(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, portion_of_sampled_dataset_vector=portion_of_sampled_dataset_vector, classifiers_for_experiments=classifiers_for_experiments)
    #     elif experiment_type == 6:
    #         recognition_rate = experiments.classify_using_sortingByDistanceToMean(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, portion_of_sampled_dataset_vector=portion_of_sampled_dataset_vector, classifiers_for_experiments=classifiers_for_experiments)
    #     # --- saving results:
    #     recognition_rate_in_folds.append(recognition_rate)
    #     save_variable(recognition_rate, 'recognition_rate', path_to_save=path_to_save+'fold_'+str(fold_index+1)+'/')
    #     save_np_array_to_txt(recognition_rate, 'recognition_rate', path_to_save=path_to_save+'fold_'+str(fold_index+1)+'/')
    # # --- averaging the rates:
    # recognition_rate_average = np.asarray(recognition_rate_in_folds).mean(axis=0)
    # # --- saving results in folder:
    # save_variable(recognition_rate_in_folds, 'recognition_rate_in_folds', path_to_save=path_to_save)
    # save_variable(recognition_rate_average, 'recognition_rate_average', path_to_save=path_to_save)
    # save_np_array_to_txt(recognition_rate_average, 'recognition_rate_average', path_to_save=path_to_save)
    # # --- report the results:
    # print('############################## The average recognition rates (rows are classifiers, and columns are portions of sampling):')
    # print(recognition_rate_average)


####### functions:

def read_dataset(data, labels_of_classes):
    data = data.values  # converting pandas dataFrame to numpy array
    labels = data[:,1]
    total_number_of_samples = data.shape[0]
    X = data[:,2:]
    X = X.astype(np.float32)  # if we don't do that, we will have this error: https://www.reddit.com/r/learnpython/comments/7ivopz/numpy_getting_error_on_matrix_inverse/
    y = [None] * (total_number_of_samples)  # numeric labels
    for sample_index in range(total_number_of_samples):
        if labels[sample_index] == labels_of_classes[0]:  # first class
            y[sample_index] = 0
        elif labels[sample_index] == labels_of_classes[1]:  # second class
            y[sample_index] = 1
    return X, y

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

if __name__ == '__main__':
    main()
