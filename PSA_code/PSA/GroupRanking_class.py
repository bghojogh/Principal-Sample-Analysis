import math
import random
from PSA.Regression_class import *


class GroupRanking:

    def __init__(self, number_of_iterations_of_RANSAC = 1):
        self.number_of_iterations_of_RANSAC = number_of_iterations_of_RANSAC

    def find_best_group(self, number_of_selected_samples_in_group, class_index, class_samples_list, simplified_version=True, HavingClasses=True):
        samples_of_class = class_samples_list[class_index]  # ----- samples_of_class --> rows: samples, columns: dimensions
        number_of_samples = samples_of_class.shape[0]
        dimension_of_data = samples_of_class.shape[1]
        if number_of_selected_samples_in_group < dimension_of_data:
            print('Technical Error: cannot do regression (because number of data samples is less than dimension of data)')
            return -1
        # ----- instantiate object from class Regression:
        regression = Regression()
        # ----- RANSAC:
        best_group_score = -1 * math.inf
        group_sample_indices = np.zeros((1,number_of_selected_samples_in_group))
        for iteration in range(self.number_of_iterations_of_RANSAC):
            # ----- finding mask for select several samples out of samples of class:
            selected_samples_indices = random.sample(range(0, number_of_samples), number_of_selected_samples_in_group)  # https://stackoverflow.com/questions/22842289/generate-n-unique-random-numbers-within-a-range
            mask = np.in1d(range(number_of_samples), selected_samples_indices)  # https://stackoverflow.com/questions/27303225/numpy-vstack-empty-initialization
            # ----- Extract Y (labels for regression) as one of the dimensions of data:
            for dimension in range(dimension_of_data-1):  # iterate on all dimensions except the last one which will be redundant
                # ----- exclude the dimension of label from samples:
                labels_for_regression = np.zeros((samples_of_class.shape[0], 1))
                labels_for_regression[:,0] = samples_of_class[:,dimension]
                if dimension == 0:
                    X_for_regression = samples_of_class[:,1:]
                else:
                    X_for_regression = np.hstack([samples_of_class[:,:dimension], samples_of_class[:,dimension+1:]])
                # ----- regression on all samples of class:
                beta_all = regression.linearRegression_findBeta(X=X_for_regression, Y=labels_for_regression)
                # ----- select several samples out of samples of class
                X_selected = X_for_regression[mask,:]
                Y_selected = labels_for_regression[mask,:]
                samples_of_class_selected = samples_of_class[mask,:]
                # ----- regress on the selected samples:
                beta_group = regression.linearRegression_findBeta(X=X_selected, Y=Y_selected, resolve_singularity=True)
                # ----- find group score:
                group_score = self.calculate_group_score(beta_all, beta_group, samples_of_class_selected, class_index, class_samples_list, simplified_version=simplified_version, HavingClasses=HavingClasses)
                # ----- update best group score (if the new group is better):
                if group_score > best_group_score:
                    best_group_score = group_score
                    group_sample_indices = selected_samples_indices
                    beta_of_best_group = beta_group
        return group_sample_indices, best_group_score, beta_all, beta_of_best_group

    def calculate_group_score(self, beta_all, beta_group, samples_of_class_selected, class_index, class_samples_list, simplified_version=True, HavingClasses=True):
        # ----- find cosine of vectors:
        cosine_of_normal_vectors = self.cosine_of_vectors(vector1=beta_all, vector2=beta_group)
        # ----- find variance of vectors:
        variance_of_samples_of_group = self.variance_of_vectors(samples_of_class_selected)
        # ----- find between-scatter:
        if HavingClasses:
            if not simplified_version:
                between_scatter = self.between_scatter_NotSimplifiedVersion(class_samples_list=class_samples_list, grouped_samples=samples_of_class_selected, class_index=class_index)
            else:
                between_scatter = self.between_scatter_SimplifiedVersion(class_samples_list=class_samples_list, grouped_samples=samples_of_class_selected, class_index=class_index)
        else:
            between_scatter = 1
        # ----- find within-scatter:
        if not simplified_version:
            within_scatter = self.within_scatter_NotSimplifiedVersion(class_samples_list=class_samples_list, grouped_samples=samples_of_class_selected, class_index=class_index)
        else:
            within_scatter = self.within_scatter_SimplifiedVersion(class_samples_list=class_samples_list, grouped_samples=samples_of_class_selected, class_index=class_index)
        # ----- find group score:
        group_score = (1/within_scatter) * between_scatter * variance_of_samples_of_group * cosine_of_normal_vectors
        return group_score

    def cosine_of_vectors(self, vector1, vector2):
        inner_product_of_vectors = np.sum(vector1 * vector2)
        multiplication_of_magnitudes_of_vectors = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cosine_of_vectors = inner_product_of_vectors / multiplication_of_magnitudes_of_vectors
        return cosine_of_vectors

    def calculate_weight(self, other_sample, mean_of_class_of_other_sample):
        cosine = self.cosine_of_vectors(vector1=other_sample, vector2=mean_of_class_of_other_sample)
        weight = (cosine + 1) / 2
        return weight

    def variance_of_vectors(self, vectors):
        # ---- vectors --> rows: samples, columns: dimensions
        vectors_average = vectors.mean(axis=0)
        x_average = vectors_average.T
        number_of_samples = vectors.shape[0]
        scatter = 0
        for sample in range(number_of_samples):
            x = np.matrix(vectors[sample,:].T)
            difference_from_mean = np.array(x - x_average)
            scatter += np.multiply(difference_from_mean, difference_from_mean.T)
        e_vals, e_vecs = np.linalg.eigh(scatter)
        # for eigenvalue, see: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eigh.html
        # https://stackoverflow.com/questions/6684238/whats-the-fastest-way-to-find-eigenvalues-vectors-in-python
        variance = e_vals.sum()
        return variance

    def variance_of_vectors_UsingEuclidean(self, vectors):
        # ---- vectors --> rows: samples, columns: dimensions
        vectors_average = vectors.mean(axis=0)
        number_of_samples = vectors.shape[0]
        variance = 0
        for sample in range(number_of_samples):
            vector = vectors[sample, :]
            difference = self.Euclidean_distance(vector1=vector, vector2=vectors_average)
            variance += difference**2
        variance = variance / number_of_samples
        return variance

    def between_scatter_NotSimplifiedVersion(self, class_samples_list, grouped_samples, class_index):
        weight = np.array([0.05,0.2,0.5,0.2,0.05])
        mean_of_grouped_samples = grouped_samples.mean(axis=0)
        number_of_classes = len(class_samples_list)
        dimension_of_data = grouped_samples.shape[1]
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index_2 in range(number_of_classes):  # iteration on other classes
            if class_index != class_index_2:
                important_regions_of_distributions = self.find_important_regions_in_distibution(class_samples_list[class_index_2])
                for dimension_index in range(len(important_regions_of_distributions)):
                    for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
                        important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
                        x1 = np.matrix(mean_of_grouped_samples).T
                        x2 = np.matrix(important_region_point).T
                        difference = np.array(x1 - x2)
                        scatter += np.dot(difference, difference.T) * weight[region_index]
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    def between_scatter_SimplifiedVersion(self, class_samples_list, grouped_samples, class_index):
        mean_of_grouped_samples = grouped_samples.mean(axis=0)
        number_of_classes = len(class_samples_list)
        dimension_of_data = grouped_samples.shape[1]
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index_2 in range(number_of_classes):  # iteration on other classes
            if class_index != class_index_2:
                samples_of_other_class = class_samples_list[class_index_2]
                variance_of_other_class = self.variance_of_vectors(vectors=samples_of_other_class)
                mean_of_other_class = samples_of_other_class.mean(axis=0)
                number_of_samples_of_other_class = samples_of_other_class.shape[0]
                for sample_index_of_other_class in range(number_of_samples_of_other_class):
                    sample_in_other_class = class_samples_list[class_index_2][sample_index_of_other_class, :]
                    x1 = np.matrix(mean_of_grouped_samples).T
                    x2 = np.matrix(sample_in_other_class).T
                    difference = np.array(x1 - x2)
                    weight = self.calculate_weight(other_sample=sample_in_other_class, mean_of_class_of_other_sample=mean_of_other_class)
                    # weight = (variance_of_samples_in_group - self.Euclidean_distance(vector1=sample2, vector2=mean_of_samples_in_group)) / variance_of_samples_in_group
                    # if weight < 0: weight = 0
                    scatter += np.dot(difference, difference.T) * weight
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    def within_scatter_NotSimplifiedVersion(self, class_samples_list, grouped_samples, class_index):
        weight = np.array([0.05,0.2,0.5,0.2,0.05])
        dimension_of_data = grouped_samples.shape[1]
        important_regions_of_distributions = self.find_important_regions_in_distibution(class_samples_list[class_index])
        number_of_samples_in_group = grouped_samples.shape[0]
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for sample_index in range(number_of_samples_in_group):
            sample = grouped_samples[sample_index,:]
            for dimension_index in range(len(important_regions_of_distributions)):
                for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
                    important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
                    x1 = np.matrix(sample).T
                    x2 = np.matrix(important_region_point).T
                    difference = np.array(x1 - x2)
                    scatter += np.dot(difference, difference.T) * weight[region_index]
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    def within_scatter_SimplifiedVersion(self, class_samples_list, grouped_samples, class_index):
        dimension_of_data = grouped_samples.shape[1]
        number_of_samples_in_group = grouped_samples.shape[0]
        mean_of_samples_in_group = grouped_samples.mean(axis=0)
        variance_of_samples_in_group = self.variance_of_vectors(vectors=grouped_samples)
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for sample_index in range(number_of_samples_in_group):
            sample = grouped_samples[sample_index,:]
            for sample_index2 in range(number_of_samples_in_group):
                if sample_index != sample_index2:
                    sample2 = grouped_samples[sample_index2,:]
                    x1 = np.matrix(sample).T
                    x2 = np.matrix(sample2).T
                    difference = np.array(x1 - x2)
                    weight = self.calculate_weight(other_sample=sample2, mean_of_class_of_other_sample=mean_of_samples_in_group)
                    # weight = (variance_of_samples_in_group - self.Euclidean_distance(vector1=sample2, vector2=mean_of_samples_in_group)) / variance_of_samples_in_group
                    # if weight < 0: weight = 0
                    scatter += np.dot(difference, difference.T) * weight
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    def Euclidean_distance(self, vector1, vector2):
        return (np.sum((vector1 - vector2)**2))**0.5

    def find_important_regions_in_distibution(self, data):
        # ----> data: rows are samples, columns are dimensions
        dimension_of_data = data.shape[1]
        covariance_matix = np.cov(data.T)
        mean_of_distribution = data.mean(axis=0)
        important_regions_of_distributions = [None] * dimension_of_data
        for dimension_index1 in range(dimension_of_data):
            standard_deviation = covariance_matix[dimension_index1,dimension_index1]**0.5
            important_regions = np.zeros((5,dimension_of_data))
            for important_region_index in range(5):  # the 5 important region
                for dimension_index2 in range(dimension_of_data):  # dimensions of important regions
                    if dimension_index2 == dimension_index1:
                        if important_region_index == 0:
                            important_regions[important_region_index, dimension_index2] = mean_of_distribution[dimension_index1] - 2*standard_deviation
                        elif important_region_index == 1:
                            important_regions[important_region_index, dimension_index2] = mean_of_distribution[dimension_index1] - 1*standard_deviation
                        elif important_region_index == 2:
                            important_regions[important_region_index, dimension_index2] = mean_of_distribution[dimension_index1] - 0*standard_deviation
                        elif important_region_index == 3:
                            important_regions[important_region_index, dimension_index2] = mean_of_distribution[dimension_index1] + 1*standard_deviation
                        elif important_region_index == 4:
                            important_regions[important_region_index, dimension_index2] = mean_of_distribution[dimension_index1] + 2*standard_deviation
                    else:
                        important_regions[important_region_index, dimension_index2] = mean_of_distribution[dimension_index2]
            important_regions_of_distributions[dimension_index1] = important_regions
        return important_regions_of_distributions

