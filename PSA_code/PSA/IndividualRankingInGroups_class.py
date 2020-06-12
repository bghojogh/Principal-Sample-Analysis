import numpy as np

class IndividualRankingInGroups:

    def __init__(self):
        pass

    def individual_rank_of_qualified_samples(self, class_index, grouped_samples, class_samples_list, simplified_version=True, HavingClasses=True):
        # ----- grouped_samples --> rows: samples, columns: dimensions
        number_of_grouped_samples = grouped_samples.shape[0]
        individual_scores = np.zeros((number_of_grouped_samples,1))
        for sample_index in range(number_of_grouped_samples):
            sample = grouped_samples[sample_index]
            individual_scores[sample_index] = self.calculate_individual_score_of_qualified_samples(sample=sample, class_index=class_index, class_samples_list=class_samples_list, simplified_version=simplified_version, HavingClasses=HavingClasses)
        ranks = np.argsort(individual_scores, axis=0)[::-1]   # sorting in descending order
        return ranks, individual_scores

    def calculate_individual_score_of_qualified_samples(self, sample, class_index, class_samples_list, simplified_version=True, HavingClasses=True):
        # ----- find between-scatter:
        if HavingClasses:
            if not simplified_version:
                between_scatter = self.between_scatter_NotSimplifiedVersion(class_samples_list=class_samples_list, sample=sample, class_index=class_index)
            else:
                between_scatter = self.between_scatter_SimplifiedVersion(class_samples_list=class_samples_list, sample=sample, class_index=class_index)
        else:
            between_scatter = 1
        # ----- find within-scatter:
        if not simplified_version:
            within_scatter = self.within_scatter_NotSimplifiedVersion(class_samples_list=class_samples_list, sample=sample, class_index=class_index)
        else:
            within_scatter = self.within_scatter_SimplifiedVersion(class_samples_list=class_samples_list, sample=sample, class_index=class_index)
        # ----- find group score:
        individual_score = (1/within_scatter) * between_scatter
        return individual_score

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

    def Euclidean_distance(self, vector1, vector2):
        return (np.sum((vector1 - vector2)**2))**0.5

    def cosine_of_vectors(self, vector1, vector2):
        inner_product_of_vectors = np.sum(vector1 * vector2)
        multiplication_of_magnitudes_of_vectors = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cosine_of_vectors = inner_product_of_vectors / multiplication_of_magnitudes_of_vectors
        return cosine_of_vectors

    def calculate_weight(self, other_sample, mean_of_class_of_other_sample):
        cosine = self.cosine_of_vectors(vector1=other_sample, vector2=mean_of_class_of_other_sample)
        weight = (cosine + 1) / 2
        return weight

    def within_scatter_NotSimplifiedVersion(self, class_samples_list, sample, class_index):
        # sample: a horizontal vector whose columns are dimension
        weight = np.array([0.05,0.2,0.5,0.2,0.05])
        dimension_of_data = len(sample)
        important_regions_of_distributions = self.find_important_regions_in_distibution(class_samples_list[class_index])
        scatter = np.zeros((dimension_of_data, dimension_of_data))
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

    def within_scatter_SimplifiedVersion(self, class_samples_list, sample, class_index):
        # sample: a horizontal vector whose columns are dimension
        dimension_of_data = len(sample)
        samples_of_class = class_samples_list[class_index]
        number_of_samples_in_class = samples_of_class.shape[0]
        mean_of_samples_in_class = samples_of_class.mean(axis=0)
        variance_of_samples_in_class = self.variance_of_vectors(vectors=samples_of_class)
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for sample_index2 in range(number_of_samples_in_class):
            sample2 = samples_of_class[sample_index2,:]
            x1 = np.matrix(sample).T
            x2 = np.matrix(sample2).T
            difference = np.array(x1 - x2)
            weight = self.calculate_weight(other_sample=sample2, mean_of_class_of_other_sample=mean_of_samples_in_class)
            # weight = (variance_of_samples_in_class - self.Euclidean_distance(vector1=sample2, vector2=mean_of_samples_in_class)) / variance_of_samples_in_class
            # if weight < 0: weight = 0
            scatter += np.dot(difference, difference.T) * weight
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    def between_scatter_NotSimplifiedVersion(self, class_samples_list, sample, class_index):
        # sample: a horizontal vector whose columns are dimension
        weight = np.array([0.05,0.2,0.5,0.2,0.05])
        number_of_classes = len(class_samples_list)
        dimension_of_data = len(sample)
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index_2 in range(number_of_classes):  # iteration on other classes
            if class_index != class_index_2:
                important_regions_of_distributions = self.find_important_regions_in_distibution(class_samples_list[class_index_2])
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

    def between_scatter_SimplifiedVersion(self, class_samples_list, sample, class_index):
        # sample: a horizontal vector whose columns are dimension
        number_of_classes = len(class_samples_list)
        dimension_of_data = len(sample)
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index_2 in range(number_of_classes):  # iteration on other classes
            if class_index != class_index_2:
                samples_of_other_class = class_samples_list[class_index_2]
                variance_of_other_class = self.variance_of_vectors(vectors=samples_of_other_class)
                mean_of_other_class = samples_of_other_class.mean(axis=0)
                number_of_samples_of_other_class = samples_of_other_class.shape[0]
                for sample_index_of_other_class in range(number_of_samples_of_other_class):
                    sample_in_other_class = class_samples_list[class_index_2][sample_index_of_other_class, :]
                    x1 = np.matrix(sample).T
                    x2 = np.matrix(sample_in_other_class).T
                    difference = np.array(x1 - x2)
                    weight = self.calculate_weight(other_sample=sample_in_other_class, mean_of_class_of_other_sample=mean_of_other_class)
                    # weight = (variance_of_other_class - self.Euclidean_distance(vector1=sample_in_other_class, vector2=mean_of_other_class)) / variance_of_other_class
                    # if weight < 0: weight = 0
                    scatter += np.dot(difference, difference.T) * weight
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

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