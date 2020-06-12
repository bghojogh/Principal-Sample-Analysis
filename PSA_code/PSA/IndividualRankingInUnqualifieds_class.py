import numpy as np

class IndividualRankingInUnqualifieds:

    def __init__(self):
        pass

    def individual_rank_of_unqualified_samples(self, class_index, class_samples_list, Individual_ranks, group_sample_indices, HavingClasses=True):
        samples = class_samples_list[class_index]
        number_of_samples = samples.shape[0]
        # ----- extract indices of unqualified samples:
        sample_indices = [x for x in range(number_of_samples)]
        unqualified_samples_indices = sample_indices
        for group_sample_index in group_sample_indices[class_index]:
            unqualified_samples_indices.remove(group_sample_index)
        # ----- extract unqualified samples:
        mask = np.in1d(range(number_of_samples), group_sample_indices[class_index])  # https://stackoverflow.com/questions/27303225/numpy-vstack-empty-initialization
        unqualified_samples = samples[~mask,:]
        # ----- find the rank among unqualified samples of class:
        number_of_unqualified_samples = unqualified_samples.shape[0]
        individual_scores = np.zeros((number_of_unqualified_samples,1))
        for sample_index in range(number_of_unqualified_samples):
            sample = unqualified_samples[sample_index]
            individual_scores[sample_index] = self.calculate_individual_score_of_unqualified_samples(sample=sample, class_index=class_index, class_samples_list=class_samples_list, Individual_ranks=Individual_ranks, group_sample_indices=group_sample_indices, HavingClasses=HavingClasses)
        ranks_among_unqualified_samples = np.argsort(individual_scores, axis=0)[::-1]   # sorting in descending order
        # ----- find the rank among all samples of class:
        number_of_grouped_samples = len(group_sample_indices[class_index])
        rank_among_all_samples = number_of_grouped_samples-1   # initialization: the ranks of unqualified samples are after qualified (grouped) ones # -1: because rank starts from 0
        for i in range(number_of_unqualified_samples):
            sample_index_among_all_samples_of_class = unqualified_samples_indices[int(ranks_among_unqualified_samples[i])]
            rank_among_all_samples += 1
            Individual_ranks[class_index][rank_among_all_samples] = sample_index_among_all_samples_of_class
        return ranks_among_unqualified_samples, individual_scores, Individual_ranks

    def calculate_individual_score_of_unqualified_samples(self, sample, class_index, class_samples_list, Individual_ranks, group_sample_indices, HavingClasses=True):
        # ----- find between-scatter:
        if HavingClasses:
            between_scatter = self.between_scatter(class_samples_list=class_samples_list, sample=sample, class_index=class_index, Individual_ranks=Individual_ranks, group_sample_indices=group_sample_indices)
        else:
            between_scatter = 1
        # ----- find within-scatter:
        within_scatter = self.within_scatter(class_samples_list=class_samples_list, sample=sample, class_index=class_index, Individual_ranks=Individual_ranks, group_sample_indices=group_sample_indices)
        # ----- find group score:
        individual_score = (1/within_scatter) * between_scatter
        return individual_score

    def within_scatter(self, class_samples_list, sample, class_index, Individual_ranks, group_sample_indices):
        # sample: a horizontal vector whose columns are dimension
        dimension_of_data = len(sample)
        number_of_grouped_samples = len(group_sample_indices[class_index])
        samples_of_class = class_samples_list[class_index]
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for index in range(number_of_grouped_samples):
            index_of_qualified_sample_in_class_samples = int(Individual_ranks[class_index][index])
            qualified_sample = samples_of_class[index_of_qualified_sample_in_class_samples,:]
            x1 = np.matrix(sample).T
            x2 = np.matrix(qualified_sample).T
            difference = np.array(x1 - x2)
            weight = (number_of_grouped_samples - index) / ((number_of_grouped_samples * (number_of_grouped_samples+1))/2)
            scatter += np.dot(difference, difference.T) * weight
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    def between_scatter(self, class_samples_list, sample, class_index, Individual_ranks, group_sample_indices):
        # sample: a horizontal vector whose columns are dimension
        number_of_classes = len(class_samples_list)
        dimension_of_data = len(sample)
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        for class_index_2 in range(number_of_classes):  # iteration on other classes
            if class_index != class_index_2:
                samples_of_class = class_samples_list[class_index]
                number_of_grouped_samples = len(group_sample_indices[class_index])
                for index in range(number_of_grouped_samples):
                    index_of_qualified_sample_in_class_samples = int(Individual_ranks[class_index][index])
                    qualified_sample = samples_of_class[index_of_qualified_sample_in_class_samples,:]
                    x1 = np.matrix(sample).T
                    x2 = np.matrix(qualified_sample).T
                    difference = np.array(x1 - x2)
                    weight = (number_of_grouped_samples - index) / ((number_of_grouped_samples * (number_of_grouped_samples+1))/2)
                    scatter += np.dot(difference, difference.T) * weight
        e_vals, e_vecs = np.linalg.eigh(scatter)
        scatter_value = e_vals.sum()
        return scatter_value

    # def between_scatter(self, class_samples_list, sample, class_index):
    #     # sample: a horizontal vector whose columns are dimension
    #     weight = np.array([0.05,0.2,0.5,0.2,0.05])
    #     number_of_classes = len(class_samples_list)
    #     dimension_of_data = len(sample)
    #     scatter = np.zeros((dimension_of_data, dimension_of_data))
    #     for class_index_2 in range(number_of_classes):  # iteration on other classes
    #         if class_index != class_index_2:
    #             important_regions_of_distributions = self.find_important_regions_in_distibution(class_samples_list[class_index_2])
    #             for dimension_index in range(len(important_regions_of_distributions)):
    #                 for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
    #                     important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
    #                     x1 = np.matrix(sample).T
    #                     x2 = np.matrix(important_region_point).T
    #                     difference = np.array(x1 - x2)
    #                     scatter += np.dot(difference, difference.T) * weight[region_index]
    #     e_vals, e_vecs = np.linalg.eigh(scatter)
    #     scatter_value = e_vals.sum()
    #     return scatter_value

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