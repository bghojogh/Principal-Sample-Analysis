import matplotlib.pyplot as plt
import numpy as np
from PSA.GroupRanking_class import *
from PSA.IndividualRankingInUnqualifieds_class import *


def plot_samples(self, class_samples_list, noise_labels, color_of_plot, save_figures, show_images):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt1 = []; plt2 = []
    number_of_classes = len(class_samples_list)
    for class_index in range(number_of_classes):
        samples = class_samples_list[class_index]
        number_of_samples = samples.shape[0]
        for sample_index in range(number_of_samples):
            if noise_labels[class_index] is not None:
                print(noise_labels)
                if noise_labels[class_index][sample_index] == False:
                    marker_size=70
                    plt1 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='o', s=marker_size, edgecolors='k')
                else:
                    marker_size=70
                    plt2 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='s', s=marker_size, edgecolors='k', alpha=0.3)
            else:  # means that noisy samples are not added to this class:
                marker_size=70
                plt1 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='o', s=marker_size, edgecolors='k')
        # groupRanking = GroupRanking()
        # important_regions_of_distributions = groupRanking.find_important_regions_in_distibution(samples)
        # for dimension_index in range(len(important_regions_of_distributions)):
        #     for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
        #         important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
        #         plt.scatter(important_region_point[0], important_region_point[1], c='r', marker='x', s=100, linewidths=10)
        if noise_labels[class_index] is not None:   # means that we have noisy samples
            if type(plt1) is not list: plt1.set_label('Class ' + str(class_index+1) + ': Original Samples')
            if type(plt2) is not list: plt2.set_label('Class ' + str(class_index+1) + ': Noisy Samples')
        else:   # means that we don't have noisy samples
            if type(plt1) is not list: plt1.set_label('Class ' + str(class_index+1))
        ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        # fig.suptitle('Original Samples and Added Noisy Samples')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    if save_figures:
        save_path = './PSA_outputs/'
        name_of_image = 'plot_samples'
        format_of_save = 'png'
        plt.savefig(save_path + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
    if show_images: plt.show()

def plot_group_ranking(self, class_samples_list, group_sample_indices, color_of_plot, save_figures, show_images):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt1 = []; plt2 = []
    number_of_classes = len(class_samples_list)
    for class_index in range(number_of_classes):
        samples = class_samples_list[class_index]
        number_of_samples = samples.shape[0]
        for sample_index in range(number_of_samples):
            if sample_index in group_sample_indices[class_index]:
                marker_size=150
                plt1 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='s', s=marker_size, edgecolors='k')
            else:
                marker_size=40
                plt2 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='o', s=marker_size, edgecolors='k', alpha=0.3)
        # groupRanking = GroupRanking()
        # important_regions_of_distributions = groupRanking.find_important_regions_in_distibution(samples)
        # for dimension_index in range(len(important_regions_of_distributions)):
        #     for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
        #         important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
        #         plt.scatter(important_region_point[0], important_region_point[1], c='r', marker='x', s=100, linewidths=10)
        if type(plt1) is not list: plt1.set_label('Class ' + str(class_index+1) + ', major sample')
        if type(plt2) is not list: plt2.set_label('Class ' + str(class_index+1) + ', minor sample')
        ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        # fig.suptitle('Qualified (Grouped) Samples')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    if save_figures:
        save_path = './PSA_outputs/'
        name_of_image = 'plot_group_ranking'
        format_of_save = 'png'
        plt.savefig(save_path + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
    if show_images: plt.show()

def plot_individual_ranking_in_groups(self, class_samples_list, group_sample_indices, Individual_ranks, color_of_plot, save_figures, show_images):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt2 = []
    number_of_classes = len(class_samples_list)
    for class_index in range(number_of_classes):
        samples = class_samples_list[class_index]
        number_of_samples = samples.shape[0]
        for sample_index in range(number_of_samples):
            if sample_index in group_sample_indices[class_index]:
                number_of_samples_in_group = len(group_sample_indices[class_index])
                rank_among_all_samples_of_class = int(np.where(Individual_ranks[class_index][0:number_of_samples_in_group] == sample_index)[0])  # find place of sample_index in Individual_ranks[class_index][0:number_of_samples_in_group]
                biggest_marker_size = 500; smallest_marker_size = 50; step_marker_size = (biggest_marker_size-smallest_marker_size)/number_of_samples_in_group
                marker_size=biggest_marker_size-(step_marker_size*rank_among_all_samples_of_class)
                plt1 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='s', s=marker_size, edgecolors='k')
            else:
                marker_size=40
                plt2 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='o', s=marker_size, edgecolors='k', alpha=0.3)
        # groupRanking = GroupRanking()
        # important_regions_of_distributions = groupRanking.find_important_regions_in_distibution(samples)
        # for dimension_index in range(len(important_regions_of_distributions)):
        #     for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
        #         important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
        #         plt.scatter(important_region_point[0], important_region_point[1], c='r', marker='x', s=100, linewidths=10)
        if type(plt2) is not list: plt2.set_label('Class ' + str(class_index+1))
        # ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        # fig.suptitle('Ranking Qualified Samples')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    if save_figures:
        save_path = './PSA_outputs/'
        name_of_image = 'plot_ranking_qualified_samples'
        format_of_save = 'png'
        plt.savefig(save_path + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
    if show_images: plt.show()

def plot_individual_ranking_in_all(self, class_samples_list, group_sample_indices, Individual_ranks, color_of_plot, save_figures, show_images):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt1 = []
    number_of_classes = len(class_samples_list)
    grouped_samples_list = [None] * number_of_classes
    rank_of_grouped_samples_among_all_list = [None] * number_of_classes
    for class_index in range(number_of_classes):
        samples = class_samples_list[class_index]
        number_of_samples = samples.shape[0]
        number_of_samples_of_class = class_samples_list[class_index].shape[0]
        biggest_marker_size = 500; smallest_marker_size = 1; step_marker_size = (biggest_marker_size-smallest_marker_size)/number_of_samples_of_class
        grouped_samples = np.empty((0, samples.shape[1]))
        rank_of_grouped_samples_among_samples_of_class = np.empty((0, 1))
        for sample_index in range(number_of_samples):
            rank_among_all_samples_of_class = int(np.where(Individual_ranks[class_index][:] == sample_index)[0])  # find place of sample_index in Individual_ranks[class_index][:]
            marker_size=biggest_marker_size-(step_marker_size*rank_among_all_samples_of_class)
            if sample_index in group_sample_indices[class_index]:
                grouped_samples = np.vstack([grouped_samples, samples[sample_index,:]])
                rank_of_grouped_samples_among_samples_of_class = np.vstack([rank_of_grouped_samples_among_samples_of_class, rank_among_all_samples_of_class])
                # color = PSA.color_variant(self, color_of_plot[class_index], brightness_offset=-5)
                # plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color, marker='s', s=marker_size, edgecolors='k')
            else:
                color = color_variant(self, color_of_plot[class_index], brightness_offset=30)
                if rank_among_all_samples_of_class == number_of_samples_of_class-10:   # a small marker
                    plt1 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color, marker='o', s=marker_size, edgecolors='k', alpha=1)
                else:
                    plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color, marker='o', s=marker_size, edgecolors='k', alpha=1)
        grouped_samples_list[class_index] = grouped_samples
        rank_of_grouped_samples_among_all_list[class_index] = rank_of_grouped_samples_among_samples_of_class
        # groupRanking = GroupRanking()
        # important_regions_of_distributions = groupRanking.find_important_regions_in_distibution(samples)
        # for dimension_index in range(len(important_regions_of_distributions)):
        #     for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
        #         important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
        #         plt.scatter(important_region_point[0], important_region_point[1], c='r', marker='x', s=100, linewidths=10)
        if type(plt1) is not list: plt1.set_label('Class ' + str(class_index+1))
        # ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        # fig.suptitle('Ranking Samples')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    # plot the grouped samples (on top of the plot of samples):
    for class_index in range(number_of_classes):
        # number_of_samples_of_class = class_samples_list[class_index].shape[0]
        number_of_samples_in_group = len(group_sample_indices[class_index])
        biggest_marker_size = 500; smallest_marker_size = 50; step_marker_size = (biggest_marker_size-smallest_marker_size)/number_of_samples_in_group
        grouped_samples = grouped_samples_list[class_index]
        rank_of_grouped_samples_among_samples_of_class = rank_of_grouped_samples_among_all_list[class_index]
        for grouped_sample_index in range(grouped_samples.shape[0]):
            rank_among_all_samples_of_class = rank_of_grouped_samples_among_samples_of_class[grouped_sample_index]
            sample = grouped_samples[grouped_sample_index, :]
            marker_size=biggest_marker_size-(step_marker_size*rank_among_all_samples_of_class)
            color = color_variant(self, color_of_plot[class_index], brightness_offset=-5)
            plt.scatter(sample[0], sample[1], c=color, marker='s', s=marker_size, edgecolors='k')
    if save_figures:
        save_path = './PSA_outputs/'
        name_of_image = 'plot_ranking_samples_and_showing_qualified_samples'
        format_of_save = 'png'
        plt.savefig(save_path + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
    if show_images: plt.show()

def plot_final_ranked_samples(self, class_samples_list, Individual_ranks, color_of_plot, save_figures, show_images, name_image_to_save):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt1 = []
    number_of_classes = len(class_samples_list)
    for class_index in range(number_of_classes):
        samples = class_samples_list[class_index]
        number_of_samples = samples.shape[0]
        for sample_index in range(number_of_samples):
            number_of_samples_of_class = class_samples_list[class_index].shape[0]
            rank_among_all_samples_of_class = int(np.where(Individual_ranks[class_index][:] == sample_index)[0])  # find place of sample_index in Individual_ranks[class_index][:]
            biggest_marker_size = 500; smallest_marker_size = 1; step_marker_size = (biggest_marker_size-smallest_marker_size)/number_of_samples_of_class
            marker_size=biggest_marker_size-(step_marker_size*rank_among_all_samples_of_class)
            if rank_among_all_samples_of_class == number_of_samples_of_class-10:   # a small marker
                plt1 = plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='o', s=marker_size, edgecolors='k')
            else:
                plt.scatter(samples[sample_index,0], samples[sample_index,1], c=color_of_plot[class_index], marker='o', s=marker_size, edgecolors='k')
        # groupRanking = GroupRanking()
        # important_regions_of_distributions = groupRanking.find_important_regions_in_distibution(samples)
        # for dimension_index in range(len(important_regions_of_distributions)):
        #     for region_index in range(important_regions_of_distributions[dimension_index].shape[0]):
        #         important_region_point = important_regions_of_distributions[dimension_index][region_index]  # important_regions_of_distributions is a list, and important_regions_of_distributions[dimension_index] is an array
        #         plt.scatter(important_region_point[0], important_region_point[1], c='r', marker='x', s=100, linewidths=10)
        if type(plt1) is not list: plt1.set_label('Class ' + str(class_index))
        # ax.legend()  # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html     # https://stackoverflow.com/questions/40351026/plotting-a-simple-3d-numpy-array-using-matplotlib
        # fig.suptitle('Ranking Samples')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    if save_figures:
        save_path = './PSA_outputs/'
        name_of_image = name_image_to_save
        format_of_save = 'png'
        plt.savefig(save_path + name_of_image + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
    if show_images: plt.show()

def color_variant(self, rgb_color, brightness_offset=1):
    """ takes a color like rgb (its hex is #87c95f) and produces a lighter or darker variant """
    red = int(rgb_color[0][0] * 255); green = int(rgb_color[0][1] * 255); blue = int(rgb_color[0][2] * 255)
    hex_color = rgb_to_hex(self, red, green, blue)
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    darker_or_lighter_color_hex = "#" + "".join([hex(i)[2:] for i in new_rgb_int])
    # http://chase-seibert.github.io/blog/2011/07/29/python-calculate-lighterdarker-rgb-colors.html
    darker_or_lighter_color_rgb = hex_to_rgb(self, darker_or_lighter_color_hex)
    return np.array([[darker_or_lighter_color_rgb[0]/255, darker_or_lighter_color_rgb[1]/255, darker_or_lighter_color_rgb[2]/255]])

def hex_to_rgb(self, value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    # https://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
    # https://stackoverflow.com/questions/3380726/converting-a-rgb-color-tuple-to-a-six-digit-code-in-python

def rgb_to_hex(self, red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)