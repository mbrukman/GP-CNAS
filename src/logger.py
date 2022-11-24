import numpy as np


class StatWriter(object):
    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []

    @staticmethod
    def accumulate_stat(min: float, max: float, mean: float, std: float):
        """
        Static method which accumulate minimum, maximum, average and standard-deviation of fitness values

        :param min: Minimum value
        :param max: Maximum value
        :param mean: Average value
        :param std: Standard-Deviation value
        """
        StatWriter.min_vals.append(min)
        StatWriter.max_vals.append(max)
        StatWriter.mean_vals.append(mean)
        StatWriter.std_vals.append(std)

    @staticmethod
    def print_global_statistics():
        """
        Static function that prints the max, min, avg and std of the accumulated statistics
        """
        print("\n#### Global Statistics ####")
        print(
            "Min: %.3f | Max: %.3f | Mean: %.3f | Std: %.3f\n"
            % (
                np.min(StatWriter.min_vals),
                np.max(StatWriter.max_vals),
                np.mean(StatWriter.mean_vals),
                np.std(StatWriter.std_vals),
            )
        )
