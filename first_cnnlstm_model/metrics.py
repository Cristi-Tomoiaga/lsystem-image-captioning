class AverageMetric:
    def __init__(self):
        self.__total = 0
        self.__running_sum = 0.0
        self.__average_value = 0.0
        self.__previous_average_value = 0.0

    def reset(self):
        self.__previous_average_value = self.__average_value
        self.__total = 0
        self.__running_sum = 0.0
        self.__average_value = 0.0

    @property
    def average_value(self):
        return self.__average_value

    @property
    def previous_value(self):
        return self.__previous_average_value

    def add_value(self, value, count=1):
        self.__total += count
        self.__running_sum += value * count
        self.__average_value = self.__running_sum / self.__total
