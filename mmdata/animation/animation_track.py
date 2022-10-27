import numpy as np
import mmdata.utils.quaternion_utils as quaternion_utils
from abc import ABC, abstractmethod
from mmdata.animation.interpolation import InterpolationMethod


class AnimationTrack(ABC):
    def __init__(self, name, times, values):
        self.name = name
        self.times = times
        self.values = values

    @abstractmethod
    def get_value(self, start_index, end_index, t) -> np.ndarray:
        pass


class SkeletalTrack(AnimationTrack):
    def __init__(self, name, times, values, interpolations, interpolation_method: InterpolationMethod):
        assert values.shape[0] / times.shape[0] == 1.0
        assert interpolations.shape[0] % values.shape[0] == 0
        times, values, interpolations = self.__process_interpolations(times, values, interpolations)

        super(SkeletalTrack, self).__init__(name, times, values)
        self.interpolations = interpolations
        self.interpolation_method = interpolation_method

    def __process_interpolations(self, times, values, interpolations):
        if times.shape[0] > 2:
            times = times.copy()
            values = values.copy()
            interpolations = interpolations.copy()

            interp_stride = interpolations.shape[0] // times.shape[0]
            index = 1
            # skip any frame that has similar value to its previous and next frame

            for ahead_index in range(2, times.shape[0]):
                if np.any(values[index] != values[index - 1]) or np.any(values[index] != values[ahead_index]):
                    index += 1

                if ahead_index > index:
                    times[index] = times[ahead_index]
                    values[index] = values[ahead_index]

                    for i in range(0, interp_stride):
                        interpolations[index * interp_stride + i] = interpolations[ahead_index * interp_stride + i]

            times = times[:(index + 1)]
            values = values[:(index + 1)]
            interpolations = interpolations[:((index + 1) * interp_stride)]
        return times, values, interpolations

    def get_value(self, start_index, end_index, t):
        start_time, end_time = self.times[start_index], self.times[end_index]
        start_value, end_value = self.values[start_index], self.values[end_index]
        ratio = (t - start_time) / (end_time - start_time)

        # for bones, use Bezier interpolation
        weight1 = 0.0 if ((end_time - start_time) < (1 / 30 * 1.5)) else ratio
        stride = self.interpolations.shape[0] // self.times.shape[0]

        if stride == 1:
            # quaternion
            int_params = self.interpolations[start_index]
            ratio = self.interpolation_method.compute_ratio(weight1, int_params)
            value = quaternion_utils.slerp(start_value, end_value, ratio)
            value[0] = -value[0]

        elif stride == 3:
            # position
            value = start_value.copy()
            for j in range(0, stride):
                int_params = self.interpolations[start_index * 3 + j]
                ratio = self.interpolation_method.compute_ratio(weight1, int_params)
                value[j] = start_value[j] * (1.0 - ratio) + end_value[j] * ratio
        else:
            raise ValueError("Invalid stride!!")
        return value


class MorphTrack(AnimationTrack):
    def get_value(self, start_index, end_index, t):
        start_time, end_time = self.times[start_index], self.times[end_index]
        start_value, end_value = self.values[start_index], self.values[end_index]
        ratio = (t - start_time) / (end_time - start_time)
        return start_value * (1.0 - ratio) + end_value * ratio
