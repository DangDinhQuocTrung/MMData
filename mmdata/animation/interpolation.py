from abc import ABC, abstractmethod


class InterpolationMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_ratio(self, weight: int, params: [int]):
        """
        Get correct in-between ratio.
        :param weight:
        :param params:
        :return: ratio
        """
        return 0.0


class BezierInterpolationMethod(InterpolationMethod):
    """
    Only Bezier is supported.
    """
    def __init__(self, loop=15, eps=1e-5):
        super(BezierInterpolationMethod, self).__init__()
        self.loop = loop
        self.eps = eps

    def compute_ratio(self, weight: int, params: [int]):
        x = weight
        x1, x2, y1, y2 = params

        c = 0.5
        t = c
        s = 1.0 - t
        sst3, stt3, ttt = 0.0, 0.0, 0.0

        for i in range(0, self.loop):
            sst3 = 3.0 * s * s * t
            stt3 = 3.0 * s * t * t
            ttt = t * t * t
            ft = (sst3 * x1) + (stt3 * x2) + ttt - x

            if abs(ft) < self.eps:
                break

            c /= 2.0
            t += c if (ft < 0) else - c
            s = 1.0 - t
        return (sst3 * y1) + (stt3 * y2) + ttt
