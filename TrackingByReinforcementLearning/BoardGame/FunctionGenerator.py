import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from win32api import GetSystemMetrics

RESOLUTION = [GetSystemMetrics(0) - 1, GetSystemMetrics(1) - 1]  # [Width, Height]


def vectorize_map(f):
    return np.vectorize(lambda var: f(var))


class MathFunction(ABC):
    @abstractmethod
    def evaluate(self):
        pass

    @classmethod
    def round_coordinates(cls, coordinates):
        v_round = vectorize_map(lambda x: int(x))
        return v_round(coordinates)

    @classmethod
    def get_coordinates(cls, fig, ax, points):
        x, y = points.get_data()
        xy_pixels = ax.transData.transform(np.vstack([x, y]).T)
        xpix, ypix = xy_pixels.T
        # In matplotlib, 0,0 is the lower left corner, so we need to flip the y-coords to the upper left
        _, height = fig.canvas.get_width_height()
        ypix = height - ypix
        return MathFunction.round_coordinates(xpix), MathFunction.round_coordinates(ypix)


#############################################################################
# Hands Functions
#############################################################################
class Sinus(MathFunction):
    def evaluate(self):
        A = np.random.uniform(low=1, high=3, size=(1,))[0] * np.random.choice([-1, 1])
        B = np.random.uniform(low=0.1, high=0.5, size=(1,))[0]

        def f(t):
            return (20*A/B)*np.sin((B*A/10)*t) + A*t

        t = np.arange(-100., 100., 4)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0]/96, RESOLUTION[1]/96), dpi=96)
        points, = ax.plot(t, f(t))
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]


class Linear(MathFunction):
    def evaluate(self):
        m = np.random.uniform(low=0.1, high=10, size=(1,))[0] * np.random.choice([-1, 1])
        b = np.random.randint(0, int(RESOLUTION[0]/8)) * np.random.choice([-1, 1])

        def f(t):
            return m*t

        t = np.arange(-5., 5., 0.3)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0]/96, RESOLUTION[1]/96), dpi=96)
        points, = ax.plot(t, f(t))
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        v_bias = vectorize_map(lambda x: x + b)
        return v_bias(res[0]), res[1]


class AbsLinear(MathFunction):
    def evaluate(self):
        m = np.random.uniform(low=0.1, high=10, size=(1,))[0] * np.random.choice([-1, 1])
        b = np.random.randint(0, int(RESOLUTION[0]/10)) * np.random.choice([-1, 1])
        sign = np.random.choice([-1, 1])

        def f(t):
            return np.abs(m*t)

        t = np.arange(-5., 5., 0.3)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0]/96, RESOLUTION[1]/96), dpi=96)
        points, = ax.plot(t, f(t) * sign)
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        v_bias = vectorize_map(lambda x: x + b)
        return v_bias(res[0]), res[1]


class Point(MathFunction):
    def evaluate(self):
        return [np.int32(RESOLUTION[0]/2)], [np.int32(RESOLUTION[1]/2)]


class HorizontalLineHand(MathFunction):
    def evaluate(self):
        def f(t):
            return t * 0

        t = np.arange(-4, 4, 0.2)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 96, RESOLUTION[1] / 96), dpi=96)
        plt.gca().set_aspect('equal', adjustable='box')
        points, = ax.plot(t[1:], f(t[1:]))
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]


class VerticalLineHand(MathFunction):
    def evaluate(self):
        def f():
            return np.arange(-4, 4, 0.4)

        t = np.zeros(20)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0] / 96, RESOLUTION[1] / 96), dpi=96)
        plt.gca().set_aspect('equal', adjustable='box')
        points, = ax.plot(t, f())
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]

#############################################################################
# Fingers Functions
#############################################################################
class HalfCircle(MathFunction):
    def evaluate(self):
        sign = np.random.choice([-1, 1])
        r = 4

        def f(t):
            return np.sqrt(np.power(r, 2) - np.power(t, 2))

        t = np.arange(-r, r, 0.8)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0] / (4*96), RESOLUTION[1] / (4*96)), dpi=96)
        plt.gca().set_aspect('equal', adjustable='box')
        points, = ax.plot(t[1:], f(t[1:]) * sign)
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]


class HorizontalLine(MathFunction):
    def evaluate(self):
        def f(t):
            return t*0

        t = np.arange(-4, 4, 1)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0] / (4*96), RESOLUTION[1] / (4*96)), dpi=96)
        plt.gca().set_aspect('equal', adjustable='box')
        points, = ax.plot(t[1:], f(t[1:]))
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]


class VerticalLine(MathFunction):
    def evaluate(self):
        def f():
            return np.arange(-4, 4, 1)

        t = np.zeros(8)

        fig, ax = plt.subplots(figsize=(RESOLUTION[0] / (4*96), RESOLUTION[1] / (4*96)), dpi=96)
        plt.gca().set_aspect('equal', adjustable='box')
        points, = ax.plot(t, f())
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]


class Corner(MathFunction):
    def evaluate(self):
        # Horizontal part
        x_h = np.arange(-5, 5, 1)
        y_h = np.zeros(10)

        # Vertical part
        x_v = np.zeros(3)
        y_v = np.arange(1, 4, 1)

        if np.random.rand(1) < 0.5:  # right
            x_v += 4
            if np.random.rand(1) < 0.5:  # up
                x = np.append(x_h, x_v)
                y = np.append(y_h, y_v)
            else:  # down
                x = np.append(x_h, x_v)
                y_v = y_v * (-1)
                y = np.append(y_h, y_v)

        else:  # left
            x_v -= 5
            if np.random.rand(1) < 0.5:  # up
                x = np.append(x_h[::-1], x_v)
                y = np.append(y_h, y_v)
            else:  # down
                x = np.append(x_h[::-1], x_v)
                y_v = y_v * (-1)
                y = np.append(y_h, y_v * (-1))

        fig, ax = plt.subplots(figsize=(RESOLUTION[0] / (4*96), RESOLUTION[1] / (4*96)), dpi=96)
        plt.gca().set_aspect('equal', adjustable='box')
        points, = ax.plot(x, y, "o-")
        res = self.get_coordinates(fig, ax, points)
        plt.close()
        return res[0], res[1]


#############################################################################
# Generators
#############################################################################
class FunctionGenerator(object):
    def __init__(self, fingers_functions=False):
        if fingers_functions:
            self.math_functions = [HalfCircle(), HorizontalLine(), VerticalLine(), Corner(), Point()]
        else:
            self.math_functions = [Sinus(), Linear(), AbsLinear(), Point(), VerticalLineHand(), HorizontalLineHand()]
        self.counter = 0
        self.current_f = None
        self.result = None

    def generate_function(self):
        if self.counter == 0:
            self.counter = 3
            self.current_f = np.random.choice(self.math_functions)
            self.result = self.current_f.evaluate()
        self.counter -= 1
        return self.result

if __name__ == '__main__':
    x, y = Corner().evaluate()
    for i in range(len(x)):
        print(x[i], y[i])
