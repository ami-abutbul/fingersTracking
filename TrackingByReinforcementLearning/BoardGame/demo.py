# import numpy as np
# import matplotlib.pyplot as plt
#
# #
# # # # plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
# # # fig, ax = plt.subplots(figsize=(1920/96, 1080/96), dpi=96)
# # # points, = ax.plot(range(10), 'ro')
# # # ax.axis([-1, 10, -1, 10])
# # #
# # # # Get the x and y data and transform it into pixel coordinates
# # # x, y = points.get_data()
# # # xy_pixels = ax.transData.transform(np.vstack([x,y]).T)
# # # xpix, ypix = xy_pixels.T
# # #
# # # # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper
# # # # right for most image software, so we'll flip the y-coords...
# # # width, height = fig.canvas.get_width_height()
# # # print("###### {}, {}".format(width, height))
# # # ypix = height - ypix
# # #
# # # print('Coordinates of the points in pixel coordinates...')
# # # for xp, yp in zip(xpix, ypix):
# # #     print('{x:0.2f}\t{y:0.2f}'.format(x=xp, y=yp))
# # #
# # # # We have to be sure to save the figure with it's current DPI
# # # # (savfig overrides the DPI of the figure, by default)
# # # fig.savefig('test.png', dpi=fig.dpi)
# # # plt.show()
# # #
# # #
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # #
# # #
# # def get_coordinates(fig, ax, points):
# #     x, y = points.get_data()
# #     xy_pixels = ax.transData.transform(np.vstack([x, y]).T)
# #     xpix, ypix = xy_pixels.T
# #
# #     # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper
# #     # right for most image software, so we'll flip the y-coords...
# #     width, height = fig.canvas.get_width_height()
# #     # print("###### {}, {}".format(width, height))
# #     ypix = height - ypix
# #
# #
# #     print('Coordinates of the points in pixel coordinates...')
# #     for xp, yp in zip(xpix, ypix):
# #         print('{x:0.2f}\t{y:0.2f}'.format(x=xp, y=yp))
# #     return xpix, ypix
# #
# #
# # def sinus():
# #     A = 5
# #     B = 0.2
# #
# #     def f(t):
# #         return (20/B)*(A)*np.sin(B*(A/10)*t) + A*t
# #
# #     t1 = np.arange(-100., 100., 4.4)
# #
# #     fig, ax = plt.subplots(figsize=(1919/96, 1079/96), dpi=96)
# #     points, = ax.plot(t1, f(t1), 'bo-')
# #     # ax.axis([-101, 101, f(-100) - 1, f(100) + 1])
# #
# #     # fig = plt.figure(1, figsize=(1919/96, 1079/96))
# #     # ax = plt.axis([-101, 101, f(-100) - 1, f(100) + 1])
# #     # points, _ = plt.plot(t1, f(t1), 'bo-')
# #     res = get_coordinates(fig, ax, points)
# #     plt.close()
# #     return res
# # #
# # #     plt.show()
# # #
# # #
# # def linear():
# A = 0.1
#
# def f(t):
#     return A*t
#
# t1 = np.arange(-5., 5., 0.3)
#
# plt.figure(1, figsize=(1829 / 96, 899 / 96))
# plt.axis([-6, 6, f(-5) - 1, f(5) + 1])
# plt.plot(t1, f(t1), 'bo-')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
# # #
# # #
# # # # import pyautogui
# # # # while True:
# # # #     print(pyautogui.position())
# # #
# # # # pyautogui.moveTo(100, 150)
# # # # pyautogui.moveRel(0, 10)  # move mouse 10 pixels down
# # # # pyautogui.dragTo(100, 150)
# # # # pyautogui.dragRel(0, 10)  # drag mouse 10 pixels down
# # #
# # # # if __name__ == '__main__':
# # # #     sinus()
# #
# # # Define the colors we will use in RGB format
# # BLACK = (  0,   0,   0)
# # WHITE = (255, 255, 255)
# # BLUE =  (  0,   0, 255)
# # GREEN = (  0, 255,   0)
# # RED =   (255,   0,   0)
# #
# # import pygame
# #
# # pygame.init()
# # screen = pygame.display.set_mode((0, 0), pygame.NOFRAME|pygame.FULLSCREEN)
# # screen.fill(WHITE)
# # done = False
# #
# # x_points, y_points = sinus()
# # p_px = x_points[0]
# # p_py = y_points[0]
# # for xp, yp in zip(x_points, y_points):
# #     pygame.draw.circle(screen, BLUE, [int(xp), int(yp)], 5)
# #     pygame.draw.line(screen, GREEN, [int(p_px), int(p_py)], [int(xp), int(yp)], 3)
# #     p_px = xp
# #     p_py = yp
# #
# # while not done:
# #     for event in pygame.event.get():
# #         # pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(30, 30, 60, 60))
# #         # screen.set_at((500, 500), (128, 128, 128))
# #         if event.type == pygame.MOUSEBUTTONUP:
# #             done = True
# #
# #     pygame.display.flip()
#


import numpy as np
import matplotlib.pyplot as plt

def evaluate():
    # Horizontal part
    x_h = np.arange(-5, 5, 1)
    y_h = np.zeros(10)

    # Vertical part
    x_v = np.zeros(5)
    y_v = np.arange(0, 5, 1)

    if np.random.rand(1) < 0.5:  # right
        x_v += 5
        if np.random.rand(1) < 0.5:  # up
            x = np.append(x_h, x_v)
            y = np.append(y_h, y_v)
        else:  # down
            x = np.append(x_h, x_v)
            y = np.append(y_h, y_v * (-1))

    else:  # left
        x_v -= 5
        if np.random.rand(1) < 0.5:  # up
            x = np.append(x_v, x_h)
            y = np.append(y_v, y_h)
        else:  # down
            x = np.append(x_v, x_h)
            y = np.append(y_v * (-1), y_h)

    fig, ax = plt.subplots(figsize=(1829 / (3 * 96), 899 / (3 * 96)), dpi=96)
    plt.plot(x, y, 'bo-')

    # fig, ax = plt.subplots(figsize=(1829/(3*96), 899/(3*96)), dpi=96)
    # plt.plot(t[1:], f(t[1:]), 'bo-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    #
    # points, = ax.plot(t, f(t))
    # res = self.get_coordinates(fig, ax, points)
    # plt.close()

if __name__ == '__main__':
    evaluate()
