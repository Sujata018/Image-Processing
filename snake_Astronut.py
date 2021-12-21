import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data,io
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from imageio import mimwrite

'''
Create an animation of a snake active near the face of an astronut.
It segments the face of an image of an astronut, using snake active contour model.
'''

## Read image of an Astronut and smooth it
img = data.astronaut()
img = rgb2gray(img)

blurr_img=gaussian(img, 3, preserve_range=False)

## Draw a contour around the face
s = np.linspace(0, 2*np.pi, 400)
r = 100 + 100*np.sin(s)
c = 220 + 100*np.cos(s)
init = np.array([r, c]).T

## Segment the face using snake model, and keep intermittent images for animation
snakes=[]

for n in range(1,75,2):

    snake = active_contour(blurr_img,init, alpha=0.015, beta=10, gamma=0.001,max_iterations=n)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    
    plt.savefig("Astronut_snake.png")
    oimg=io.imread("Astronut_snake.png")
    snakes += [oimg]
    plt.close()

## Create the animation
mimwrite("Astronut_snake_animated.gif",snakes,loop=1,fps=5)
