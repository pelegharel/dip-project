
# coding: utf-8

# In[1]:


from functools import reduce
from operator import or_, add
from random import randint
from itertools import product
from collections import deque

import cv2
from matplotlib.pyplot import imshow, figure
import numpy as np
from numpy import array, flip

# pylint: disable=undefined-variable
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Morphing

# In[2]:


def cross(size):
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

def circle(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def cvclose(image, kernel):
    return cv2.erode(cv2.dilate(image, kernel), kernel)


# # m-adjacent neighbors <a id='m-adjacent'></a>
# 
#  For a given image $I$ and a mask $M$, pixel $p, q$ are considered m-adjacent if one of the following is true
#  
#   1. $q$ is a 4-adjcent of $p$ where $p,q\in M$
#   2. $q$ is diagonal of $q$ where $p, q\in M$ *and*
#      there is no $\omega\in M$ where $\omega$  is both 4-adjecent of $p$ and 4-adjecent of $q$
# 

# In[3]:


def is_in_image(pixel, shape):
    r, c = pixel
    rows, cols = shape
    return ((0 <= r < rows) and
            (0 <= c < cols))

def adjesent_m(pixel, mask):
    def is_in(pixel):
        r, c = pixel
        return is_in_image((r, c), mask.shape) and mask[r, c]
    
    def addp(offset):
        px, py = pixel
        ox, oy = offset
        return (px + ox, py + oy)
    
    neighbors_4 = [offset
                   for offset in [(1, 0), (0, 1), (-1, 0), (0, -1)]
                   if is_in(addp(offset))]
    
    neighbors_diag = [(o_r, o_c)
                      for o_r, o_c in [(1, 1), (-1, 1), (-1, -1), (1, -1)]
                      if set([(0, o_r), (o_c, 0)]).isdisjoint(neighbors_4) and is_in(addp((o_r, o_c)))]
    
    return [addp(offset) for offset in neighbors_4 + neighbors_diag]


# ## Convertions

# In[4]:


def uint8(image):
    return image.astype(np.uint8)

def arrayuint8(rows):
    return uint8(array(rows))


# ## Display

# In[5]:


def imshow_gray(image, figsize=(50, 50)):
    figure(figsize=figsize)
    imshow(image, cmap='gray')
    
def as_display(image):
    return cv2.cvtColor(cv2.normalize(image.astype(np.float),
                                      None,
                                      0,
                                      255,
                                      cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLOR_GRAY2RGB)

def rand_color():
    return (randint(150, 200), randint(150, 200), randint(0, 255))

def show_lines(image, lines):
    image_with_lines = as_display(image)
    for p1, p2 in lines:
        cv2.line(image_with_lines, p1, p2, rand_color(), 1)
        
    return image_with_lines

def show_points(image, points, radius=2):
    image_with_circles = as_display(image)
    for point in points:
        cv2.circle(image_with_circles, point, radius, (100, randint(150, 200), randint(0, 255)), thickness=-1)
    return image_with_circles


# # input document

# In[6]:


text = cv2.threshold(src=cv2.imread("arabic.jpg", cv2.IMREAD_GRAYSCALE),
                     thresh=200,
                     maxval=1,
                     type=cv2.THRESH_BINARY)[1]

imshow_gray(text)


# # Distance transform

# In[7]:


dist = cv2.distanceTransform(text, cv2.DIST_L2, cv2.DIST_MASK_5)
imshow_gray(dist)


# # Local maxima
# 
# Each pixel $p$ is consider local maximum if $p > q_1 \wedge p> q_2$
# 
# for some $q_1, q_1$ where $q_1, q_2$ are opposite pixels in the 8-member inviroment of $p$
# 
# Possible arrangements:
# 
# \begin{equation}
#  \begin{pmatrix}
#  -   & - & -\\
#  q_2 & p & q_1\\
#  -   & - & -
#  \end{pmatrix},
#  \begin{pmatrix}
#  -   & - & q_1\\
#  -   & p & -\\
#  q_2 & - & -
#  \end{pmatrix},
#  \begin{pmatrix}
#  - & q_1 & -\\
#  - & p & -\\
#  - & q_2 & -
#  \end{pmatrix},
#  \begin{pmatrix}
#  q_1 & - & -\\
#  -   & p & -\\
#  -   & - & q_2
#  \end{pmatrix}
# \end{equation}

# In[8]:


def local_maxima(image):
    horizontal = arrayuint8(
        [[0, 0, 0],
         [1, 0, 1],
         [0, 0, 0]])

    diagonal = arrayuint8(
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 1]])

    kernels = [horizontal, horizontal.T, diagonal, flip(diagonal, 1)]

    local_maximas = (image > cv2.dilate(image, kernel)
                     for kernel in kernels)

    return uint8(reduce(or_, local_maximas))


# In[9]:


local_dist_maxima = cvclose(local_maxima(dist), cross(3))

imshow_gray(local_dist_maxima, figsize=(50, 50))


# # Vertices
# 

#  ## Neighbor counting
#  

# In[10]:


def rotations(matrix):
    return [matrix, matrix.T, flip(matrix, 0), flip(matrix.T, 1)]

def neighbor_count(binary_matrix):
    down = arrayuint8(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]])

    down_right = arrayuint8(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0]])

    down_up_right = arrayuint8(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0]])


    sides = zip(*map(rotations, (down,
                                 down_right,
                                 flip(down_right, 1),
                                 down_up_right,
                                 flip(down_up_right, 1)
                                )))

    neighbors = [uint8(reduce(or_, (cv2.erode(binary_matrix, kernel)
                                    for kernel in kernels)))
                 for kernels in sides]

    res = reduce(add, neighbors)

    res[0, :] = 0
    res[:, 0] = 0
    res[-1, :] = 0
    res[:, -1] = 0
    return res

def vertices(skeleton):
    vertex_pixels = uint8((neighbor_count(skeleton) > 2))
    centeroids = cv2.connectedComponentsWithStats(cv2.dilate(vertex_pixels, cross(3)))[-1]
    return [tuple(map(int, point)) for point in centeroids[1:-1]]


local_max_verts = vertices(local_dist_maxima)
figure(figsize=(50, 50))
imshow(show_points(local_dist_maxima, local_max_verts, 2))


#  # Edges

# ## BFS
# 
# [Another Cell](#m-adjacent)

# In[11]:


def area_to_vert(verts, radius):
    return dict((point, (x, y))
                for (x, y) in verts
                for point in product(range(x - radius, x + radius),
                                     range(y - radius, y + radius)))


# In[12]:


def edges_scan(search_mask, get_vert, start_points):
    edges = set()
    while start_points:
        start_vert = start_points.pop()
        q = deque([(start_vert, start_vert)])
        while q:
            pixel, vert = q.popleft()
            next_neighbors = [(n_pixel, get_vert.get(n_pixel, vert))
                              for n_pixel in adjesent_m(pixel, search_mask)]
            
            for (n_x, n_y), _ in next_neighbors:
                search_mask[n_x, n_y] = 0
                
            q.extend(next_neighbors)
            edges = edges.union((vert, n_vert) for _, n_vert in next_neighbors if vert != n_vert)
    
    return [((x1, y1), (x2, y2)) for (y1, x1), (y2, x2) in edges]

row_index_verts = [(y, x) for x, y in local_max_verts]

cvedges = edges_scan(
    search_mask=local_dist_maxima.copy(),
    get_vert=area_to_vert(row_index_verts, 5),
    start_points=set(row_index_verts))

imshow(show_lines(text, cvedges), figure=figure(figsize=(50, 50)))

