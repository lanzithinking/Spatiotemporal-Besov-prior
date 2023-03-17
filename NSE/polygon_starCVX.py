"""
Generate random (star-convex) polygon
-------------------------------------
Mike Ounsworth
https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
"""
import numpy as np
import math, random
from typing import List, Tuple
from matplotlib.path import Path

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def polygon_mask(vertices, width=256, height=256):
    """
    Generate polygon mask
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    path = Path(vertices)
    mask = path.contains_points(points).reshape((height,width))
    return mask

if __name__ == '__main__':
    # np.random.seed(2023)
    import matplotlib.pyplot as plt
    
    # Set image size
    height = 256
    width = 256
    num_vertices = 10
    
    # generate vertices of polygon
    vertices = generate_polygon(center=(height//2, width//2),
                                avg_radius=50,
                                irregularity=0.35,
                                spikiness=0.4,
                                num_vertices=num_vertices)
    
    mask=[]
    # solution 1
    from PIL import Image, ImageDraw
    img = Image.new('L', (height, width), 0)
    draw = ImageDraw.Draw(img)
    # either use .polygon(), if you want to fill the area with a solid colour
    draw.polygon(vertices, outline=0, fill=1)
    # convert to mask
    mask.append(np.array(img))
    
    # solution 2
    from matplotlib.path import Path
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    path = Path(vertices)
    mask.append(path.contains_points(points).reshape((height,width)))
    
    # draw
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(10,4))
    for i,ax in enumerate(axes.flat):
        plt.axes(ax)
        ax.imshow(mask[i], origin='lower', cmap='gray')
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()
    