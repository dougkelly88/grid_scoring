import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import csv

class Droplet(object):

    def __init__(self):
        self._diameterUm = 5
        self._x = 0
        self._y = 0
        self._neighbourVectors = []
        self._innerGrid = False
        self._hitNS = True
        self._hitEW = True
        self.vectorGroup = 0
        self.deviations = [0, 0, 0, 0]

    # pythonic setters and getters
    @property
    def diameterUm(self):
        return self._diameterUm

    @diameterUm.setter
    def diameterUm(self, value):
        self._diameterUm = float(value)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = float(value)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = float(value)

    @property
    def neighbourVectors(self):
        return self._neighbourVectors

    @property 
    def innerGrid(self):
        return self._innerGrid

    @innerGrid.setter
    def innerGrid(self, value):
        self._innerGrid = value

    @property 
    def hitNS(self):
        return self._hitNS

    @hitNS.setter
    def hitNS(self, value):
        self._hitNS = value

    @property 
    def hitEW(self):
        return self._hitEW

    @hitEW.setter
    def hitEW(self, value):
        self._hitEW = value

    def setXY(grid_pos):
        self.x = grid_pos[0]
        self.y = grid_pos[1]

    def addVector(self, vector):
        self._neighbourVectors.append(vector)

def makeBaseGrid(xsize, ysize):
    # make a regular square array of size [xsize ysize] as a base
    
    gridbase = np.empty([1, 2])
    #xsize = 10
    #ysize = 10

    for x in np.arange(0, xsize):
        for y in np.arange(0, ysize):
            gridbase = np.append(gridbase, [[x, y]], axis=0)

    gridbase = np.delete(gridbase, 0, 0)

    trimmed_grid_idx = removeDropletsFromEdge(gridbase, 1)
    
    return gridbase, trimmed_grid_idx

def changeGrid(base_grid, sd_noise, scale):
    # modify the base grid by adding noise and scaling to appropriate spacing

    add_noise = np.reshape(np.random.normal(0, sd_noise, 2*len(base_grid)), [len(base_grid), 2])
    #add_noise = np.reshape(np.random.random_integers(0, 1, 2*len(base_grid))*0.1, [len(base_grid), 2])
    grid = base_grid + add_noise
    grid = grid * scale
    theory_grid = base_grid * scale

    return grid, theory_grid

def removeDropletsFromEdge(grid, margin):
    # identify droplets around the edge of integer base grid that won't have [N,S,E,W]-nearest neighbours
    
    trimmed_grid_idx = []
    xlim = [min(grid[:,0]) + margin - 1, max(grid[:,0]) - margin + 1]
    ylim = [min(grid[:,1]) + margin - 1, max(grid[:,1]) - margin + 1]

    for idx, el in enumerate(grid):
        x = el[0]
        y = el[1]
        
        if ((x > xlim[0]) & (x < xlim[1])):
            if ((y > ylim[0]) & (y < ylim[1])):
                trimmed_grid_idx.append(idx)

    return trimmed_grid_idx

def generateNearestNeighbours(grid, trimmed_grid):
    # loop through trimmed grid positions identifying nearest neighbours and removing self

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(grid)
    distances, indices =nbrs.kneighbors(trimmed_grid)

    vectors = np.empty([1,2])
    for idx, inds in enumerate(indices):
        for idxx, ind in enumerate(inds):
            if idxx > 0:
                d = grid[inds[0]] - grid[ind]
                vectors = np.append(vectors, [d], axis=0)

    vectors = np.delete(vectors, 0, 0)

    return distances, indices, vectors

def visualiseGrid(grid, droplet_r):
    # plot grid
    gridfig, gridax = plt.subplots()
    plt.scatter(grid[:,0], grid[:,1], c='r', marker="x", s=5)
    for xy in grid:
        circ = plt.Circle((xy[0], xy[1]), radius = droplet_r, color=(1, 0, 0, 0.5))
        gridax.add_patch(circ)
    plt.axis('equal')
    plt.show()
    gridfig.savefig("C:/Users/d.kelly/Desktop/dummy input grid.png", dpi=600)

def visualiseGridsRealTheory(realgrid, theorygrid, droplet_r, ds):
    print(np.ma.shape(realgrid))
    print(np.ma.shape(theorygrid))

    fig, ax = plt.subplots()
    plt.scatter(realgrid[:,0], realgrid[:,1], c='r', marker="x", s=5)
    plt.scatter(theorygrid[:,0], theorygrid[:,1], c='b', marker="x", s=5)
    plt.axis('equal')
    for xy, d in zip(realgrid, ds):
        circ = plt.Circle((xy[0], xy[1]), radius = droplet_r, color=(1, 0, 0, 0.5))
        ax.add_patch(circ)
        if (d > 2 * droplet_r):
            badcirc = plt.Circle((xy[0], xy[1]), radius = 2.5 * droplet_r, ec='k', fill=False)
            ax.add_patch(badcirc)
    for xy in theorygrid:
        circ = plt.Circle((xy[0], xy[1]), radius = droplet_r, color=(0, 0, 1, 0.5))
        ax.add_patch(circ)

    plt.show()
    print("Saving figure...")
    fig.savefig("C:/Users/d.kelly/Desktop/dummy.png", dpi=600);

def generateMeanGrid(mean_vec1, mean_vec2, mean_vec3, mean_vec4, grid, mean_grid_xsz, mean_grid_ysz):

    # define basis vectors from mean vectors, and work out the position of the grid centre
    basis1 = 0.5 * (abs(mean_vec1) + abs(mean_vec2))
    basis2 = 0.5 * (abs(mean_vec4) + abs(mean_vec3))

    nsAvDistance = np.linalg.norm(basis1)
    print("NS average distance = %0.2f " % nsAvDistance)

    ewAvDistance = np.linalg.norm(basis2)
    print("EW average distance = %0.2f " % ewAvDistance)
   
    origin = sum(grid)/len(grid)

    # construct a best fit grid from bases and origin...
    mean_grid = np.empty([1,2])
    mean_grid_row = np.empty([1,2])
    # build row
    for x in np.arange(0, mean_grid_xsz):
        mean_grid_row = np.append(mean_grid_row, [x*basis1], axis=0)
    mean_grid_row = np.delete(mean_grid_row, 0, 0)
    for el in mean_grid_row:
        for y in np.arange(0, mean_grid_ysz):
            mean_grid = np.append(mean_grid, [el + y*basis2], axis=0)
    mean_grid = np.delete(mean_grid, 0, 0)
    mean_grid = mean_grid - sum(mean_grid)/len(mean_grid) + [origin]

    return mean_grid

def saveGridAsDropletsID(grid, filepath, droplet_r):
    # export grid to *.dropletsid in order to compare with James B C# solution

     with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Version","8","DropletsMD5","QmSKBvA2lHb3+8W+TVjkzw==","B4ImageIsTopDown","True",""])
        ty = "Droplet"
        tog = "False"
        las = "L700nm"
        shift = "0:0"
        for xy in grid:
            row = (xy[0], xy[1], droplet_r, ty, tog, las, shift)
            print(row)
            writer.writerow(row)


if __name__ == "__main__":

    droplet_r = 2

    grid, trimmed_grid_idx = makeBaseGrid(10, 10)

    grid, theory_grid = changeGrid(grid, 0.1, 25)
    visualiseGrid(grid, droplet_r)
    saveGridAsDropletsID(grid, "C:/Users/d.kelly/Desktop/dummy.dropletsid", droplet_r)
    #visualiseGridsRealTheory(grid, theory_grid, droplet_r)
    trimmed_grid = grid[trimmed_grid_idx]
    distances, indices, vectors = generateNearestNeighbours(grid, trimmed_grid)
    
    # show nearest neighbour vectors
    #print(grid)
    #visualiseGrid(vectors)

    # perform K means clustering on vectors
    estimator = KMeans(n_clusters=4, n_init=10)
    a = estimator.fit(vectors)
    
    # show clustered vectors
    klabels = estimator.labels_
    kmfig, kmax = plt.subplots()
    kmax.scatter(vectors[:,0], vectors[:,1], c=klabels.astype(np.float))
    plt.axis('equal')
    plt.show()
    kmfig.savefig("C:/Users/d.kelly/Desktop/dummy vector clusters.png", dpi=600)
    print(klabels)
    
    g1 = []
    g2 = []
    g3 = []
    g4 = []

    for g, vec in zip(klabels, vectors):
        if g == 0:
            g1.append(vec)
        elif g == 1:
            g2.append(vec)
        elif g == 2:
            g3.append(vec)
        elif g == 3:
            g4.append(vec)

    #print(g1)
    #print(sum(g1)/float(len(g1)))

    mean_vec1 = sum(g1)/float(len(g1))
    mean_vec2 = sum(g2)/float(len(g2))
    mean_vec3 = sum(g3)/float(len(g3))
    mean_vec4 = sum(g4)/float(len(g4))

    mean_grid = generateMeanGrid(mean_vec1, mean_vec2, mean_vec3, mean_vec4, grid, 12, 12)
    

    vectorFromMeanGrid_g1 = g1 - mean_vec1
    vectorFromMeanGrid_g2 = g2 - mean_vec2
    vectorFromMeanGrid_g3 = g3 - mean_vec3
    vectorFromMeanGrid_g4 = g4 - mean_vec4

    distanceFromMeanGrid1 = []
    distanceFromMeanGrid2 = []
    distanceFromMeanGrid3 = []
    distanceFromMeanGrid4 = []
    
    for el in vectorFromMeanGrid_g1:
        distanceFromMeanGrid1.append(np.linalg.norm(el))
    for el in vectorFromMeanGrid_g2:
        distanceFromMeanGrid2.append(np.linalg.norm(el))
    for el in vectorFromMeanGrid_g3:
        distanceFromMeanGrid3.append(np.linalg.norm(el))
    for el in vectorFromMeanGrid_g4:
        distanceFromMeanGrid4.append(np.linalg.norm(el))

    nsScoreSum = 0;
    for n in distanceFromMeanGrid1:
        if n < 2 * droplet_r:
            nsScoreSum = nsScoreSum + 1

    for s in distanceFromMeanGrid3:
        if s < 2 * droplet_r:
            nsScoreSum = nsScoreSum + 1;

    nsScorePercentage = 100 * nsScoreSum/ (len(distanceFromMeanGrid1) + len(distanceFromMeanGrid3))

    ewScoreSum = 0;
    for ea in distanceFromMeanGrid2:
        if ea < 2 * droplet_r:
            ewScoreSum = ewScoreSum + 1

    for w in distanceFromMeanGrid4:
        if w < 2 *  droplet_r:
            ewScoreSum = ewScoreSum + 1;
            
    ewScorePercentage = 100 * ewScoreSum/ (len(distanceFromMeanGrid2) + len(distanceFromMeanGrid4))

    print('NS score = %0.2d pc' % nsScorePercentage)
    print('EW score = %0.2d pc' % ewScorePercentage)

    # try score based on overlap with mean grid (assumes perfect second print) - N.B. this won't work if the droplets are too erratic!
    print(np.ma.shape(np.vstack((grid, mean_grid))))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(np.vstack((grid, mean_grid)))
    distances, indices =nbrs.kneighbors(grid)

    ds = distances[:,1]
    print(ds)
    overlapScoreSum = 0;
    for d in ds:
        if d < 2 * droplet_r:
            overlapScoreSum = overlapScoreSum + 1;
            # might be better to do this adding printed positions to list, or otherwise tying successful merges with droplets
    overlapScorePercentage = 100 * overlapScoreSum/len(ds)

    print('Overlap score = %0.2d pc' % overlapScorePercentage)

    visualiseGridsRealTheory(grid, mean_grid, droplet_r, ds)

    # TODO: add same grid noise to the mean grid
    # TODO: save visualisations as high-res images
    # TODO: export "real" grid as *.dropletsid