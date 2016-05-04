import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class Droplet(object):

    def __init__(self):
        self._diameterUm = 5
        self._x = 0
        self._y = 0
        self._neighbourVectors = []
        self._innerGrid = False
        self._hitNS = True
        self._hitEW = True

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
    grid = base_grid + add_noise
    grid = grid * scale

    return grid

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

def visualiseGrid(grid):
    # plot grid
    plt.scatter(grid[:,0], grid[:,1]);
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":

    grid, trimmed_grid_idx = makeBaseGrid(10, 10)
    grid = changeGrid(grid, 0.01, 30)
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

    print(klabels)
    