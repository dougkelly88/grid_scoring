import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import csv
import os

try:
    from Tkinter import *
    from Tkinter.ttk import *
except:
    from tkinter import *

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

def chooseFile(initialdir, multipleFiles):
    try:
        import tkFileDialog
    except:
        from tkinter import filedialog as tkFileDialog

    root = Tk()
    try:
        options = {}
        options['initialdir']=initialdir
        options['filetypes'] = [('DROPLETSID', '.dropletsid') ]
        if not multipleFiles:
            options['title'] = 'Choose a time lapse image to run analysis on...'
            file = tkFileDialog.askopenfilename(**options)
        else:
            options['title'] = 'Choose one or more images to analyse...'
            file = tkFileDialog.askopenfilenames(**options)

        #print(file)
        
        if file == "":
            print("INVALID_PATH")
            exit(0)

    except IOError:
        print("INVALID_PATH")
        exit(0)

    root.destroy()
    return file

def makeBaseGrid(xsize, ysize):
    # make a regular square array of size [xsize ysize] as a base
    
    gridbase = np.empty([1, 2])
    #xsize = 10
    #ysize = 10

    for x in np.arange(0, xsize):
        for y in np.arange(0, ysize):
            gridbase = np.append(gridbase, [[x, y]], axis=0)

    gridbase = np.delete(gridbase, 0, 0)
    
    return gridbase

def changeGrid(base_grid, sd_noise, scale):
    # modify the base grid by adding noise and scaling to appropriate spacing

    add_noise = np.reshape(np.random.normal(0, sd_noise, 2*len(base_grid)), [len(base_grid), 2])
    #add_noise = np.reshape(np.random.random_integers(0, 1, 2*len(base_grid))*0.1, [len(base_grid), 2])
    grid = base_grid + add_noise
    grid = grid * scale
    theory_grid = base_grid * scale

    return grid, theory_grid

def removeDropletsFromEdgeByDistance(grid, margin_pixels):
    # identify droplets around the edge of a real grid that won't have [N,S,E,W]-nearest neighbours based on distance from min/max x/y coords. 

    trimmed_grid_idx = []
    xlim = [min(grid[:,0]) + margin_pixels, max(grid[:,0]) - margin_pixels]
    ylim = [min(grid[:,1]) + margin_pixels, max(grid[:,1]) - margin_pixels]

    for idx, el in enumerate(grid):
        x = el[0]
        y = el[1]
        
        if ((x > xlim[0]) & (x < xlim[1])):
            if ((y > ylim[0]) & (y < ylim[1])):
                trimmed_grid_idx.append(idx)

    return trimmed_grid_idx

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

def visualiseGrid(grid, droplet_r, title):
    # plot grid
    gridfig, gridax = plt.subplots()
    gridfig.canvas.set_window_title(title) 
    plt.scatter(grid[:,0], grid[:,1], c='r', marker="x", s=5)
    plt.title(title)
    for xy in grid:
        circ = plt.Circle((xy[0], xy[1]), radius = droplet_r, color=(1, 0, 0, 0.5))
        gridax.add_patch(circ)
    plt.axis('equal')

    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()           # works with QT4AGG, WXAGG BACKENDS ONLY!
    
    except:
        pass
    
    plt.show()
    print('saving figure...')
    gridfig.savefig(root_path + "/" + title + ".png", dpi=600)
    print('figure saved!')

def visualiseGridsRealTheory(realgrid, theorygrid, droplet_r, ds, title):
    print(np.ma.shape(realgrid))
    print(np.ma.shape(theorygrid))

    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title) 
    plt.title(title)
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

    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()           # works with QT4AGG, WXAGG BACKENDS ONLY!

    except:
        pass

    plt.show()
    print("saving figure...")
    fig.savefig(root_path + "/Real first print, ideal second print.png", dpi=600);
    print('figure saved!')

def generateMeanGrid(mean_vec1, mean_vec2, mean_vec3, mean_vec4, grid, mean_grid_xsz, mean_grid_ysz):

    print("mean_vec1 (%0.2f, %0.2f) " % (mean_vec1[0], mean_vec1[1]) )
    print("mean_vec2 (%0.2f, %0.2f) " % (mean_vec2[0], mean_vec2[1]) )
    print("mean_vec3 (%0.2f, %0.2f) " % (mean_vec3[0], mean_vec3[1]) )
    print("mean_vec4 (%0.2f, %0.2f) " % (mean_vec4[0], mean_vec4[1]) )

    mean_vecs = [mean_vec1, mean_vec2, mean_vec3, mean_vec4]

    print('mean_vecs')
    print(mean_vecs)

    # clunky means of ensuring that opposing vectors are paired appropriately - improve!
    idxs = [0,1,2]
    thetas = [math.atan(x[1]/x[0]) for x in mean_vecs]
    print('thetas')
    print(thetas)
    dthetas = [abs(theta) - abs(thetas[0]) for theta in thetas]
    
    dthetas.pop(0)
    print('dthetas')
    print(dthetas)
    pair_with_1_idx = np.argmin(abs(np.asarray(dthetas)))
    print('pair with 1 idx')
    print(pair_with_1_idx)
    pair_with_1_vec = mean_vecs[pair_with_1_idx+1]
    idxs.remove(pair_with_1_idx)
    dont_pair_with_1_vec1 = mean_vecs[idxs[0]+1]
    dont_pair_with_1_vec2 = mean_vecs[idxs[1]+1]

    print("mean_vec1 (%0.2f, %0.2f) " % (mean_vec1[0], mean_vec1[1]) )
    print("pair_with_1_vec (%0.2f, %0.2f) " % (pair_with_1_vec[0], pair_with_1_vec[1]) )
    print("dont_pair_with_1_vec1 (%0.2f, %0.2f) " % (dont_pair_with_1_vec1[0], dont_pair_with_1_vec1[1]) )
    print("dont_pair_with_1_vec2 (%0.2f, %0.2f) " % (dont_pair_with_1_vec2[0], dont_pair_with_1_vec2[1]) )

    ## should only take the abs value of the long component of the mean_vecs when calculating bases!
    #if (abs(mean_vec1[0]) > abs(mean_vec1[1])):
    #    basis1 = 0.5 * np.asarray([abs(mean_vec1[0]) + abs(mean_vec3[0]), mean_vec1[1] + mean_vec3[1]])
    #    basis2 = 0.5 * np.asarray([mean_vec2[0] + mean_vec4[0], abs(mean_vec2[1]) + abs(mean_vec4[1])])
    #else:
    #    basis1 = 0.5 * np.asarray([mean_vec1[0] + mean_vec3[0], abs(mean_vec1[1]) + abs(mean_vec3[1])])
    #    basis2 = 0.5 * np.asarray([abs(mean_vec2[0]) + abs(mean_vec4[0]), mean_vec2[1] + mean_vec4[1]])

    # should only take the abs value of the long component of the mean_vecs when calculating bases!
    if (abs(mean_vec1[0]) > abs(mean_vec1[1])):
        basis1 = 0.5 * np.asarray([abs(mean_vec1[0]) + abs(pair_with_1_vec[0]), mean_vec1[1] + pair_with_1_vec[1]])
        basis2 = 0.5 * np.asarray([dont_pair_with_1_vec1[0] + dont_pair_with_1_vec2[0], abs(dont_pair_with_1_vec1[1]) + abs(dont_pair_with_1_vec2[1])])
    else:
        basis1 = 0.5 * np.asarray([mean_vec1[0] + pair_with_1_vec[0], abs(mean_vec1[1]) + abs(pair_with_1_vec[1])])
        basis2 = 0.5 * np.asarray([abs(dont_pair_with_1_vec1[0]) + abs(dont_pair_with_1_vec2[0]), dont_pair_with_1_vec1[1] + dont_pair_with_1_vec2[1]])

    print("basis1 (%0.2f, %0.2f) " % (basis1[0], basis1[1]) )
    print("basis2 (%0.2f, %0.2f) " % (basis2[0], basis2[1]) )
    

    if (abs(basis1[0]) > abs(basis1[1])):
        xbasis = basis1
        ybasis = basis2
    else:
        xbasis = basis2
        ybasis = basis1

    nsAvDistance = np.linalg.norm(ybasis)
    print("NS average distance = %0.2f " % nsAvDistance)
    print("ybasis (%0.2f, %0.2f) " % (ybasis[0], ybasis[1]) )

    ewAvDistance = np.linalg.norm(xbasis)
    print("EW average distance = %0.2f " % ewAvDistance)
    print("xbasis (%0.2f, %0.2f) " % (xbasis[0], xbasis[1]) )
   
    origin = sum(grid)/len(grid)
    origin = grid[0,:]
    distance_from_zero = np.linalg.norm(grid, axis = 1)
    print(distance_from_zero[1:10])
    origin_idx = np.argmin(distance_from_zero)
    origin = grid[origin_idx, :]
    origin = origin - xbasis - ybasis
    print(origin)

    # construct a best fit grid from bases and origin...
    mean_grid = np.empty([1,2])
    mean_grid_row = np.empty([1,2])
    # build row
    for y in np.arange(0, mean_grid_ysz):
        mean_grid_row = np.append(mean_grid_row, [y*ybasis], axis=0)
    mean_grid_row = np.delete(mean_grid_row, 0, 0)
    for el in mean_grid_row:
        for x in np.arange(0, mean_grid_xsz):
            mean_grid = np.append(mean_grid, [el + x*xbasis], axis=0)
    mean_grid = np.delete(mean_grid, 0, 0)
    #mean_grid = mean_grid - sum(mean_grid)/len(mean_grid) + [origin]
    print(mean_grid[1:10,:])
    mean_grid = mean_grid + [origin]
    print(mean_grid[1:10,:])

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

def importFromDropletsID(filepath, margin_pixels):
    
    grid = np.empty([1, 2])
    rs = []

    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)    # skip first (header) row
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            r = float(row[2])

            grid = np.append(grid,  [[x, y]], axis=0)
            rs.append(r)

    grid = np.delete(grid, 0, 0)
    droplet_r = sum(rs)/len(rs)
    trimmed_grid_idx = removeDropletsFromEdgeByDistance(grid, margin_pixels)
    
    return grid, trimmed_grid_idx, droplet_r

def scoreThisGrid(grid_np_array, droplet_r, root_path):
    print('nonsense')


if __name__ == "__main__":

    droplet_r = 2.5
    array_size_x = 5
    array_size_y = 40
    array_pitch = 25
    root_path = os.environ['HOMEPATH'] + '\\Desktop' # WILL WORK ONLY UNDER WINDOWS!
    realOrSimulated = True  # TRUE for real data from dropletsid file, FALSE for simulated data

    if realOrSimulated:
        fpath = chooseFile(root_path, False)
        root_path, dummy = os.path.split(fpath)
        grid, droplet_r = importFromDropletsID(fpath)

    else:
        grid = makeBaseGrid(array_size_x, array_size_y)
        #trimmed_grid_idx = removeDropletsFromEdge(grid, 1)
        grid, theory_grid = changeGrid(grid, 0.01, array_pitch)

    # find extent of grid in short dimension
    lenX = max(grid[:,0]) - min(grid[:,0])
    lenY = max(grid[:,1]) - min(grid[:,1])
    shortLen = min(lenX, lenY)
            
    trimmed_grid_idx = removeDropletsFromEdgeByDistance(grid, shortLen/10)
    trimmed_grid = grid[trimmed_grid_idx]
    
    print('droplet_r = %f' % droplet_r)
    visualiseGrid(grid, droplet_r, 'Input grid')
    visualiseGrid(trimmed_grid, droplet_r, 'Trimmed grid')
    
    if not realOrSimulated:
        saveGridAsDropletsID(grid, root_path + "/dummy.dropletsid", droplet_r)
    
    #visualiseGridsRealTheory(grid, theory_grid, droplet_r)
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
    plt.title('Kmeans vectors')
    kmax.scatter(vectors[:,0], vectors[:,1], c=klabels.astype(np.float))
    plt.axis('equal')
    plt.show()
    print('saving figure...')
    kmfig.savefig(root_path + "/Vector clusters.png", dpi=600)
    print('figure saved!')
    #print(klabels)
    
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

    #mean_vec1 = sum(g1)/float(len(g1))
    #mean_vec2 = sum(g2)/float(len(g2))
    #mean_vec3 = sum(g3)/float(len(g3))
    #mean_vec4 = sum(g4)/float(len(g4))

    mean_vec1 = np.median(g1, axis=0)
    mean_vec2 = np.median(g2, axis=0)
    mean_vec3 = np.median(g3, axis=0)
    mean_vec4 = np.median(g4, axis=0)

    if realOrSimulated:
        # take vec1 and vec2, work out which is x and which is y
        if (abs(mean_vec1[0]) > abs(mean_vec1[1])):
            xvec = mean_vec1
            yvec = mean_vec2
        else:
            xvec = mean_vec2
            yvec = mean_vec1
        # approximate array sizes based on extent in x, y, and mean vector lengths
        array_size_x = int( (max(grid[:,0]) - min(grid[:,0]))/np.linalg.norm(xvec) )
        array_size_y = int( (max(grid[:,1]) - min(grid[:,1]))/np.linalg.norm(yvec) )

    print(array_size_x)
    print(array_size_y)
    mean_grid = generateMeanGrid(mean_vec1, mean_vec2, mean_vec3, mean_vec4, grid, array_size_x + 4, array_size_y + 4)
    

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
    #print(np.ma.shape(np.vstack((grid, mean_grid))))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(np.vstack((grid, mean_grid)))
    distances, indices =nbrs.kneighbors(grid)

    ds = distances[:,1]
    #print(ds)
    overlapScoreSum = 0;
    for d in ds:
        if d < 2 * droplet_r:
            overlapScoreSum = overlapScoreSum + 1;
            # might be better to do this adding printed positions to list, or otherwise tying successful merges with droplets
    overlapScorePercentage = 100 * overlapScoreSum/len(ds)

    print('Overlap score = %0.2d pc' % overlapScorePercentage)

    visualiseGridsRealTheory(grid, mean_grid, droplet_r, ds, ('Ideal second print overlaid on real first print,\n overlap = %0.2d%%, NS = %0.2d%%, EW = %0.2d%%' % (overlapScorePercentage, nsScorePercentage, ewScorePercentage)))

    # TODO: add same grid noise to the mean grid