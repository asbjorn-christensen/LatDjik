#!/usr/bin/env python

##############################################################################
#   --------------------------------------------------------------------------
#   LatDjik: Identify minimum travel length path via a regular finite lattice
#   --------------------------------------------------------------------------
#
#   Apply regular finite 2D lattice as support for optimal path resulution
#   Filter lattice points, based on topography
#   Can easily be adapted to other cost metrics to identify minimum cost paths
#
#   Based on https://en.wikipedia.org/wiki/Dijkstra's_algorithm
#   --------------------------------------------------------------------------
#   Version 23Feb2023: validated in simple 12x15 lattice with center obstackle
#   and GEBCO sub map. This version only features direct contact (edge/corner)
#   connectivity on lattice. This makes it difficult to represent curved paths,
#   as only 45deg direction change is possible, triggering weave instability/degeneracy
#   on optimal paths (ABABABA has same length as AAAABBB or BBBAAAA), since
#   optimal paths are variational. Adding longer lattice connectivity is believed to reduce
#   this by creating more atomic path angles
#
#   24 Feb 2023: extended to two shells, code prepared for higher shells
#   06 Mar 2023: fixed issue with loops over lists that changes during loops
#                fixed recursion nuisance in __str__
#   TODO: check for self-connect/self-detach
#   --------------------------------------------------------------------------
#   Author  == Asbjorn Christensen (asc@aqua.dtu.dk)   Feb 24, 2023
#   License == LGPL (https://www.gnu.org/copyleft/lesser.html)
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt # LatticeNodes.plot_path


# ------------------------------------------------------------------------
# Generate neighbor lattice indices to (ix,iy) on confined finite lattice
# of dimension (nx,ny). Take edge effect into account, so edge/corner
# points has fewer neighbors.
# The result is a list with elements like [(kx0,kx1,jx), (ky0,ky1,jy...)]
# where (jx,jy) is the terminal lattice neighbor, and (kx0,ky0), (kx1,ky1), ...
# are intermediate neighbor cells that are crossed from (ix,iy) to (jx,jy)
# Corner touching is not counted as crossing. No intermediate neighbor cells are 
# reported if (ix,iy) and (jx,jy) are direct neighbors.
# This function does not check wetness (legality) of sites (kx0,ky0), (kx1,ky1), ..., (jx,jy)
# This return format allows to use numpy fancy extraction like
# validmap[ (kx0,kx1,..., jx), (ky0,ky1,..., jy) ]
# to pick cell set transversed and check wetness (legality)
# In higher shells (shell > 1), include only sites giving new path angles
# ------------------------------------------------------------------------
def get_neighbor_indices(ix,iy, nx,ny, shells=2):
    assert 0 <= ix < nx   # require indices are on lattice 
    assert 0 <= iy < ny   # require indices are on lattice
    assert 0 < shells <=2  # current implementation range
    nn = []
    # ------------------------ shell 1 ------------------------
    # --- Eastern sides
    if ix < nx-1: # not at Eastern edge
        nn.append( [(ix+1,), (iy,)] )
        if iy < ny-1: nn.append( [(ix+1,), (iy+1,)] ) # not at Northern edge
        if iy > 0:    nn.append( [(ix+1,), (iy-1,)] ) # not at Southern edge
    # --- Western sides
    if ix > 0:    # not at Western edge
        nn.append( [(ix-1,), (iy,)] )
        if iy < ny-1: nn.append( [(ix-1,), (iy+1,)] ) # not at Northern edge
        if iy > 0:    nn.append( [(ix-1,), (iy-1,)] ) # not at Southern edge
    # --- North / South steps
    if iy < ny-1: nn.append( [(ix,), (iy+1,)] ) # not at Northern edge
    if iy > 0:    nn.append( [(ix,), (iy-1,)] ) # not at Southern edge
    #
    if shells == 1: return nn
    # ------------------------ shell 2 ------------------------
    # --- Eastern side
    if ix < nx-2: # away from Eastern edge
        if iy < ny-1: nn.append([(ix+1, ix+1, ix+2), (iy, iy+1, iy+1)]) # ENE site
        if iy > 0:    nn.append([(ix+1, ix+1, ix+2), (iy, iy-1, iy-1)]) # ESE site
    # --- Western side
    if ix > 1:    # away from Western edge
        if iy < ny-1: nn.append([(ix-1, ix-1, ix-2), (iy, iy+1, iy+1)]) # WNW site
        if iy > 0:    nn.append([(ix-1, ix-1, ix-2), (iy, iy-1, iy-1)]) # WSW site
    # --- North side
    if iy < ny-2: # away from  Northern edge
        if ix < nx-1: nn.append([(ix , ix+1, ix+1), (iy+1, iy+1, iy+2)]) # NNE site
        if ix > 0:    nn.append([(ix , ix-1, ix-1), (iy+1, iy+1, iy+2)]) # NNW site
    # --- South side
    if iy > 1:    # away from  Southern edge
        if ix < nx-1: nn.append([(ix , ix+1, ix+1), (iy-1, iy-1, iy-2)]) # SSE site
        if ix > 0:    nn.append([(ix , ix-1, ix-1), (iy-1, iy-1, iy-2)]) # SSW site    
    #
    if shells == 2: return nn
    #
    # --------------- further shells not implemented ---------------
    #

    
# ------------------------------------------------------------------------
# Determine the distance[km] between longitude[degE], latitudes[degN]
# (x0,y0) and (x1,y1) on idealized Earth sphere
# 0 < arccos < pi
# v = (cosx*cosy, cosx*siny, siny)
# ------------------------------------------------------------------------
earth_radius  = 6371.0      # km
deg2rad       = np.pi/180   # rad/deg
def sphere_distance(x0,y0,x1,y1):
    dlam   = deg2rad*(x1-x0)
    lamsum = deg2rad*(x1+x0)
    dphi   = deg2rad*(y1-y0)
    phisum = deg2rad*(y1+y0)
    v0xv1  = 0.5*(np.cos(dphi)*(1+np.cos(dlam)) - np.cos(phisum)*(1-np.cos(dlam))) # may be Taylor expanded
    v0xv1  = max(-1, min(v0xv1, 1)) # stricktly confine to [-1;1]
    return earth_radius*np.arccos(v0xv1)


# ================================================================================================
#    Represent a static finite regular 1D grid (with at least two points)
#
#    Not necessarely ascenting order. Assume homogeneous on input. input grid argument mandatory
#    Lattice points are cell centers. Apply tuple super lcass to ensure immutability of grid
#    Attributes:
#         dx: grid spacing (signed)
#    
# ================================================================================================
class Regular1DGrid(tuple):
    def __new__(self, xpts):
        return tuple.__new__(self, xpts)
    def __init__(self, xpts):
        self.dx = self[1]-self[0] # lattice spacing, 
    def grid_coordinate(self, x):
        return (x-self[0])/self.dx # unrestricted coordinate
    def cell_index(self, x):
        ix = int(round(self.grid_coordinate(x)))
        assert 0 <= ix < len(self)  # must be interior point
        return ix
    # generate corresponding boundary points 
    def boundary_points(self):
        return self[0] + self.dx*(-0.5 + np.arange(len(self)))
    
# ================================================================================================
#    Represent a node in a network (lattice, source and target)
#
#    Attributes:
#       pos:        x,y (not necessarely lattice points)
#       neighbors:  list of direct neighbors; Node instances. Possibly empty
#       nndist:     distances from pos to corresponding neighbors (nndist and neighbors synced)
#       dist2src:   current max distance from source to this node (search auxillary attribute, initially sys.maxsize)
#       parent:     parent node instance in minimal path through this point (search auxillary, initially None)
#       visited:    whether this node has been visited in the Djikstra algorithm (search auxillary, initially False)
# ================================================================================================
class Node:
    def __init__(self, pos=(None,None)):
        self.pos       = pos    # x,y (not necessarely lattice points)
        self.neighbors = []     # Node instances
        self.nndist    = []     # distances corresponding to neighbors (synced lists)
        self.reset()            # clear search history
        
    # connect two nodes mutually. Do nothing, if already connected or attempting self-connection
    # costmetric assumed symmetric
    def connect(self, other, costmetric = sphere_distance): #
        if self in other.neighbors or self is other:  # no self connection or duplication
            return # do nothing
        else:
            self.neighbors.append(other)              # connect mutually
            other.neighbors.append(self)              # connect mutually
            dist = costmetric(*self.pos, *other.pos)  # assumed symmetric, sphere_distance module function
            self.nndist.append(dist)                  # synced to neighbors 
            other.nndist.append(dist)                 # synced to neighbors 
            
    # disconnect two nodes mutually. Do nothing, if not connected
    # assume properly connected - otherwise intrinsic exceptions are raised
    # do not accept an attemp to detach self-connection
    def detach(self, other):
        assert self is not other
        if self not in other.neighbors:
            return # do nothing
        else:
            #
            ise = other.neighbors.index(self)
            del other.neighbors[ise]
            del other.nndist[ise] # synced to neighbors 
            # 
            iot = self.neighbors.index(other)
            del self.neighbors[iot]
            del self.nndist[iot]  # synced to neighbors 

            
    # reset search history variables
    def reset(self):    
        self.dist2src  = sys.maxsize  # current min distance from source to this node
        self.parent    = None         # parent node instance in minimal path
        self.visited   = False        # whether node has been visited in this Djikstra run

    # avoid recursion via parent reference by reporting memory address using id()    
    def __str__(self):
        tmp   = "Node instance (@mem %d):\n  pos=(%f,%f)\n  #neighbors=%d\n  nndist=%s\n  dist2src=%f\n  visited=%s\n"
        asstr = tmp % (id(self), self.pos[0], self.pos[1], len(self.neighbors),
                      self.nndist, self.dist2src, self.visited)
        if self.parent is None:
            asstr += "No parent\n"
        else:
            asstr += "Parent node @mem %d\n" % id(self.parent)
        return asstr
        
# ================================================================================================
#    Represent set of valid nodes on a regular lattice
#
#    validmap[i,j] >= 0 for valid lattice points; 
#    validmap[i,j] <  0 for invalid lattice points
#    validmap (integer numpy matrix like) contains consequtive integer sequence 0, 1, ..., nds-1 corresponding to node indices
#    only step one neighbor direct relations are allowed (i.e. up to 8 connections)
#    xgrid, ygrid are Regular1DGrid instances (or arrays)
#    shells is the number of neighbors shells to connect
# ================================================================================================       
class LatticeNodes(list):
    # ----------------------------------------------------------------------------------
    # Constructor for LatticeNodes
    # Create nodes corresponding to validmap and create node connectivity
    # ----------------------------------------------------------------------------------
    def __init__(self, validmap, xgrid, ygrid, shells=2):
        #
        nx = len(xgrid)
        ny = len(ygrid)
        assert validmap.shape == (nx,ny)
        self.validmap = validmap # keep for potential plotting 
        self.xgrid    = xgrid    # needed in run_Djikstra
        self.ygrid    = ygrid    # needed in run_Djikstra
        self.source   = None
        self.target   = None
        #
        # --- initialize lattice nodes corresponding to validmap
        #     first without connections, because all positions need to be defined first
        #
        nds  = np.amax(validmap) # integer
        ipts = 0 
        for ix in range(nx):
            for iy in range(ny):
                if validmap[ix,iy] >= 0: # valid lattice point
                    self.append(Node((xgrid[ix], ygrid[iy]))) # pass position
                    ipts += 1
        assert nds == ipts-1 # basic check on validmap integer consecutivity
        #
        # --- now connect up nodes, based on neighbor relations in validmap
        #
        for ix in range(nx):
            for iy in range(ny):
                ino = validmap[ix,iy]
                if ino >= 0: # valid lattice point - check paths to neighbor sites
                    for [xpath, ypath] in get_neighbor_indices(ix,iy,nx,ny, shells):   #
                        valueset = validmap[xpath, ypath]
                        OKpath   = all(valueset >= 0)      # check legality of path (ix,iy) -> neighbor
                        (jx,jy)  = (xpath[-1], ypath[-1])  # terminal site indices, which should be connected
                        jno = validmap[jx,jy]              # terminal site number
                        if OKpath and jno > ino:           # valid neighbor path to terminal site, avoid duplicate connect requests
                            self[ino].connect(self[jno])   # create mutual connection (no action if already exist)

                            
    # ----------------------------------------------------------------
    # Run Djikstra algorithm for source to target via lattice self
    #
    # if source/target are on lattice, node copies should be used
    # return distance from source to target
    # return None, if unconnected (on different sub lattices)
    # updates lattice node search variables
    #
    # Notice that run_Djikstra works by splicing in source and target
    # into lattice as independent nodes - therefore it is important
    # to apply reset AFTER optimization results has been collected
    # to detach source and target from the lattice (and clean up all
    # nodes in self, source and target)
    # ----------------------------------------------------------------
    def run_Djikstra(self, source, target):
        if source is target:
            return 0  # conclude optimization without calculation
        #
        self.reset()                  # prepare search variable for new run
        self.source = source          # keep reference, so they can be disconnected
        self.target = target          # keep reference, so they can be disconnected
        self.connect_new_node(source) # connect source to lattice
        self.connect_new_node(target) # connect target to lattice
        source.dist2src = 0
        next_nodes = PriorityList([source]) # local auxillary, not stored
        # ---- run scan until next_nodes is exhausted or current is target
        istep = 0
        #
        #   --------- main loop ---------
        #
        while len(next_nodes) > 0:
            current       = next_nodes[0]  
            next_nodes[:] = next_nodes[1:]  # remove current; must have [:], otherwise slicing produces a pure list
            # --- loop over neighbors of current
            for nbor, dist in zip(current.neighbors, current.nndist):
                if nbor.visited == False:    # only consider unvisited nodes
                    newd = current.dist2src + dist
                    if newd < nbor.dist2src: # best path to nbor goes through current
                        nbor.dist2src = newd
                        nbor.parent  = current
                    next_nodes.inject(nbor)  # add to list of unvisited nodes
            current.visited = True
            if current is target: # object comparison
                break             # even if next_nodes not exhausted
            istep += 1
            #print(istep/len(self))
            
        # --- wrap up ---
        if target.dist2src == sys.maxsize:
            return None # implies source/target unconnected
        else:
            return target.dist2src 

    #  -----------------------------------------------------    
    #  Recreate 2D path from source to target via lattice
    #  include source and target points
    #  if source and target unconnected, return empty path
    #  -----------------------------------------------------
    def backtrack_path(self):
        path = []
        if self.target.dist2src < sys.maxsize:
            this = self.target
            while this != None: # 
                path.append(this.pos)
                this = this.parent # this == None if this above was source
        path.reverse() # report part oriented from source to target
        return path

    # -----------------------------------------------------
    # Prepare node list for new Djikstra run
    # detach target + source from node list (= self) and
    # remove target + source reference
    # clear search history on self and target + source
    #  -----------------------------------------------------
    def reset(self):
        if self.source is not None:
            for member in self.source.neighbors[:]:   # must use slicing, because list changes during loop
                member.detach(self.source) # mutual detachment
            self.source.reset()    
            self.source = None
        #
        if self.target is not None:
            for member in self.target.neighbors[:]:   # must use slicing, because list changes during loop
                member.detach(self.target) # mutual detachment
            self.target.reset()
            self.target = None
        #
        for node in self:
            node.reset() # clear search history
        #

    # ----------------------------------------------------------------------------------    
    # Connect external node (with defined position) to lattice
    # extnode must be interior on the lattice (i.e. be located in a valid lattice cell
    # but not a lattice node). A copy (same position) of a lattice node is accepted.
    # This method do not store a reference to the node
    # ----------------------------------------------------------------------------------  
    def connect_new_node(self, extnode):
        assert extnode not in self
        # --- locate hosting cell on lattice ---
        ix  = self.xgrid.cell_index(extnode.pos[0])
        iy  = self.ygrid.cell_index(extnode.pos[1])
        ino = self.validmap[ix,iy]
        assert ino >= 0  # must be in valid cell
        host = self[ino]
        # --- inherit neighbors from host, and connect to them
        for nbor in host.neighbors[:]:
            nbor.connect(extnode)
        host.connect(extnode) # connect to host itself AFTER neighbor loop

    # ----------------------------------------------------------------------------------     
    # Plot raster solution of last Djikstra run, using coloring invalid cells of validmap 
    # and (xgrid,ygrid) as axes
    # ---------------------------------------------------------------------------------- 
    def plot_path(self, show=True):    
        x       = self.xgrid.boundary_points() # pcolormesh applies quadrilateral corners
        y       = self.ygrid.boundary_points() # pcolormesh applies quadrilateral corners
        X, Y    = np.meshgrid(x, y)
        Z       = np.transpose(np.where(self.validmap<0, 1, 0))
        fighan  = plt.figure()
        plothan = plt.pcolormesh(X,Y,Z)
        path    = np.array(self.backtrack_path())
        plt.plot(path[:,0], path[:,1])
        if show:
            plt.show()
    def savefig_path(self,fname):
        self.plot_path(show=False)
        plt.savefig(fname, dpi=300)

# ================================================================================================
#    Manage list of nodes to visit next
#
#    list is kept sorted ascendingly according to dist2src
#    Rely on super class initialization
# ================================================================================================     
class PriorityList(list):
    # isolate proper position by bisection
    def inject(self, anode):
        # --- empty list special case ---
        if len(self) == 0:
            self.append(anode)
            return
        # --- first remove anode, if present ---
        if anode in self:
            del self[self.index(anode)]
        n = len(self)
        if len(self)==0 or anode.dist2src < self[0].dist2src: # evaluated left to right 
            self.insert(0, anode)
            return
        elif anode.dist2src > self[-1].dist2src:
            self.insert(n, anode)
            return
        # --- anode should be intermediate, locate by bisection ---
        low = 0
        up  = n-1
        while up-low > 1:
            mid = round(0.5*(up+low))
            if self[mid].dist2src > anode.dist2src:
                up = mid
            else:
                low = mid
        self.insert(low, anode)
        
        

###################################################################################
if __name__ == "__main__":
    #
    # ------------ example 1 : path around an obstracle ----------------
    #
    # nx = 12
    # ny = 15
    # validmap = -np.ones((nx,ny), np.int)
    # i = 0
    # # --- testcase: box with an island in the middle
    # for ix in range(nx):
    #     for iy in range(ny):
    #         if 3 < ix < 8 and 4 < iy < 11: continue   
    #         validmap[ix,iy] = i
    #         i += 1
    # #print(np.flip(np.transpose(validmap), axis=0))  
    # xgrid  = Regular1DGrid(3  + 0.17*np.arange(nx))
    # ygrid  = Regular1DGrid(55 + 0.1*np.arange(ny))
    # medium = LatticeNodes(validmap, xgrid, ygrid)
    # #
    # src = Node((3.45, 55.49))
    # tgt = Node((4.4,  55.91))
    # plen = medium.run_Djikstra(src,tgt)
    # #print(plen)
    # #for xy in medium.backtrack_path():
    # #   print(*xy)
    # medium.plot_path()
    #
    #
    # ------------ example 2 : path in subarea of GEBCO map (gridone.nc) ----------------
    #                          GEBCO: depths negative, heights positive, all grid points defined
    import netCDF4 as NetCDF
    # ----- load full data set ----
    ncfile    = NetCDF.Dataset('gridone.nc', 'r')
    xa0,xa1   = ncfile.variables['x_range'][:]    # full grid dim
    ya0,ya1   = ncfile.variables['y_range'][:]    # full grid dim
    nxa,nya   = ncfile.variables['dimension'][:]  # full grid dim
    data      = np.transpose(np.reshape(ncfile.variables['z'][:] , (nya,nxa))) # shape (nxa,nya)
    data      = np.flip(data, axis=1)
    xfg       = Regular1DGrid(np.linspace(xa0,xa1,nxa)) # full xgrid
    yfg       = Regular1DGrid(np.linspace(ya0,ya1,nya)) # full ygrid
    # ----- scoop out sub map ----
    W,S,E,N   = 8.1, 56, 13.6, 58 # analysis window
    ix0 = xfg.cell_index(W)
    ix1 = xfg.cell_index(E)+1  # include E cells
    iy0 = yfg.cell_index(S)
    iy1 = yfg.cell_index(N)+1  # include N cells
    xgrid    = Regular1DGrid(xfg[ix0:ix1])
    ygrid    = Regular1DGrid(yfg[iy0:iy1])
    validmap = data[ix0:ix1, iy0:iy1]
    # ----- preprocess validmap
    ipt = 0
    for ix in range(len(xgrid)):
        for iy in range(len(ygrid)):
            if validmap[ix,iy] < 0:
                validmap[ix,iy] = ipt  # include all wet, label sequentially
                ipt += 1
            else:
                #print(xgrid[ix], ygrid[iy])
                validmap[ix,iy] = -1   # mark land points as invalid
    validmap = validmap.astype(np.int) # recast to integer
    #
    medium   = LatticeNodes(validmap, xgrid, ygrid)
    src      = Node((8.2,    57.1)) # point in North Sea 
    tgt      = Node((10.48,  56.9)) # point in Kattegat, must swim around Skagen
    plen = medium.run_Djikstra(src,tgt)
    #print(plen)
    medium.plot_path()
    # #
