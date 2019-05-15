import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


delta = 0.999
ddelta = 0.05

# 3D dist
def dist(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(sum((pt2 - pt1)**2))

def unit_vector(v):
    norm_v = np.linalg.norm(v)
    if norm_v < 0.00000000001:
        norm_v = 0.000000000001
    return v / norm_v

def angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class RRT:
    __slots__ = ['boundary', 'obstacles', 'start', 'goal', 'G', 'path']

    def __init__(self, boundary, obstacles):
        self.boundary = boundary[0]
        self.obstacles = obstacles
        self.G = {
                    'nodes' : [],
                    'costs' : [],
                    'parents' : dict()
                }   
        self.path = []




    # TRUE means there was a collision
    def check_collisions(self, pt):
        for (x1,y1,z1,x2,y2,z2,r,g,b) in self.obstacles:
            if( pt[0] > x1 and pt[0] < x2 and\
                pt[1] > y1 and pt[1] < y2 and\
                pt[2] > z1 and pt[2] < z2 ):
                    return True
        return False



    def check_collisions_between(self, pt1, pt2, delta=delta, ddelta=ddelta):
        # Check for collisions along the way
        direction = unit_vector(np.array(pt2) - np.array(pt1))
        waypoint = copy.copy(pt1)
        collision = False
        dd=ddelta
        while (dd < delta) and not collision:
            waypoint += ddelta*direction
            dd += ddelta
            collision = self.check_collisions(waypoint)
        return collision

    def rand_free_pt(self, goalsample=False, goal=None, randfunc=np.random.uniform):
        # Randomly sample a location with a goalsample probability of sampling the goal point
        if goalsample and (goal is not None):
            if(np.random.uniform(0,1) < goalsample):
                return goal

        rand_pt = [ randfunc(self.boundary[0], self.boundary[3]), 
                    randfunc(self.boundary[1], self.boundary[4]),
                    randfunc(self.boundary[2], self.boundary[5]) ]

        # Resample if it's in an obstacle
        while( self.check_collisions(rand_pt) ):
            rand_pt = [ randfunc(self.boundary[0], self.boundary[3]), 
                        randfunc(self.boundary[1], self.boundary[4]),
                        randfunc(self.boundary[2], self.boundary[5]) ]
        return np.array(rand_pt)


    def append_node(self,node,parent,cost):
        ind = len(self.G['nodes'])
        self.G['nodes'].append(np.array(node))
        self.G['parents'].update({ind: parent})
        self.G['costs'].append(cost)

    def find_nearest_node_ind(self,pt):
        # Find nearest node
        min_dist = float("inf")
        for i, node in enumerate(self.G['nodes']): 
            d = dist(pt,node)
            if d < min_dist:
                min_dist = d
                nearest_node_ind = i
                nearest_node = node 

        return nearest_node_ind

    def find_near_node_ind(self, pt, gamma=500, r=None):
        n_nodes = len(self.G['nodes'])
        if not r:
            r = max(1.0, gamma * math.pow((math.log(n_nodes) / n_nodes), 1.0/len(pt)))
        distances = [dist(pt,node) for node in self.G['nodes']]
        near_node_indices = [i for i,val in enumerate(distances) if val < r]
        return near_node_indices

    

    def steer(self, pt, delta=1.0):
        # Get nearest node
        nnode_ind = self.find_nearest_node_ind(pt)
        nnode = self.G['nodes'][nnode_ind]
        # If pt is more than a delta away from the nearest node, make proxy point 
        # delta away in direction of pt.
        min_dist = dist(pt,nnode)
        if (min_dist >= delta):
            direction = unit_vector(pt-nnode)
            pt = nnode + delta*direction

        return np.array(pt)
        

    def choose_parent(self, pt, near_node_indices=None, r=None, step_cost_max=0.999):
        if not near_node_indices:
            near_node_indices = self.find_near_node_ind(pt,r=0.999)

        # Search through near nodes to find the closest parent that can 
        # be connected without collision.
        min_cost = float("inf")
        for i in near_node_indices:
            travel_cost = dist(self.G['nodes'][i],pt)
            if travel_cost < step_cost_max:
                potential_cost = self.G['costs'][i] + travel_cost
                if potential_cost < min_cost:
                    # We have a candidate for a closer parent!
                    # But let's make sure they can be connected without collision.
                    if not self.check_collisions_between(self.G['nodes'][i], pt):
                        min_cost = potential_cost
                        min_ind = i 

        if min_cost == float("inf"):
            return False

        # Return index of parent and value of cost
        return min_ind, min_cost


    def rewire(self, pt, pt_ind=-1, near_node_indices=None):
        # Assumes the point you are inputting is the most recently added point,
        # unless you use the pt_ind argument.
        # Note: G is a dictionary, a mutable datatype. Manipulations to G in the
        # function persist out of the scope of the function.
        if not near_node_indices:
            near_node_indices = self.find_near_node_ind(pt,r=0.999)

        for i in near_node_indices:
            nnode = self.G['nodes'][i]
            nnode_cost = self.G['costs'][i]

            direction = unit_vector(np.array(pt)-np.array(nnode))
            distance = dist(pt,nnode)
            
            pt_cost = self.G['costs'][pt_ind]
            rewire_cost = pt_cost + distance 

            if nnode_cost > rewire_cost:
                if not self.check_collisions_between(nnode, pt):
                    # Rewire such that pt is the parent of nnode
                    # and update cost
                    num_nodes = len(self.G['nodes'])
                    self.G['parents'][i] = num_nodes-1
                    self.G['costs'][i] = rewire_cost




    


    

    def run_rrt_star(self, start, goal, delta=1, ddelta=0.1):
        
        self.append_node(start,None,0)

        MAX_ITERS = 1000000
        at_goal = False
        iters = 0
        while not at_goal and iters < MAX_ITERS:
            # Sample a random point (with 5% chance of sampling goal point)
            rand_pt = self.rand_free_pt(goalsample=0.05) 

            # Make proxy point if rand_pt is more than delta away from nearest point
            new_node = self.steer(rand_pt, delta=delta)

            # Check that (possibly updated) rand_pt is collision free.
            # If there is a collision, point is not added to graph, time to re-sample.
            if not self.check_collisions(new_node):
                near_nodes = self.find_near_node_ind(new_node, r=.999)
                success = self.choose_parent(new_node,near_node_indices=near_nodes)  # True means the point was added to the graph
                if success:
                    self.append_node(new_node, success[0], success[1])
                    self.rewire(new_node, near_node_indices=near_nodes)

                # End condition
                goal_unc = 0.01 
                d = dist(new_node,goal)
                if ( d < goal_unc ):
                    at_goal = True

            iters += 1

        # Was the goal reached?
        if at_goal:
            print('RRT*: Goal reached')
        else:
            print('RRT*: No luck')

        # Backtrack to find correct path
        # Find parent of this node
        path_inds = [ len(self.G['nodes'])-1 ] # Start with last point
        while( np.all(self.G['nodes'][path_inds[-1]]!=start) ):
            child = path_inds[-1]
            parent = self.G['parents'][child]
            path_inds.append(parent)

        # Now we have the indices of the path, backwards
        path_inds.reverse() # now goes from start_pt to (near) goal_pt
        self.path = []
        for ind in path_inds:
            self.path.append(self.G['nodes'][ind])
        
        return self.path

