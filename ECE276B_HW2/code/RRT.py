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


# TRUE means there was a collision
def check_collisions(pt, obstacles):
    for (x1,y1,z1,x2,y2,z2,r,g,b) in obstacles:
        if( pt[0] > x1 and pt[0] < x2 and\
            pt[1] > y1 and pt[1] < y2 and\
            pt[2] > z1 and pt[2] < z2 ):
                return True
    return False



def check_collisions_between(pt1, pt2, obstacles, delta=delta, ddelta=ddelta):
    # Check for collisions along the way
    direction = unit_vector(np.array(pt2) - np.array(pt1))
    waypoint = copy.copy(pt1)
    collision = False
    dd=ddelta
    while (dd < delta) and not collision:
        waypoint += ddelta*direction
        dd += ddelta
        collision = check_collisions(waypoint, obstacles)
    return collision

def rand_free_pt(boundary, obstacles, goalsample=False, goal=False, randfunc=np.random.uniform):
    # Randomly sample a location with a goalsample probability of sampling the goal point
    if goalsample:
        if(np.random.uniform(0,1) < goalsample):
            return goal

    rand_pt = [ randfunc(boundary[0], boundary[3]), 
                randfunc(boundary[1], boundary[4]),
                randfunc(boundary[2], boundary[5]) ]

    # Resample if it's in an obstacle
    while( check_collisions(rand_pt, obstacles) ):
        rand_pt = [ randfunc(boundary[0], boundary[3]), 
                    randfunc(boundary[1], boundary[4]),
                    randfunc(boundary[2], boundary[5]) ]
    return np.array(rand_pt)


def append_node(G,node,parent,cost):
    ind = len(G['nodes'])
    G['nodes'].append(np.array(node))
    G['parents'].update({ind: parent})
    G['costs'].append(cost)

    return G

def find_near_node_ind(pt, G, gamma=500, r=None):
    n_nodes = len(G['nodes'])
    if not r:
        r = max(1.0, gamma * math.pow((math.log(n_nodes) / n_nodes), 1.0/len(pt)))
    # # print('r: {}'.format(r))
    distances = [dist(pt,node) for node in G['nodes']]
    near_node_indices = [i for i,val in enumerate(distances) if val < r]
    return near_node_indices

def choose_parent(pt, G, obstacles, near_node_indices=None, r=None, step_cost_max=0.999):
    if not near_node_indices:
        near_node_indices = find_near_node_ind(pt,G,r=0.999)
    # print('Near node indices: {}'.format(near_node_indices))

    # Search through near nodes to find the closest parent that can 
    # be connected without collision.
    min_cost = float("inf")
    for i in near_node_indices:
        travel_cost = dist(G['nodes'][i],pt)
        if travel_cost < step_cost_max:
            # print('node: {}, travel_cost: {}'.format(i,travel_cost))
            potential_cost = G['costs'][i] + travel_cost
            if potential_cost < min_cost:
                # We have a candidate for a closer parent!
                # But let's make sure they can be connected without collision.
                if not check_collisions_between(G['nodes'][i], pt, obstacles):
                    min_cost = potential_cost
                    min_ind = i 

    if min_cost == float("inf"):
        return False

    # If you got this far, the new pt has a valid parent, and can be added to the tree
    return min_ind, min_cost





def rewire(pt, G, obstacles, pt_ind=-1, near_node_indices=None):
    # Assumes the point you are inputting is the most recently added point,
    # unless you use the pt_ind argument.
    # Note: G is a dictionary, a mutable datatype. Manipulations to G in the
    # function persist out of the scope of the function.
    if not near_node_indices:
        near_node_indices = find_near_node_ind(pt,G,r=1.0)

    for i in near_node_indices:
        nnode = G['nodes'][i]
        nnode_cost = G['costs'][i]

        direction = unit_vector(np.array(pt)-np.array(nnode))
        distance = dist(pt,nnode)
        
        pt_cost = G['costs'][pt_ind]
        rewire_cost = pt_cost + distance 

        if nnode_cost > rewire_cost:
            if not check_collisions_between(nnode, pt, obstacles):
                # Rewire such that pt is the parent of nnode
                # and update cost
                num_nodes = len(G['nodes'])
                G['parents'][i] = num_nodes-1
                G['costs'][i] = rewire_cost

    return G



def find_nearest_node_ind(pt, G):
    #near_node_indices = find_near_node_ind(pt,G)
    # Find nearest node
    min_dist = float("inf")
    for i, node in enumerate(G['nodes']): #near_node_indices:
        #node = G['nodes'][i]
        d = dist(pt,node)
        if d < min_dist:
            min_dist = d
            nearest_node_ind = i
            nearest_node = node 

    return nearest_node_ind


def steer(pt, G, delta=1.0):
    # Get nearest node
    nnode_ind = find_nearest_node_ind(pt, G)
    nnode = G['nodes'][nnode_ind]
    # If pt is more than a delta away from the nearest node, make proxy point 
    # delta away in direction of pt.
    min_dist = dist(pt,nnode)
    if (min_dist >= delta):
        direction = unit_vector(pt-nnode)
        pt = nnode + delta*direction

    return np.array(pt)

def run_rrt_star(start_pt, goal_pt, boundary, obstacles, delta=1, ddelta=0.1):
    boundary = boundary[0]

    # Graph
    G = {
            'nodes' : [start_pt],
            'costs' : [0],
            'parents' : dict()
        }        

    ## RRT!
    MAX_ITERS = 1000000
    goal = False
    iters = 0
    while not goal and iters < MAX_ITERS:
        
        # Sample a random point (with 5% chance of sampling goal point)
        rand_pt = rand_free_pt(boundary, obstacles, goalsample=0.05, goal=goal_pt) 
        # print('\nRP: {}'.format(rand_pt))
        # nearest = find_nearest_node_ind(rand_pt,G)
        # print('NearestNode: {}'.format(G['nodes'][nearest]))
        # Make proxy point if rand_pt is more than delta away from nearest point
        new_node = steer(rand_pt, G, delta=delta)
        # print('NewNode: {}'.format(new_node))
        # print('Dist to nearest node: {}'.format(dist(new_node, G['nodes'][nearest])))

        # Check that (possibly updated) rand_pt is collision free.
        # If there is a collision, point is not added to graph, time to re-sample.
        if not check_collisions(new_node, obstacles):
            # print('No collision.')
            near_nodes = find_near_node_ind(new_node, G, r=.999)
            success = choose_parent(new_node,G,obstacles,near_node_indices=near_nodes)  # True means the point was added to the graph
            if success:
                # print('success: {}'.format(success))
                G = append_node(G, new_node, success[0], success[1])
                # ind = len(G['nodes']) - 1
                # parent_ind = G['parents'][ind]
                # parent = G['nodes'][parent_ind]
                # print('Parent: {}, index {}'.format(parent, parent_ind))
                # print('Dist to parent: {}'.format(dist(new_node, parent)))
                # only need to rewire if new node was added
                G = rewire(new_node,G,obstacles,near_node_indices=near_nodes)

            # End condition
            goal_unc = 0.01 
            d = dist(new_node,goal_pt)
            if ( d < goal_unc ):
                goal = True


        iters += 1
        # # print(len(G['nodes']))



    # Was the goal reached?
    if goal:
        print('RRT*: Goal reached')
    else:
        print('RRT*: No luck')

    # Backtrack to find correct path
    # Find parent of this node
    path_inds = [ len(G['nodes'])-1 ] # Start with last point
    while( np.all(G['nodes'][path_inds[-1]]!=start_pt) ):
        child = path_inds[-1]
        parent = G['parents'][child]
        path_inds.append(parent)

    # Now we have the indices of the path, backwards
    path_inds.reverse() # now goes from start_pt to (near) goal_pt
    path = []
    for ind in path_inds:
        path.append(G['nodes'][ind])


    # print('path_inds\n{}'.format(path_inds))
    # print('Path\n{}'.format(path))
    # new_pos = path[1]
    # print(new_pos)
    # print('Dist to nearest node: {}'.format(dist(new_pos, start_pt)))
    return path

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


# def run_rrt(start_pt, goal_pt, boundary, obstacles, delta=1, ddelta=0.1):
#     boundary = boundary[0]

#     # Graph
#     G = {
#             'nodes' : [start_pt],
#             'costs' : [0],
#             'parents' : dict()
#         }        

#     ## RRT!
#     MAX_ITERS = 1000000
#     goal = False
#     iters = 0
#     while not goal and iters < MAX_ITERS:
        
#         # Sample a random point
#         rand_pt = rand_free_pt(boundary, obstacles)
        
#         # Find nearest node
#         min_dist = dist(boundary[:3], boundary[3:6])
#         for i,node in enumerate(G['nodes']):
#             d = dist(rand_pt,node)
#             if d < min_dist:
#                 min_dist = d
#                 nearest_node_index = i
#                 nearest_node = node 

#         # If rand_pt is more than a delta away from the nearest node, make proxy point 
#         # delta away in direction of rand_pt.
#         if (min_dist >= delta):
#             direction = unit_vector(np.array(rand_pt) - np.array(nearest_node))
#             rand_pt = nearest_node + delta*direction

#         # # print('Dist to nearest node: {}'.format(dist(rand_pt,nearest_node)))


#         # Check if there are any collisions between nearest node and (possibly updated) rand_pt.
#         # If there is a collision, point is not added to graph, time to re-sample.
#         collision = check_collisions_between(nearest_node,rand_pt,obstacles)
#         if not collision:
#             G['nodes'].append(rand_pt)
#             G['parents'].update({(len(G['nodes'])-1) : int(nearest_node_index)}) # key=child, value=parent -- makes it easy
#                                                                             # to search for parent of child    
#             parent_cost = G['costs'][int(nearest_node_index)]
#             G['costs'].append(parent_cost + dist(nearest_node,rand_pt))

#             # End condition
#             goal_unc = 10.0 # TODO
#             if ( dist(rand_pt,goal_pt)<goal_unc ):
#                 goal = True
        
#         iters += 1
#         # # print(len(G['nodes']))



#     # Was the goal reached?
#     if goal:
#         # print('RRT: Goal reached')
#     else:
#         # print('RRT: No luck')

#     # Backtrack to find correct path
#     # Find parent of this node
#     path = [ len(G['nodes'])-1 ]
#     while( np.all(G['nodes'][path[-1]]!=start_pt) ):
#         child = path[-1]
#         parent = G['parents'][child]
#         path.append(parent)

#     # print('Path\n{}'.format(path))
#     new_pos = G['nodes'][path[-2]]
#     # print(new_pos)
#     # print('Dist to nearest node: {}'.format(dist(new_pos, G['nodes'][path[-1]])))
#     return new_pos

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

    #     # Backtrack to find correct path
    # # Find parent of this node
    # path_indices = [ len(G['nodes'])-1 ]
    # while( np.all(G['nodes'][path_indices[-1]]!=start_pt) ):
    #     child = path_indices[-1]
    #     parent = G['parents'][child]
    #     path_indices.append(parent)

    
    # path = []
    # path_indices.reverse()
    # for i in path_indices:
    #     pt = G['nodes'][i]
    #     path.append(pt)

    # # print('Path\n{}'.format(path))
    # new_pos = G['nodes'][path[-2]]
    # # print(new_pos)
    # # print('Dist to nearest node: {}'.format(dist(new_pos, G['nodes'][path[-1]])))
    # return path.reverse()
        

# if __name__=="__main__":
#     # ENVIRONMENT
#     bounds = { 'x': (0, 10),
#                'y': (0, 10)
#              }

#     start_pt = [1,1]
#     goal_pt = [5,6]


#     # All obstacles here are circles centered at ([0], [1]) with radius [2]
#     obstacle_list = [[5,5,1], [2,6,3], [4,4.5,2], [6,4,2]];

#     delta = 2
#     ddelta = 0.05

#     # Plot found path
#     plot_env()
#     last_pt = G['nodes'][path[0]]
#     for ind in path:
#         pt = G['nodes'][ind]
#         plt.plot([last_pt[0], pt[0]], [last_pt[1], pt[1]], 'g')
#         plt.plot(pt[0],pt[1],'*g')
#         last_pt = pt

    