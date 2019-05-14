import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


delta = 2
ddelta = 0.05

def print_tree(G):
    for (e1,e2) in G['edges']:
        x = [G['nodes'][e1][0], G['nodes'][e2][0]]
        y = [G['nodes'][e1][1], G['nodes'][e2][1]]
        plt.plot(x,y,'g')
        
# def plot_env(pt=None, ptStyle='og', G=None, obstacle_list=obstacle_list, start_pt=start_pt, goal_pt=goal_pt, bounds=bounds):
#     plt.clf()
#     for (obx, oby, obr) in obstacle_list:
#         plt.plot(obx, oby, "ok", ms=30 * obr)
#     plt.plot(start_pt[0], start_pt[1], "xb")
#     plt.plot(goal_pt[0], goal_pt[1], "xr")
#     if pt:
#         plt.plot(pt[0], pt[1], ptStyle)
#     if G:
#         print_tree(G)
    
#     plt.axis([bounds['x'][0], bounds['x'][1], bounds['y'][0], bounds['y'][1]])
    
# 3D dist
def dist(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(sum((pt2 - pt1)**2))

def unit_vector(v):
    return v / np.linalg.norm(v)

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
    theta = angle(pt1,pt2)
    waypoint = copy.copy(pt1)
    collision = False
    dd=ddelta
    while (dd < delta) and not collision:
        waypoint[0] += ddelta*math.cos(theta)
        waypoint[1] += ddelta*math.sin(theta)
        dd += ddelta
        collision = check_collisions(waypoint, obstacles)
    return collision

def rand_free_pt(boundary, obstacles, randfunc=np.random.uniform):
    # Randomly sample a location
    rand_pt = [ randfunc(boundary[0], boundary[3]), 
                randfunc(boundary[1], boundary[4]),
                randfunc(boundary[2], boundary[5]) ]

    # Resample if it's in an obstacle
    while( check_collisions(rand_pt, obstacles) ):
        rand_pt = [ randfunc(boundary[0], boundary[3]), 
                    randfunc(boundary[1], boundary[4]),
                    randfunc(boundary[2], boundary[5]) ]
    return rand_pt


def run_rrt(start_pt, goal_pt, boundary, obstacles, delta=1, ddelta=0.1):
    boundary = boundary[0]

    # Graph
    G = {
            "nodes" : [start_pt],
            "edges" : dict()
        }        

    ## RRT!
    MAX_ITERS = 1000000
    goal = False
    iters = 0
    while not goal and iters < MAX_ITERS:
        
        # Sample a random point
        rand_pt = rand_free_pt(boundary, obstacles)
        
        # Find nearest node
        min_dist = dist(boundary[:3], boundary[3:6])
        for i,node in enumerate(G['nodes']):
            d = dist(rand_pt,node)
            if d < min_dist:
                min_dist = d
                nearest_node_index = i
                nearest_node = node 

        # If rand_pt is more than a delta away from the nearest node, make proxy point 
        # delta away in direction of rand_pt.
        if (min_dist >= delta):
            diff_vector = np.array(nearest_node) - np.array(rand_pt)
            rand_pt = nearest_node + delta*unit_vector(diff_vector)

        print('Dist to nearest node: {}'.format(dist(rand_pt,nearest_node)))


        # Check if there are any collisions between nearest node and (possibly updated) rand_pt.
        # If there is a collision, point is not added to graph, time to re-sample.
        collision = check_collisions_between(nearest_node,rand_pt,obstacles)
        if not collision:
            G['nodes'].append(rand_pt)
            G['edges'].update({(len(G['nodes'])-1) : int(nearest_node_index)}) # key=child, value=parent -- makes it easy
                                                                            # to search for parent of child    
            # End condition
            goal_unc = 50
            if ( dist(rand_pt,goal_pt)<goal_unc ):
                goal = True
        
        iters += 1
        # print(len(G['nodes']))



    # Was the goal reached?
    if goal:
        print('RRT: Goal reached')
    else:
        print('RRT: No luck')

    # Backtrack to find correct path
    # Find parent of this node
    path = [ len(G['nodes'])-1 ]
    while( np.all(G['nodes'][path[-1]]!=start_pt) ):
        child = path[-1]
        parent = G['edges'][child]
        path.append(parent)

    print('Path\n{}'.format(path))
    new_pos = G['nodes'][path[-2]]
    print(new_pos)
    print('Dist to nearest node: {}'.format(dist(new_pos, G['nodes'][path[-1]])))
    return new_pos
        

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

    