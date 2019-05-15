
import numpy as np
import time
import matplotlib;matplotlib.use("TkAgg")
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import RobotPlanner


DELAY = 0.5 #sec

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  

def load_map(fname):
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = np.array(mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].tolist())
  blocks = np.array(mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].tolist())
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])  
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def runtest(mapfile, start, goal, verbose = True, delay=False):
  # Instantiate a robot planner
  boundary, blocks = load_map(mapfile)
  RP = RobotPlanner.RobotPlanner(boundary, blocks)
  # Display the environment
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)  
  
  # Main loop
  robotpos = np.copy(start)
  numofmoves = 0
  success = True
  while True:
    print()
  
    # Call the robot planner
    t0 = tic()
    newrobotpos = RP.planRRTstar(robotpos, goal)
    t1 = (tic()-t0)
    print('computation time: {}'.format(t1))
    movetime = max(1, np.ceil(t1/2.0))
    print('move time: %d' % movetime)

    # See if the planner was done on time
    # if movetime > 1:
    #   newrobotpos = robotpos-0.5 + np.random.rand(3)
    print('newrobotpos: {}'.format(newrobotpos))

    # Check if the commanded position is valid
    delta_pos = np.sqrt(sum((newrobotpos - robotpos)**2))
    if delta_pos > 1:
      print('ERROR: the robot cannot move so fast: {}\n'.format(delta_pos))
      success = False
    if( newrobotpos[0] < boundary[0,0] or newrobotpos[0] > boundary[0,3] or \
        newrobotpos[1] < boundary[0,1] or newrobotpos[1] > boundary[0,4] or \
        newrobotpos[2] < boundary[0,2] or newrobotpos[2] > boundary[0,5] ):
      print('ERROR: out-of-map robot position commanded\n')
      success = False
    for k in range(blocks.shape[0]):
      if( newrobotpos[0] > blocks[k,0] and newrobotpos[0] < blocks[k,3] and\
          newrobotpos[1] > blocks[k,1] and newrobotpos[1] < blocks[k,4] and\
          newrobotpos[2] > blocks[k,2] and newrobotpos[2] < blocks[k,5] ):
        print('ERROR: collision... BOOM, BAAM, BLAAM!!!\n')
        success = False
        time.sleep(5)
        break
    if( success is False ):
      break
    
    # Make the move
    robotpos = newrobotpos
    numofmoves += 1
    
    # Update plot
    if verbose:
      hs[0].set_xdata(robotpos[0])
      hs[0].set_ydata(robotpos[1])
      hs[0].set_3d_properties(robotpos[2])
      fig.canvas.flush_events()
      plt.show()
      
    # Check if the goal is reached
    if sum((robotpos-goal)**2) <= 0.1:
      break

    if delay:
      time.sleep(delay)

  return success, numofmoves



def test_single_cube():    
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 6.0])
  success, numofmoves = runtest('./maps/single_cube.txt', start, goal, True, DELAY)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  
  
def test_maze():
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  success, numofmoves = runtest('./maps/maze.txt', start, goal, True, DELAY)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  
def test_window():
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([6.0, 18.0, 3.0])
  success, numofmoves = runtest('./maps/window.txt', start, goal, True, DELAY)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)

def test_tower():
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  success, numofmoves = runtest('./maps/tower.txt', start, goal, True, DELAY)
 
def test_flappy_bird():
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  success, numofmoves = runtest('./maps/flappy_bird.txt', start, goal, True, DELAY)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves) 

def test_room():
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  success, numofmoves = runtest('./maps/room.txt', start, goal, True, DELAY)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)

def test_monza():
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  success, numofmoves = runtest('./maps/monza.txt', start, goal, True, DELAY)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)


if __name__=="__main__":
  test_single_cube()
  # test_maze()
  # test_flappy_bird()
  # test_monza()
  # test_window()
  # test_tower()
  # test_room()
  








