import numpy as np

class RobotPlanner:
  __slots__ = ['boundary', 'blocks']

  def __init__(self, boundary, blocks):
    self.boundary = boundary
    self.blocks = blocks
  
  def plan(self,start,goal):
    # for now greedily move towards the goal
    newrobotpos = np.copy(start)
    
    numofdirs = 26
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
    dR = dR / np.sqrt(np.sum(dR**2,axis=0)) / 2.0
    
    mindisttogoal = 1000000
    for k in range(numofdirs):
      newrp = start + dR[:,k]
      
      # Check if this direction is valid
      if( newrp[0] < self.boundary[0,0] or newrp[0] > self.boundary[0,3] or \
          newrp[1] < self.boundary[0,1] or newrp[1] > self.boundary[0,4] or \
          newrp[2] < self.boundary[0,2] or newrp[2] > self.boundary[0,5] ):
        continue
      
      valid = True
      for k in range(self.blocks.shape[0]):
        if( newrp[0] > self.blocks[k,0] and newrp[0] < self.blocks[k,3] and\
            newrp[1] > self.blocks[k,1] and newrp[1] < self.blocks[k,4] and\
            newrp[2] > self.blocks[k,2] and newrp[2] < self.blocks[k,5] ):
          valid = False
          break
      if not valid:
        break
      
      # Update newrobotpos
      disttogoal = sum((newrp - goal)**2)
      if( disttogoal < mindisttogoal):
        mindisttogoal = disttogoal
        newrobotpos = newrp
    
    return newrobotpos

