# record the map of a maze

import numpy as np
from skimage import transform

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ANGLE_STEP = 0.1514 # 80/3320 * 2pi
FORWARD_STEP = 1

class MazeMap(object):
  def __init__(self, height=84, width=84):
    self.height = height
    self.width = width
    self.maze = np.zeros((self.height, self.width))
    self.current_h = self.height / 2.0 # current height position
    self.current_w = self.width / 2.0 # current width position
    self.current_a = 0.0 # current angle

  def update(self, action, obstacle=False):
    if action == ACT_LEFT:
      self.current_a -= ANGLE_STEP
      self.current_a = self.current_a % (2*np.pi)
    if action == ACT_RIGHT:
      self.current_a += ANGLE_STEP
      self.current_a = self.current_a % (2*np.pi)
    if action == ACT_FORWARD:
      self.current_h += FORWARD_STEP * np.sin(self.current_a)
      self.current_w += FORWARD_STEP * np.cos(self.current_a)
      self.maze[int(self.current_h),int(self.current_w)] += 0.1
      # update the maze map if neccessary
      if (self.current_h > self.height - FORWARD_STEP*3):
        self.height += 20
        new_maze = np.zeros((self.height, self.width))
        new_maze[:-20,:] = self.maze
        self.maze = new_maze
      if (self.current_h < FORWARD_STEP*3):
        self.height += 20
        new_maze = np.zeros((self.height, self.width))
        new_maze[20:,:] = self.maze
        self.maze = new_maze
      if (self.current_w >self.width - FORWARD_STEP*3):
        self.width += 20
        new_maze = np.zeros((self.height, self.width))
        new_maze[:,:-20] = self.maze
        self.maze = new_maze
      if (self.current_w < FORWARD_STEP*3):
        self.width += 20
        new_maze = np.zeros((self.height, self.width))
        new_maze[:,20:] = self.maze
        self.maze = new_maze

    if obstacle:
      h = self.current_h + np.ceil(2*np.sin(self.current_a))
      w = self.current_w + np.ceil(2*np.cos(self.current_a))
      self.maze[int(h),int(w)] = -1

  def get_map(self, height, width):
    return transform.resize(self.maze, (height, width))
    

