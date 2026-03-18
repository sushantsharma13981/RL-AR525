"""
==========================================================================
                    MAIN.PY - UR5 GRID NAVIGATION
==========================================================================
Students implement DP algorithms in utils.py and run this to see results.

Dependencies:
    - pybullet
    - numpy
    - utils.py

Usage:
    python main.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import time
import os
import sys


from utils import (
    GridEnv,
    policy_iteration,
    value_iteration
)



def state_to_position(state, rows, cols, grid_size=0.10, 
                      table_center=[0, -0.3, 0.65], z_offset=0.10):

    row = state // cols
    col = state % cols
    
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    
    return [x, y, z]


def get_m_shape_cells():

    m_cells = set()

    # Left vertical leg (cols 0-1, all rows) - 2 cells wide
    for row in range(7):
        m_cells.add((row, 0))
        m_cells.add((row, 1))

    # Right vertical leg (cols 7-8, all rows) - 2 cells wide
    for row in range(7):
        m_cells.add((row, 7))
        m_cells.add((row, 8))

    # Left diagonal (from top-left going down to center)
    # Row 5: extend to col 2
    m_cells.add((5, 2))
    # Row 4: cols 2, 3
    m_cells.add((4, 2))
    m_cells.add((4, 3))
    # Row 3: cols 2, 3, 4 (connecting to middle)
    m_cells.add((3, 2))
    m_cells.add((3, 3))
    m_cells.add((3, 4))
    # Row 2: col 4 (bottom of V)
    m_cells.add((2, 4))

    # Right diagonal (from top-right going down to center)
    # Row 5: extend to col 6
    m_cells.add((5, 6))
    # Row 4: cols 5, 6
    m_cells.add((4, 5))
    m_cells.add((4, 6))
    # Row 3: cols 4, 5, 6 (connecting to middle - 4 already added)
    m_cells.add((3, 5))
    m_cells.add((3, 6))

    # Middle vertical extension (above the V bottom)
    m_cells.add((4, 4))  # extends middle upward
    m_cells.add((5, 4))  # continues up
    m_cells.add((6, 4))  # top of middle extension

    return m_cells


def cell_to_position(row, col, rows, cols, grid_size=0.10,
                     table_center=[0, -0.3, 0.65], z_offset=0.10):
    """Convert (row, col) to world position."""
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    return [x, y, z]


def draw_m_grid(grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Draw the M-shaped grid with individual blocks.
    Returns the valid cells, start cell, and end cell.
    """
    m_cells = get_m_shape_cells()
    rows, cols = 7, 9  # Grid dimensions for the M shape

    line_color = [0, 0, 0]  # Black borders
    line_width = 2
    z = table_center[2] + 0.001
    half = grid_size / 2

    # Draw each M-shaped cell as a block
    for (row, col) in m_cells:
        pos = cell_to_position(row, col, rows, cols, grid_size, table_center, z_offset=0.001)
        x, y = pos[0], pos[1]

        # Draw filled square (4 border lines)
        p.addUserDebugLine([x-half, y-half, z], [x+half, y-half, z], line_color, line_width)
        p.addUserDebugLine([x+half, y-half, z], [x+half, y+half, z], line_color, line_width)
        p.addUserDebugLine([x+half, y+half, z], [x-half, y+half, z], line_color, line_width)
        p.addUserDebugLine([x-half, y+half, z], [x-half, y-half, z], line_color, line_width)

    # Define start and end cells
    start_cell = (0, 0)   # Bottom-left of M
    end_cell = (0, 8)     # Bottom-right of M

    # Draw start marker (green)
    start_pos = cell_to_position(start_cell[0], start_cell[1], rows, cols, grid_size, table_center, z_offset=0.005)
    marker_half = half * 0.6
    green = [0, 1, 0]
    p.addUserDebugLine([start_pos[0]-marker_half, start_pos[1]-marker_half, start_pos[2]],
                       [start_pos[0]+marker_half, start_pos[1]-marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]+marker_half, start_pos[1]-marker_half, start_pos[2]],
                       [start_pos[0]+marker_half, start_pos[1]+marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]+marker_half, start_pos[1]+marker_half, start_pos[2]],
                       [start_pos[0]-marker_half, start_pos[1]+marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]-marker_half, start_pos[1]+marker_half, start_pos[2]],
                       [start_pos[0]-marker_half, start_pos[1]-marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugText("START", [start_pos[0], start_pos[1], start_pos[2] + 0.05], green, 1.0)

    # Draw end marker (red)
    end_pos = cell_to_position(end_cell[0], end_cell[1], rows, cols, grid_size, table_center, z_offset=0.005)
    red = [1, 0, 0]
    p.addUserDebugLine([end_pos[0]-marker_half, end_pos[1]-marker_half, end_pos[2]],
                       [end_pos[0]+marker_half, end_pos[1]-marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]+marker_half, end_pos[1]-marker_half, end_pos[2]],
                       [end_pos[0]+marker_half, end_pos[1]+marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]+marker_half, end_pos[1]+marker_half, end_pos[2]],
                       [end_pos[0]-marker_half, end_pos[1]+marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]-marker_half, end_pos[1]+marker_half, end_pos[2]],
                       [end_pos[0]-marker_half, end_pos[1]-marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugText("END", [end_pos[0], end_pos[1], end_pos[2] + 0.05], red, 1.0)



    return m_cells, start_cell, end_cell





if __name__ == "__main__":
    

    ROWS = 5
    COLS = 6
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    
    env = GridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL)

    

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    p.loadURDF("plane.urdf")
    
    table_path = os.path.join("assest", "table", "table.urdf")
    p.loadURDF(table_path, [0, -0.3, 0], globalScaling=2.0)
    
    stand_path = os.path.join("assest", "robot_stand.urdf")
    p.loadURDF(stand_path, [0, -0.8, 0], useFixedBase=True)
    
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_start_pos = [0, -0.8, 0.65]
    ur5_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    ur5_id = p.loadURDF(ur5_path, ur5_start_pos, ur5_start_orn, useFixedBase=True)
    
    sys.stderr = old_stderr

    # Draw M-shaped grid with start and end markers
    m_cells, start_cell, end_cell = draw_m_grid()

    # Add fixed obstacles on the M-shaped grid (path is guaranteed to exist)
    rows, cols = 7, 9

   
    OBSTACLE_CELLS = [
        (3, 3),  
        (5, 1),  
        (5, 6),  
        (2, 1),  
        (2, 3),
        (5, 8),
    ]

    obstacle_path = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")

    print(f"\nFixed obstacles placed at:")
    for cell in OBSTACLE_CELLS:
        if cell in m_cells and cell != start_cell and cell != end_cell:
            obs_pos = cell_to_position(cell[0], cell[1], rows, cols, z_offset=0.025)
            p.loadURDF(obstacle_path, obs_pos)
            print(f"  Cell {cell}")

    print(f"Total obstacles: {len(OBSTACLE_CELLS)}")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except:
        pass

    # ============================================================
    # TODO: Implement DP algorithms in utils.py, then add simulation code here
    # ============================================================
