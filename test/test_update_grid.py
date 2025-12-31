"""
Quick test for update_grid functionality
快速测试 update_grid 功能
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import world, cam

# Test update_grid
cam.update_grid(world, 0.01)
print('update_grid OK / update_grid 正常')
