import unittest
from task import *


class LoadGridTest(unittest.TestCase):
    def test(self):
        task = SpatialNavigationTask("latent-learning")
        grid = task.get_grid()
        marked_states = task.get_marked_states()
        width, height = task.get_grid_size()
        # Grid size
        self.assertEqual(width, 10)
        self.assertEqual(height, 10)
        # Some cells in the middle
        self.assertEqual((6,3) in grid, False)
        self.assertEqual((7,3) in grid, True)
        self.assertEqual((6,4) in grid, True)
        self.assertEqual(grid[(7,3)], 0)
        self.assertEqual(grid[(6,4)], 0)
        # Corners
        self.assertEqual((0,0) in grid, False)
        self.assertEqual((0,9) in grid, False)
        self.assertEqual((9,0) in grid, True)
        self.assertEqual((9,9) in grid, True)
        # Marked cells
        self.assertEqual((0,7) in grid, True)
        self.assertEqual((9,1) in grid, True)
        self.assertEqual(grid[0,7], 0)
        self.assertEqual(grid[9,1], 0)
        self.assertEqual(len(marked_states), 2)
        self.assertEqual(1 in marked_states, True)
        self.assertEqual(2 in marked_states, True)
        self.assertEqual(marked_states[1], (0,7))
        self.assertEqual(marked_states[2], (9,1))
        # Top boundary
        self.assertEqual((1,0) in grid, True)
        self.assertEqual((2,0) in grid, False)
        self.assertEqual((3,0) in grid, False)
        self.assertEqual((4,0) in grid, True)
        # Left boundary
        self.assertEqual((0,1) in grid, True)
        self.assertEqual((0,2) in grid, True)
        self.assertEqual((0,3) in grid, False)
        self.assertEqual((3,0) in grid, False)
        # Bottom boundary
        self.assertEqual((1,9) in grid, True)
        self.assertEqual((2,9) in grid, False)
        self.assertEqual((3,9) in grid, False)
        self.assertEqual((4,9) in grid, True)
        # Right boundary
        self.assertEqual((9,2) in grid, False)
        self.assertEqual((9,3) in grid, False)
        self.assertEqual((9,7) in grid, False)
        self.assertEqual((9,8) in grid, True)
        # Out of bounds
        self.assertEqual((10,2) in grid, False)
        self.assertEqual((0,10) in grid, False)
        self.assertEqual((11,11) in grid, False)
        self.assertEqual((-1,3) in grid, False)
        self.assertEqual((-12,-6) in grid, False)


class LoadInstructionsTest(unittest.TestCase):
    def test(self):
        instr = SpatialNavigationTask("policy-revaluation")._instructions
        self.assertEqual(len(instr), 12)
        # Setting a single starting point
        self.assertEqual(instr[0]["type"], "set-start")
        self.assertEqual(len(instr[0]["states"]), 1)
        self.assertEqual(instr[0]["states"][0], (0,7))
        # Setting multiple starting points
        self.assertEqual(instr[7]["type"], "set-start")
        self.assertEqual(len(instr[7]["states"]), 2)
        self.assertEqual(instr[7]["states"][0], (0,7))
        self.assertEqual(instr[7]["states"][1], (8,8))
        # Placing rewards
        self.assertEqual(instr[2]["type"], "place-reward")
        self.assertEqual(instr[2]["state"], (9,1))
        self.assertEqual(instr[2]["value"], 10)
        self.assertEqual(instr[9]["type"], "place-reward")
        self.assertEqual(instr[9]["state"], (9,8))
        self.assertEqual(instr[9]["value"], 20)
        # Simulating one trial with limited number of steps
        self.assertEqual(instr[1]["type"], "simulate")
        self.assertEqual(instr[1]["num-trials"], 1)
        self.assertEqual(instr[1]["max-num-steps"], 25000)
        # Simulating one trial with unlimited number of steps
        self.assertEqual(instr[6]["type"], "simulate")
        self.assertEqual(instr[6]["num-trials"], 1)
        self.assertEqual(instr[6]["max-num-steps"], float("inf"))
        # Simulating multiple trials, each with unlimited number of steps
        self.assertEqual(instr[8]["type"], "simulate")
        self.assertEqual(instr[8]["num-trials"], 20)
        self.assertEqual(instr[8]["max-num-steps"], float("inf"))
        # Simulating multiple trials, each with a single step
        self.assertEqual(instr[4]["type"], "simulate")
        self.assertEqual(instr[4]["num-trials"], 20)
        self.assertEqual(instr[4]["max-num-steps"], 1)
        # Placing walls
        instr = SpatialNavigationTask("detour")._instructions
        self.assertEqual(len(instr), 7)
        self.assertEqual(instr[4]["type"], "place-wall")
        self.assertEqual(instr[4]["state"], (5,4))


if __name__ == "__main__":
    unittest.main()

