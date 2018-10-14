"""
The code simualates reinforcement learning agents using different algorithms
based on the successor representation on spatial navigation tasks.

Agents are defined in rl.py.

Spatial navigation tasks consist of two kinds of files:
- .grid file containing the initial environment,
- .task file containing a sequence of instructions.
These files can be found in the directory /tasks and are processed in task.py.

The grids and the results of the simulations (value function and policy) are
visualized using functions in visualizer.py and output to the folder /results.
"""

import os
import numpy as np

from rl import *
from task import *
from visualizer import Visualizer


def compute_policy(value_func, grid):
    """ Compute a policy for the given value function by choosing the action
        which minimizes the value of the next state. """
    policy = dict()
    for state in value_func:
        # If the state has a reward, collect it
        if grid[state]:
            policy[state] = GET_REWARD
            continue
        # Get values of the neighboring states
        x, y = state
        NO_VALUE = -float("inf")
        neighboring_values = [
            value_func[x,y-1] if (x,y-1) in value_func else NO_VALUE,
            value_func[x+1,y] if (x+1,y) in value_func else NO_VALUE,
            value_func[x,y+1] if (x,y+1) in value_func else NO_VALUE,
            value_func[x-1,y] if (x-1,y) in value_func else NO_VALUE
        ]
        # If no neighbor has a value, don't set an action
        if all(value == NO_VALUE for value in neighboring_values):
            continue
        # Otherwise choose the action which leads to the best neighboring cell
        else:
            policy[state] = np.argmax(neighboring_values)
    return policy


def get_state_labels(task_name, marked_states):
    """ Given a mapping from numbers to state coordinates, return a mapping
        from state coordinates to labels corresponding to each number. """
    if task_name == "latent-learning":
        return { marked_states[1]: "S",
                 marked_states[2]: "R" }
    elif task_name == "detour":
        return { marked_states[1]: "S",
                 marked_states[2]: "R",
                 marked_states[3]: "B" }
    elif task_name == "policy-revaluation":
        return { marked_states[1]: "S1",
                 marked_states[2]: "R1",
                 marked_states[3]: "S2",
                 marked_states[4]: "R2" }
    else:
        raise Exception("Unknown task '%s'." % task_name)


def simulate(task_name, agent_type, num_repetitions=5, output_directory="."):
    """ Run an agent of given type on the navigation task with given name.
        Repeat the simulation multiple times and output median results. """
    task = SpatialNavigationTask(task_name)
    agent = agent_type(task.get_num_states(), task.get_num_actions())
    grid = task.get_grid()
    # For each grid state, keep a list of values from each run
    values = dict()
    for state in grid:
        values[state] = []
    # Repeatedly run the simulation and collect value functions
    print("Starting simulations for agent '%s' on task '%s'."
          % (agent, task_name))
    for run_number in range(num_repetitions):
        print("  Finished run %2d / %2d" % (run_number, num_repetitions),
              end="\r")
        value_function, final_grid = task.run(agent)
        for state in value_function:
            values[state].append(value_function[state])
    print("  Finished run %2d / %2d" % (num_repetitions, num_repetitions))
    # Compute the median value function and policy
    median_value_func = dict()
    for state in values:
        values_sorted = sorted(values[state])
        median_value_func[state] = values_sorted[int(len(values_sorted)/2)]
    median_policy = compute_policy(median_value_func, grid)
    # Visualize the grid and the marked states of the task
    width, height = task.get_grid_size()
    state_labels = get_state_labels(task_name, task.get_marked_states())
    output_path = os.path.join(output_directory, "%s-task.png" % task_name)
    Visualizer.draw_grid(grid, width, height, state_labels=state_labels,
                         output_path=output_path)
    # Visualize the value function and policy of the agent in the grid
    output_path = os.path.join(output_directory, "%s_%s.png"%(agent, task_name))
    Visualizer.draw_grid(final_grid, width, height, state_labels=state_labels,
                         value_function=median_value_func, policy=median_policy,
                         output_path=output_path)


if __name__ == "__main__":
    agent_types = [SRTD, SRMB, SRDyna]
    task_names = ["latent-learning", "detour", "policy-revaluation"]
    # Run each task for each agent
    for task_name in task_names:
        for agent_type in agent_types:
            simulate(task_name, agent_type, output_directory="../results")

