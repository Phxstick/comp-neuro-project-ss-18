import os
import re
import numpy as np


# Action identifiers
GO_TOP = 0
GO_RIGHT = 1
GO_BOTTOM = 2
GO_LEFT = 3
GET_REWARD = 4

# Final state has ID 0
FINAL_STATE = 0

# Directory containing task data
TASKS_PATH = "../tasks"


class SpatialNavigationTask:
    """ Navigation task in a grid world where the agent can move into the
        directly neighbored cells if they're accessible. In a single trial,
        the agent starts from one of the specified starting positions and has
        to reach a goal position containing a reward which it can collect.

        A task is defined by a grid and a sequence of instructions which can
        change the environment (place rewards and alter the grid) and run trials
        after setting starting points for the agent.
    """

    def __init__(self, task_name):
        self._task_name = task_name
        self._load_grid()
        self._load_instructions()

    def get_name(self):
        return self._task_name

    def get_grid(self):
        return self._original_grid.copy()

    def get_grid_size(self):
        return self._grid_size

    def get_marked_states(self):
        return self._marked_positions.copy()

    def get_num_states(self):
        return len(self._original_grid) + 1  # +1 for final state (not in grid)

    def get_num_actions(self):
        return 5  # 4 directions to move in, 1 action to collect the reward

    def run(self, agent):
        """ Run this task for the given reinforcement learning agent.
            Return the value function of the agent as well as state of the
            grid at the end of the task. """
        agent.reset()
        grid = self._original_grid.copy()
        starting_states = None
        # Execute all task instructions in order
        for instruction in self._instructions:
            if instruction["type"] == "set-start":
                starting_states = instruction["states"]
            elif instruction["type"] == "place-reward":
                grid[instruction["state"]] = instruction["value"]
            elif instruction["type"] == "place-wall":
                del grid[instruction["state"]]
            elif instruction["type"] == "simulate":
                if starting_states is None:
                    raise Exception("Task Error: at least one starting position"
                        " for the agent must be set before simulating.")
                for trial_number in range(instruction["num-trials"]):
                    # Choose one of the starting states uniformly at random
                    state_index = np.random.randint(len(starting_states))
                    current_state = starting_states[state_index]
                    step = 0
                    # Keep simulating until either the maximum number of steps
                    # is reached or the agent has entered the final state.
                    while (current_state != FINAL_STATE and
                            step < instruction["max-num-steps"]):
                        current_state = \
                                self._execute_step(grid, agent, current_state)
                        step += 1
        return self._transform(agent.get_value_function()), grid

    def _transform(self, array):
        """ Given an array where index i holds a value for state with ID i,
            return a dictionary which maps the coordinates (x, y) of the state 
            with ID i to its value.
            Used to tranform agent's internal data structures to a
            more semantic representation corresponding to a 2D-grid). """
        mapping = dict()
        for state_id in range(1, len(array)):  # Skip ID 0 (final state)
            coord = self._state_id_to_coord[state_id]
            mapping[coord] = array[state_id]
        return mapping

    def _execute_step(self, grid, agent, state):
        """ Simulate a single step of the agent in the grid world.
            Returns the new state of the agent after the step. """
        x, y = state
        reward = grid[state]
        # Define a transition function for this state which, given an action,
        # returns a dictionary mapping each successor state to a probability
        def transition_func(action):
            if action == GO_TOP:
                return { self._coord_to_state_id[x, y - 1]: 1.0 }
            elif action == GO_RIGHT:
                return { self._coord_to_state_id[x + 1, y]: 1.0 }
            elif action == GO_BOTTOM:
                return { self._coord_to_state_id[x, y + 1]: 1.0 }
            elif action == GO_LEFT:
                return { self._coord_to_state_id[x - 1, y]: 1.0 }
            elif action == GET_REWARD:
                return { FINAL_STATE: 1.0 }
            else:
                raise Exception("'%s' is not a valid action." % action)
        # Gather all possible actions in this step (collect a reward if it
        # exists or move to one of the neighboring accessible cells)
        available_actions = []
        if reward:
            available_actions.append(GET_REWARD)
        else:
            if (x, y - 1) in grid:
                available_actions.append(GO_TOP)
            if (x + 1, y) in grid:
                available_actions.append(GO_RIGHT)
            if (x, y + 1) in grid:
                available_actions.append(GO_BOTTOM)
            if (x - 1, y) in grid:
                available_actions.append(GO_LEFT)
        # Let agent choose an action, execute it and observe the result
        s_id = self._coord_to_state_id[state]
        action = agent.choose_action(s_id, available_actions, transition_func)
        possible_transitions = transition_func(action).items()
        successor_states, transition_probas = zip(*possible_transitions)
        next_s_id = np.random.choice(successor_states, p=transition_probas)
        next_state = self._state_id_to_coord[next_s_id]
        if action != GET_REWARD:
            reward = 0
        agent.observe(s_id, action, reward, next_s_id)
        return next_state

    def _load_grid(self):
        """ Load the grid for the chosen task from the corresponding .grid file.
        
        Dots are walkable terrain, positions numbered from 1 to 9 are stored
        for later reference, everything else is considered inaccessible terrain.

        The grid is stored as a dictionary mapping coordinate tuples (x, y) to
        the reward which is earned in the state at this position (initially 0).
        """
        grid_path = os.path.join(TASKS_PATH, self._task_name + ".grid")
        if not os.path.exists(grid_path):
            raise Exception("No grid found for task '%s'!" % self._task_name)
        self._marked_positions = dict()
        self._original_grid = dict()
        height = 0
        width = 0
        # Parse the grid file line by line
        with open(grid_path, "r") as grid_file:
            for y, line in enumerate(grid_file):
                line = line[:-1]
                height = max(height, y + 1)
                for x, char in enumerate(line):
                    width = max(width, x + 1)
                    if char == '.':
                        self._original_grid[x, y] = 0
                    elif '0' <= char <= '9':
                        self._original_grid[x, y] = 0
                        self._marked_positions[int(char)] = (x, y)
        self._grid_size = (width, height)
        # The agent interface only accepts numbers between 0 and num_states - 1,
        # so a mapping from coordinates to state ids and vice versa is created.
        self._coord_to_state_id = dict()
        self._state_id_to_coord = dict()
        self._state_id_to_coord[FINAL_STATE] = FINAL_STATE
        state_id = 1  # 0 is already used for the final state
        for coord in self._original_grid:
            self._coord_to_state_id[coord] = state_id
            self._state_id_to_coord[state_id] = coord
            state_id += 1

    def _load_instructions(self):
        """ Load the sequence of instructions for the chosen task from the
            corresponding .task file. """
        instructions_path = os.path.join(TASKS_PATH, self._task_name + ".task")
        if not os.path.exists(instructions_path):
            raise Exception("No instructions found for task '%s'!"
                            % self._task_name)
        self._instructions = []
        with open(instructions_path, "r") as instructions_file:
            for line_number, line in enumerate(instructions_file):
                instruction = dict()
                # Remove newline character and trailing spaces at the line end
                line = line.strip()
                # Skip empty lines and comments
                if len(line) == 0 or line.startswith('#'):
                    continue
                # Setting starting points
                match = re.match(r"set start to ([1-9, ]+)", line)
                if match:
                    instruction["type"] = "set-start"
                    # Multiple starting points separated by comma can be given
                    instruction["states"] = [self._marked_positions[int(s)]
                                             for s in match.group(1).split(",")]
                    self._instructions.append(instruction)
                    continue
                # Placing rewards
                match = re.match(r"place reward (\d+) at (\d+)", line)
                if match:
                    instruction["type"] = "place-reward"
                    instruction["value"] = int(match.group(1))
                    instruction["state"] = \
                            self._marked_positions[int(match.group(2))]
                    self._instructions.append(instruction)
                    continue
                # Placing walls
                match = re.match(r"place wall at (\d+)", line)
                if match:
                    instruction["type"] = "place-wall"
                    instruction["state"] = \
                            self._marked_positions[int(match.group(1))]
                    self._instructions.append(instruction)
                    continue
                # Starting simulations
                match = re.match(
                        r"simulate( (\d+) steps?)?( (\d+) times?)?", line)
                if match:
                    instruction["type"] = "simulate"
                    # Get max number of steps (if given, unlimited by default)
                    if match.group(1):
                        instruction["max-num-steps"] = int(match.group(2))
                    else:
                        instruction["max-num-steps"] = float("inf")
                    # Get number of trials (if given, 1 by default)
                    if match.group(3):
                        instruction["num-trials"] = int(match.group(4))
                    else:
                        instruction["num-trials"] = 1
                    self._instructions.append(instruction)
                    continue
                # If neither of the previous cases matched, report as invalid
                raise Exception("Invalid instruction in file %s, line %d:  '%s'"
                                % (instructions_path, line_number, line))

