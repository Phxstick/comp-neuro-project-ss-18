import numpy as np
from numpy.linalg import inv


class RL:
    """ Base class for reinforcement learning agents. """

    def __init__(self, num_states, num_actions, gamma=0.95, epsilon=0.1):
        """ Initialize with the number of states and actions.
            Set discount factor gamma and epsilon for an epsilon-greedy policy.
        """
        self._num_states = num_states
        self._num_actions = num_actions
        self._gamma = gamma
        self._epsilon = epsilon

    def __str__(self):
        """ String representation of the agent. """
        raise NotImplementedError

    def reset(self):
        """ Reset all internal data structures. """
        raise NotImplementedError
    
    def get_value(self, state):
        """ Get the current value associated with the given state. """
        raise NotImplementedError

    def get_value_function(self):
        """ Return mapping of each known state to its value. """
        value_function = dict()
        for state in range(self._num_states):
            value_function[state] = self.get_value(state)
        return value_function

    def choose_action(self, state, available_actions, transition_function=None):
        """ Return the action to be executed in the given state among the given
            available actions. If the environment is locally known, a transition
            function can be given, which for each available action returns a
            dictionary mapping each successor state to a non-zero probability.
        """
        # Epsilon greedy: choose random action with probability epsilon
        choose_random = np.random.binomial(1, self._epsilon)
        if choose_random:
            return np.random.choice(available_actions)
        # Otherwise choose action which maximizes value function of next state
        else:
            best_action = -1
            best_value = -float("inf")
            for action in available_actions:
                transition_probas = transition_function(action)
                # Sum values of next states weighted by transition probability
                next_value = sum(transition_probas[next_state] *
                                 self.get_value(next_state)
                                 for next_state in transition_probas)
                if next_value > best_value:
                    best_value = next_value
                    best_action = action
            if best_value > 0:
                return best_action
            else:
                # If no action has a value better than 0, choose randomly again
                return np.random.choice(available_actions)

    def observe(self, prev_state, action, reward, next_state):
        """ Update internal data structures after a transition given the reward,
            the action, the previous state, and the next state. """
        raise NotImplementedError


class SRTD(RL):
    """ Algorithm 1: The original successor representation (SR-TD).

    Learns both the successor matrix and weights in parallel by TD learning.
    """

    def __init__(self, num_states, num_actions, gamma=0.95, epsilon=0.1,
                 alpha_td=0.3, alpha_sr=0.3):
        """
        Args:
           alpha_td: learning rate for weights of successor states.
           alpha_sr: learning rate for future state occupancies.
        """
        super().__init__(num_states, num_actions, gamma, epsilon)
        self._alpha_td = alpha_td
        self._alpha_sr = alpha_sr
        self.reset()

    def __str__(self):
        return "SR-TD"

    def reset(self):
        num_states = self._num_states
        self._M = np.identity(num_states)  # Discounted future state occupancies
        self._w = np.zeros(num_states)  # Weights for successor states

    def get_value(self, state):
        return np.dot(self._M[state,:], self._w)
    
    def observe(self, prev_state, action, reward, next_state): 
        # Update successor matrix M
        self._M[prev_state, prev_state] += self._alpha_sr
        self._M[prev_state,:] += self._alpha_sr * (
                self._gamma * self._M[next_state,:] - self._M[prev_state,:])
        # Calculate value function error
        new_value = reward + self._gamma * self.get_value(next_state)
        old_value = self.get_value(prev_state)
        error = new_value - old_value
        # Update w proportionally to the error (and successor values)
        self._w += self._alpha_td * error * self._M[prev_state,:]


class SRMB(RL):
    """ Algorithm 2: Recomputation of the successor matrix (SR-MB).

    Learns a transition function T (depending on the policy) by TD-learning to
    recompute the successor matrix after each action.
    """

    def __init__(self, num_states, num_actions, gamma=0.95, epsilon=0.1,
                 alpha_td=0.3, alpha_pi=0.1):
        """
        Args:
           alpha_td: learning rate for weights of successor states.
           alpha_pi: learning rate for the action weights in the policy.
        """
        super().__init__(num_states, num_actions, gamma, epsilon)
        self._alpha_td = alpha_td
        self._alpha_pi = alpha_pi
        self.reset()

    def __str__(self):
        return "SR-MB"

    def reset(self):
        self._w = np.zeros(self._num_states)  # Weights for successor states
        # Discounted future state occupancies, recalculated after each action
        self._M = np.identity(self._num_states)
        # Transition probability matrix (depends on the current policy)
        self._T = np.identity(self._num_states)
        # The policy (weights for each action in each state), initially uniform.
        self._pi = np.full((self._num_states, self._num_actions),
                           1 / self._num_actions)

    def get_value(self, state):
        return np.dot(self._M[state,:], self._w)

    def observe(self, prev_state, action, reward, next_state): 
        # Calculate value function error
        new_value = reward + self._gamma * self.get_value(next_state)
        old_value = self.get_value(prev_state)
        error = new_value - old_value
        # Update w proportionally to the error (and successor values)
        self._w += self._alpha_td * error * self._M[prev_state,:]

    def choose_action(self, state, available_actions, transition_function=None):
        action = super().choose_action(state, available_actions,
                                       transition_function=transition_function)
        if action == 4:
            return action
        # Update the policy: increase weight for taken action, reduce for others
        self._pi[state,:] *= (1 - self._alpha_pi)
        self._pi[state, action] += self._alpha_pi
        # Update internal transition matrix based on the weights in the policy
        action_weight_sum = np.sum(self._pi[state,:][available_actions])
        self._T[state,:] = 0
        for action in available_actions:
            transition_probas = transition_function(action)
            for next_state in transition_probas:
                self._T[state, next_state] += self._pi[state, action] * \
                                              transition_probas[next_state] \
                # Divide by sum of weights to make it a valid proba distribution
                self._T[state, next_state] /= action_weight_sum
        # Recompute the successor matrix using the updated transition matrix
        self._M = inv(np.identity(self._num_states) - self._gamma * self._T)
        return action


class SRDyna(RL):
    """ Algorithm 3: Off-policy experience resampling (SR-Dyna).

    Replays previously experienced transitions to update the successor matrix.
    """

    def __init__(self, num_states, num_actions, gamma=0.95, epsilon=0.1,
                 alpha_td=0.3, alpha_sr=0.3):
        """
        Args:
           alpha_td: learning rate for weights of successor states.
           alpha_sr: learning rate for future state occupancies.
        """
        super().__init__(num_states, num_actions, gamma, epsilon)
        self._alpha_td = alpha_td
        self._alpha_sr = alpha_sr
        self.reset()

    def __str__(self):
        return "SR-Dyna"

    def reset(self):
        # Weights for state-action pairs
        self._w = np.zeros(self._num_states * self._num_actions)
        # Discounted occupancies of future state-action pairs
        self._H = np.identity((self._num_states * self._num_actions,
                               self._num_states * self._num_actions))
        # Stores experienced transition tuples
        self._samples = []

    def get_value(self, state):
        q_values = np.empty(self._num_actions)
        for action in range(self._num_actions):
            state_action = state * self._num_actions + action
            q_values[action] = np.dot(self._M[state_action,:], self._w)
        return np.max(q_values)

    def observe(self, prev_state, action, reward, next_state): 
        return NotImplementedError # TODO

    def choose_action(self, state, available_actions, transition_function=None):
        return NotImplementedError # TODO
