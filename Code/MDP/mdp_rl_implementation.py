from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    
    # Initialize the utility matrix (with type float64)
    U_current = np.array(U_init, dtype=np.float64)
    rows, cols = mdp.num_row, mdp.num_col
    delta = float('inf')
    
    while delta > epsilon * (1 - mdp.gamma) / mdp.gamma:
        delta = 0
        U_final = np.copy(U_current)

        # Update the utility of each state
        for row in range(rows):
            for col in range(cols):
                U_current[row,col] = belman_calculation(mdp, (row, col), U_final)

        # Update delta
        delta = max(np.max(np.abs(U_final- U_current), delta))
    # ========================
    return U_final


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 

    # ========================
    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    U = None
    # TODO:
    # ====== YOUR CODE: ======

    # ========================
    return U


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======

    # ========================
    return optimal_policy


def mc_algorithm(
        sim,
        num_episodes,
        gamma,
        num_rows=3,
        num_cols=4,
        actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
        policy=None,
):
    # Given a simulator, the number of episodes to run, the number of rows and columns in the MDP, the possible actions,
    # and an optional policy, run the Monte Carlo algorithm to estimate the utility of each state.
    # Return the utility of each state.

    V = None

    # ====== YOUR CODE: ======

    # =========================

    return V


# --------------------- Helper functions --------------------- #
def belman_calculation(mdp, state, U_final):
    # Given an MDP, a state, and the last utility of each state - U_final
    # return the Belman equation calculation for the given state and action
    #
    
    # Terminal states have constant utility
    if state in mdp.terminal_states:
        return float(mdp.get_reward(state))
    
    # Iterate through available actions and calculate the action value
    max_action_value = float('-inf') 
    for action in mdp.actions: 
        action_value = 0

        # Get the transition probabilities for the action
        transition_probs = mdp.transition_function[action]
        
        # Iterate through the transition probabilities and calculate expected utility for current action
        for action_index, probability in enumerate(transition_probs):
            action_by_index = str(list(Action)[action_index])
            next_state = mdp.step(state, action_by_index)

            # Add to the action value, weighted by probability
            action_value += probability * U_final[next_state[0]][next_state[1]] 

        # Update max_action_value if needed
        max_action_value = max(max_action_value, action_value)

    # Get the reward of the current state
    reward = float(mdp.get_reward(state)) if mdp.get_reward(state) != "WALL" else 0    
    
    # Return the calculated value based on the Bellman equation
    return reward + mdp.gamma * max_action_value
