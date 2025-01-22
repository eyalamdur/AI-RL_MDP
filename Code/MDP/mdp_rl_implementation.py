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
                U_current[row, col] = belman_calculation(mdp, (row, col), U_final)
        
        # Update delta
        delta = max(np.max(np.abs(U_final - U_current), delta))
    # ========================

    return U_final


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ====== 

    # Initialize the policy array
    policy = np.empty((mdp.num_row, mdp.num_col), dtype=Action)
    rows, cols = mdp.num_row, mdp.num_col

    # Loop through each state in the MDP
    for row in range(rows):
        for col in range(cols):
            policy[row, col] = get_best_action(mdp, (row, col), mdp.actions, U)

    # ========================
    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    U = None
    # ====== YOUR CODE: ======
    rows, cols = mdp.num_row, mdp.num_col
    num_of_states = rows * cols
    
    # define gamma Reward vector and probability vector to find the Utility vector
    gamma = mdp.gamma
    R = get_reward_vector(mdp)
    P = get_probabilities_matrix(mdp, policy)
    I = np.eye(num_of_states)

    # solving the linear equation U = R - gamma * P * U => (I - gamma * P) * U = R [like Ax=b]
    V = np.zeros(num_of_states)
    V = np.linalg.solve(I - gamma * P, R)
    
    # returning from vector size num_of_states to matrix size rows * cols
    U = convert_V_to_U(mdp, V)
    # ========================
    return U


# TODO: Delete this function if not used
# TODO: threshold isn't defined in the lectures, need to understand
def policy_evaluation_2_option(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    U = None
    # ====== YOUR CODE: ======
    gamma = mdp.gamma
    theta = 10 ** (-4)
    rows, cols = mdp.num_row, mdp.num_col
    U = np.zeros((rows, cols))
    
    # Update terminal states utility
    for state in mdp.terminal_states:
        U[state[0], state[1]] = float(mdp.get_reward(state))
    
    # ensuring the Bellman equation for each state according to the existing U until delta < threshold
    while True:
        delta = 0
        for row in range(rows):
            for col in range(cols):
                temp = U[row, col]
                action = policy[row][col]
                reward = mdp.get_reward((row, col))
                if action is None or reward == "WALL":
                    U[row, col] = 0 if reward == "WALL" else float(reward)
                    continue
                action_utility = get_action_expected_utility(mdp, (row, col), action, U)
                # print("state = ", (row, col), "action = ", action, " action_utility = ", action_utility)
                U[row, col] = float(reward) + gamma * action_utility
                delta = max(delta, abs(temp - U[row, col]))

        if delta < theta:
            break
    # ========================
    return U


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    rows, cols = mdp.num_row, mdp.num_col
    U = np.zeros((rows, cols))
    optimal_policy = policy_init
    isChanged = True

    while isChanged:
        U = policy_evaluation(mdp, optimal_policy)
        isChanged = False
        for row in range(rows):
            for col in range(cols):
                state = (row, col)

                # Check if the state is a terminal state or a wall
                if state in mdp.terminal_states or not valid_state(mdp, state):
                    continue

                # Find the action with the best utility:
                best_action = get_best_action(mdp, state, mdp.actions, U)
                if best_action != optimal_policy[state[0]][state[1]]:
                    optimal_policy[state[0]][state[1]] = best_action
                    isChanged = True
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
    # Initialize the value function V and returns dictionary
    V = np.zeros((num_rows, num_cols))
    returns = { (row, col): [] for row in range(num_rows) for col in range(num_cols) }

    # Iterate through the number of episodes
    for _, episode_gen in enumerate(sim.replay(num_episodes)):
        G = 0
        steps = []

        # Collect steps in the episode
        for step in episode_gen:
            steps.append(step)
        
        # Calculate the return for each state in the episode in reverse order
        visited_states = set()
        for i in range(len(steps) - 1, -1, -1):
            state, reward, _, _ = steps[i]
            G = gamma * G + reward
            
            # First visit check
            if state not in visited_states:
                visited_states.add(state)
                returns[state].append(G)
                V[state[0], state[1]] = np.mean(returns[state])
                  
    # =========================
    return V


# --------------------- Helper functions --------------------- #
def get_action_expected_utility(mdp: MDP, state: Tuple[int, int], action: Action, U):
    # Given an MDP, a state, an action, and the utility of each state - U
    # return the expected utility of the given action in the given state
    #
    
    # Check if the action is a string and convert it to an Action
    if isinstance(action, str):
        action = Action[action]
        
    # Get the transition probabilities for the action
    transition_probs = mdp.transition_function[action]
    action_value = 0

    # Iterate through the transition probabilities and calculate expected utility for current action
    for action_index, probability in enumerate(transition_probs):
        action_by_index = str(list(Action)[action_index])
        next_state = mdp.step(state, action_by_index)

        # Add to the action value, weighted by probability
        action_value += probability * U[next_state]

    return action_value


def belman_calculation(mdp: MDP, state: Tuple[int, int], U):
    # Given an MDP, a state, and the utility of each state - U
    # return the Bellman equation calculation for the given state and action
    #

    if not valid_state(mdp, state):
        return 0
    
    # Get the reward of the current state
    reward = float(mdp.get_reward(state))

    # Terminal states have constant utility
    if state in mdp.terminal_states:
        return reward

    # Iterate through available actions and calculate the action value
    max_action_value = float('-inf')
    for action in mdp.actions:
        action_value = get_action_expected_utility(mdp, state, action, U)

        # Update max_action_value if needed
        max_action_value = max(max_action_value, action_value)

    # Return the calculated value based on the Bellman equation
    return reward + mdp.gamma * max_action_value


def get_best_action(mdp: MDP, state: Tuple[int, int], action_list: List[Tuple[Action, float]], U) -> Action:
    # Given an MDP, a state, a list of possible actions, and the utility of each state - U
    # return the best action for the given state
    # 

    # Check if the state is a terminal state or a wall
    if state in mdp.terminal_states or mdp.get_reward(state) == "WALL":
        return None

    best_action = None
    best_value = float('-inf')

    for action in action_list:
        value = get_action_expected_utility(mdp, state, action, U)

        # Check if this action is the best so far
        if value > best_value:
            best_value = value
            best_action = action

    return best_action


def valid_state(mdp: MDP, state: Tuple[int, int]) -> bool:
    return not mdp.get_reward(state) == "WALL"


def get_state_probabilty(mdp: MDP, state: Tuple[int, int], action: Action, next_state: Tuple[int, int]) -> float:
    # Given an MDP, a state, an action, and the next state
    # return the probability of the given action in the given state to reach the next state
    #

    # Get the transition probabilities for the action
    transition_probs = mdp.transition_function[action]

    # Iterate through the transition probabilities and return the probability for the next state
    for action_index, probability in enumerate(transition_probs):
        action_by_index = str(list(Action)[action_index])
        if next_state == mdp.step(state, action_by_index):
            return probability

    return 0


def state_num(mdp: MDP, row: int, col: int) -> int:
    # Given an MDP, a row, and a column
    # return the state number of the given row and column
    #
    return row * mdp.num_col + col


def get_probabilities_matrix(mdp: MDP, policy) -> np.array:
    # Calculate the probability of each state to reach the next state
    # given the action
    # P(s'|s,pi(a)) for all s, s', a
    #
    rows, cols = mdp.num_row, mdp.num_col
    num_of_states = rows * cols
    
    P = np.zeros((num_of_states, num_of_states))
    
    # calculate all the probabilities for P(s'|s,pi(a))
    for state_row in range(rows):
        for state_col in range(cols):
            state = state_num(mdp, state_row, state_col)
            action = policy[state_row][state_col]
            if isinstance(action, str):
                action = Action[action]
            if action is None:
                continue
            transition_probs = mdp.transition_function[action]
            for action_index, probability in enumerate(transition_probs):
                action_by_index = str(list(Action)[action_index])
                next_state_pos = mdp.step((state_row, state_col), action_by_index)
                next_state = state_num(mdp, next_state_pos[0], next_state_pos[1])
                P[state, next_state] += probability
                
    return P


def get_reward_vector(mdp: MDP) -> np.array:
    # Calculate the reward vector
    # R(s) for all s
    #
    
    rows, cols = mdp.num_row, mdp.num_col
    num_of_states = rows * cols
    
    # calculate all the rewards for R(s)
    R = np.zeros(num_of_states)
    for row in range(rows):
        for col in range(cols):
            state = state_num(mdp, row, col)
            R[state] = float(mdp.get_reward((row, col))) if mdp.get_reward((row, col)) != "WALL" else 0
    
    return R


def convert_V_to_U(mdp: MDP, V: np.array):
    # Convert the utility vector to a utility matrix
    # U(s) for all s
    #
    rows, cols = mdp.num_row, mdp.num_col
    U = np.zeros((rows, cols))
    
    # convert the utility vector to a utility matrix
    for row in range(rows):
        for col in range(cols):
            state = state_num(mdp, row, col)
            U[row][col] = V[state]
            
    # Update the utility matrix with terminal states and walls
    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in mdp.terminal_states:
                U[state[0]][state[1]] = float(mdp.get_reward(state))
            if not valid_state(mdp, state):
                U[state[0]][state[1]] = 0
    
    return U