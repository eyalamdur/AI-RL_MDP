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
                if valid_state(mdp, (row, col)):
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
    R = np.zeros(num_of_states)
    for row in range(rows):
        for col in range(cols):
            state = state_num(mdp, row, col)
            R[state] = float(mdp.get_reward((row, col))) if mdp.get_reward((row, col)) != "WALL" else 0
    P = np.zeros((num_of_states, num_of_states))
    I = np.ones((num_of_states, num_of_states))
    # print(policy)
    # print(type(policy))
    # calculate all the probabilities for P(s'|s,pi(a))
    for state_row in range(rows):
        for state_col in range(cols):
            state = state_num(mdp, state_row, state_col)
            # print("row: ", state_row, " col: ", state_col)
            action = policy[state_row][state_col]
            # print(type(action))
            if isinstance(action, str):
                action = Action[action]
            if action is None:
                P[state, state] = 1
                continue
            transition_probs = mdp.transition_function[action]
            for action_index, probability in enumerate(transition_probs):
                action_by_index = str(list(Action)[action_index])
                next_state_pos = mdp.step((state_row, state_col), action_by_index)
                next_state = state_num(mdp, next_state_pos[0], next_state_pos[1])
                P[next_state, state] += probability

    # solving the linear equation U = R - gamma * P * U => (I - gamma * P) * U = R [like Ax=b]
    V = np.zeros(num_of_states)
    # print("R: ", R)
    # print("P: ")
    # print(P)
    # W = np.linalg.inv(I - gamma * P)
    # W = W.dot(R)
    V = np.linalg.solve(I - gamma * P, R)
    # returning from vector size num_of_states to matrix size rows * cols
    U = np.zeros((rows, cols))
    # print("V: ", V)
    # print("W: ", W)
    for row in range(rows):
        for col in range(cols):
            state = row * cols + col
            U[row][col] = V[state]

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in mdp.terminal_states:
                U[state[0]][state[1]] = float(mdp.get_reward(state))
            if not valid_state(mdp, state):
                U[state[0]][state[1]] = 0
    # gamma = mdp.gamma
    # # TODO: threshold isn't defined in the lectures, need to understand
    # theta = 10 ** (-4)
    # rows, cols = mdp.num_row, mdp.num_col
    # U = np.zeros((rows, cols))
    # # ensuring the Bellman equation for each state according to the existing U until delta < threshold
    # while True:
    #     delta = 0
    #     for row in range(rows):
    #         for col in range(cols):
    #             temp = U[row, col]
    #             action = policy[row, col]
    #             reward = mdp.get_reward((row, col))
    #             U[row, col] = reward + gamma * get_action_expected_utility(mdp, (row, col), action, U)
    #             # U[row, col] = reward + gamma *
    #             # sum(mdp.transition_function((next_row, next_col), (row, col), action) * U[next_row, next_col]
    #             # for next_row in range(rows) for next_col in range(cols))
    #             delta = max(delta, abs(temp - U[row, col]))
    #     if delta < theta:
    #         break
    # ========================
    # print("U: ", U)
    # print("U ", U)
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
    count = 0
    while isChanged and count < 10:
        count += 1
        U = policy_evaluation(mdp, optimal_policy)
        # mdp.print_policy(optimal_policy)
        # print("Utility:")
        # mdp.print_utility(U)
        isChanged = False
        for row in range(rows):
            for col in range(cols):
                state = (row, col)

                if state in mdp.terminal_states or not valid_state(mdp, state):
                    continue

                # find the action with the best utility:
                actions = list(mdp.transition_function.keys())
                best_value = float("-inf")
                best_action = optimal_policy[state[0]][state[1]]
                for action in mdp.actions:
                    action_utility = 0
                    for outcome_action_index in range(len(mdp.actions)):
                        prob_outcome_action = mdp.transition_function.get(action)[outcome_action_index]
                        outcome_action = actions[outcome_action_index]
                        new_state = mdp.step(state, outcome_action)
                        # checking if we hit a wall
                        if new_state != state:
                            action_utility += prob_outcome_action * U[new_state]
                    if action_utility > best_value:
                        best_action = action
                        best_value = action_utility
                if best_action != optimal_policy[state[0]][state[1]]:
                    optimal_policy[state[0]][state[1]] = best_action
                    isChanged = True

                # if exists_better_policy(mdp, optimal_policy, U, state):
                #     print(":)")
                #     optimal_policy[row][col] = get_best_action(mdp, state, mdp.actions, U)
                #     isChanged = True
    # ========================
    # mdp.print_policy(optimal_policy)
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
def get_action_expected_utility(mdp: MDP, state: Tuple[int, int], action: Action, U):
    # Given an MDP, a state, an action, and the utility of each state - U
    # return the expected utility of the given action in the given state
    #
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
        # if state == (0, 0) or state == (2, 0):
        #     print("P(", next_state, "|,", state, ",", action, ") =", probability, "U(", next_state, ") = ", U[next_state])
        action_value += probability * U[next_state]
    # if state == (0, 0) or state == (2, 0):
    #     print("state = ", state, "action = ", action, " action_value = ", action_value)
    return action_value


def belman_calculation(mdp: MDP, state: Tuple[int, int], U):
    # Given an MDP, a state, and the utility of each state - U
    # return the Bellman equation calculation for the given state and action
    #

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


def exists_better_policy(mdp: MDP, policy, U, state: Tuple[int, int]) -> bool:
    # Given a mdp, a policy, a utility array and a state,
    # return true if there is a better policy to the given
    # state and False otherwise
    if state in mdp.terminal_states or not valid_state(mdp, state):
        return False

    # find the action with the best utility:
    actions = list(mdp.transition_function.keys())
    best_value = float("-inf")
    best_action = policy[state[0]][state[1]]
    for action in mdp.actions:
        action_utility = 0
        for outcome_action_index in range(len(mdp.actions)):
            prob_outcome_action = mdp.transition_function.get(action)[outcome_action_index]
            outcome_action = actions[outcome_action_index]
            new_state = mdp.step(state, outcome_action)
            # checking if we hit a wall
            if new_state != state:
                action_utility += prob_outcome_action * U[new_state]
        if action_utility > best_value:
                best_action = action
                best_value = action_utility
    if best_action != policy[state[0]][state[1]]:
        return True
    return False

    best_action = get_best_action(mdp, state, mdp.actions, U)
    # if best_action is None:
        # print(state, ":(")
    best_value = get_action_expected_utility(mdp, state, best_action, U)
    policy_value = get_action_expected_utility(mdp, state, policy[state[0]][state[1]], U)
    best_value = round(best_value, 5)
    policy_value = round(policy_value, 5)
    epsilon = 1e-10
    if best_value > policy_value + epsilon:
        print("state = ", state, "action = ", policy[state[0]][state[1]], ", policy_value = ", policy_value,
              "best action = ",best_action, "best_value = ", best_value, )
        return best_value > policy_value


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
    return row * mdp.num_col + col
