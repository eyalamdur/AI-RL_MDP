from __future__ import annotations
from termcolor import colored
from enum import Enum


class Action(Enum):
    UP = "UP"
    DOWN = "DOWN"
    RIGHT = "RIGHT"
    LEFT = "LEFT"

    def __str__(self):
        return self.value


class MDP:
    def __init__(
            self,
            board,
            terminal_states,
            transition_function,
            gamma,
    ):
        self.board = board
        self.num_row = len(board)
        self.num_col = len(board[0])
        self.terminal_states = terminal_states
        self.actions = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.RIGHT: (0, 1),
            Action.LEFT: (0, -1),
        }
        self.transition_function = transition_function
        self.gamma = gamma

    # returns the next step in the env
    def step(self, state, action):
        if isinstance(action, str):
            action = Action[action]

        next_state = tuple(map(sum, zip(state, self.actions[action])))

        if (
                next_state[0] < 0
                or next_state[1] < 0
                or next_state[0] >= self.num_row
                or next_state[1] >= self.num_col
                or self.board[next_state[0]][next_state[1]] == "WALL"
        ):
            next_state = state
        return next_state

    def get_reward(self, state):
        return self.board[state[0]][state[1]]

    ###################### Print Utilities ######################

    def format_cell(
            self, r, c, value, is_terminal, is_wall
    ):
        if is_terminal:
            return " " + colored(value[:5].ljust(5), "red") + " |"
        elif is_wall:
            return " " + colored(value[:5].ljust(5), "blue") + " |"
        else:
            return " " + value[:5].ljust(5) + " |"

    def print_board(self, content):
        res = ""
        for r in range(self.num_row):
            res += "|"
            for c in range(self.num_col):
                val = str(content[r][c])
                is_terminal = (r, c) in self.terminal_states
                is_wall = self.board[r][c] == "WALL"
                res += self.format_cell(r, c, val, is_terminal, is_wall)
            res += "\n"
        print(res)

    def print_rewards(self):
        self.print_board(self.board)

    def print_utility(self, U):
        self.print_board(U)

    def print_policy(self, policy):
        self.print_board(policy)

    @staticmethod
    def load_mdp(board='board', terminal_states='terminal_states',
                 transition_function='transition_function', gamma=0.9):
        """
        Loads an MDP from the specified files.

        :param board: Filename for the board configuration.
        :param terminal_states: Filename for the terminal states.
        :param transition_function: Filename for the transition function.
        :param gamma: Discount factor.
        :return: An instance of MDP.
        """

        board_env = []
        with open(board, 'r') as f:
            for line in f.readlines():
                row = line[:-1].split(',')
                board_env.append(row)

        terminal_states_env = []
        with open(terminal_states, 'r') as f:
            for line in f.readlines():
                row = line[:-1].split(',')
                terminal_states_env.append(tuple(map(int, row)))

        transition_function_env = {}
        with open(transition_function, 'r') as f:
            for line in f.readlines():
                action, prob = line[:-1].split(':')
                prob = prob.split(',')
                action_key = Action(action)
                transition_function_env[action_key] = tuple(map(float, prob))

        mdp = MDP(board=board_env,
                  terminal_states=terminal_states_env,
                  transition_function=transition_function_env,
                  gamma=gamma)

        return mdp


def format_transition_function(transition_function, action_order=[Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]):
    result = {}

    for outer_action, inner_dict in transition_function.items():
        # If the action is not in the inner dictionary, the probability is 0.0
        ordered_tuple = tuple(inner_dict.get(action, 0.0) for action in action_order)

        result[outer_action] = ordered_tuple

    return result


def print_transition_function(transition_function, action_order=[Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]):
    needs_formatting = isinstance(next(iter(transition_function.values())), dict)

    formatted = format_transition_function(transition_function) if needs_formatting else transition_function

    for action in action_order:
        print(f"{action}: {str(formatted[action])[1:-1]}")
