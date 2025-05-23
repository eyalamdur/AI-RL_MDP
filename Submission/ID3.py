import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list,  target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        NUMBER_OF_SAMPLES = len(rows)
        for num_of_val in counts.values():
            p_val = (num_of_val / NUMBER_OF_SAMPLES)
            impurity -= p_val * np.log2(p_val)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        LEFT_SIZE = len(left)
        RIGHT_SIZE = len(right)
        NUMBER_OF_SAMPLES = len(left) + len(right)
        left_entropy = self.entropy(left, left_labels)
        right_entropy = self.entropy(right, right_labels)
        left_value = (LEFT_SIZE / NUMBER_OF_SAMPLES) * left_entropy
        right_value = (RIGHT_SIZE / NUMBER_OF_SAMPLES) * right_entropy
        info_gain_value = current_uncertainty - left_value - right_value
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        # Initialize the lists for the true and false rows and labels
        true_rows, true_labels, false_rows, false_labels = [], [], [], []  
        
        for sample_id, sample in enumerate(rows):
            if question.match(sample):
                true_rows.append(sample)
                true_labels.append(labels[sample_id])
            else:
                false_rows.append(sample)
                false_labels.append(labels[sample_id])
                
        # Get the information gain of the split
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        feature_list = self.get_fetaure_list(rows)
        
        for feature_id, feature_name in enumerate(feature_list):
            if feature_id == 0:
                continue
            for i in range(len(rows)):
                value = rows[i][feature_id]
                question = Question(feature_name ,feature_id, value)
                gain, true_rows, true_labels, false_rows, false_labels = self.partition(rows, labels, question, current_uncertainty)
                if gain > best_gain:
                    best_gain, best_question = gain, question
                    best_true_rows, best_true_labels = true_rows, true_labels
                    best_false_rows, best_false_labels = false_rows, false_labels
        # ========================
        
        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer. 

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        gain, best_question, true_rows, true_labels, false_rows, false_labels = self.find_best_split(rows, labels)
        
        # If we can't split the data anymore, we return a leaf node
        if gain == 0:
            return Leaf(rows, labels)
        
        # Otherwise, we build the tree recursively
        true_branch = self.build_tree(true_rows, true_labels)
        false_branch = self.build_tree(false_rows, false_labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if isinstance(node, Leaf):
            return self.find_leaf_prediction(node)
        
        # If the question is true, we go to the true branch, otherwise we go to the false branch
        if node.question.match(row):
            prediction = self.predict_sample(row, node.true_branch)
        else:
            prediction = self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = []
        for sample in rows:
            y_pred.append(self.predict_sample(sample))
        y_pred = np.array(y_pred)
        # ========================

        return y_pred

    def find_leaf_prediction(self, leaf: Leaf):
        """
        Find the prediction of a leaf node.
        :param leaf: the leaf node.
        :return: the prediction of the leaf node.
        """
        leaf_labels = leaf.predictions
        max_count = 0
        diagnosis = None

        for label in leaf_labels.keys():
            if leaf_labels[label] > max_count:
                diagnosis = label
                max_count = leaf_labels[label]
                
        return diagnosis
    
    def get_fetaure_list(self, rows):
        """
        Get the list of features from the rows.
        :param rows: array of samples
        :return: list of features.
        """
        return [i for i in range(len(rows[0]))]