from statistics import stdev
from math import log
from data import extract

# overlooks a lot of empty checking and what not
# simplified greatly because we are using binary features and binary classification

# data is matrix where each row is an example, each column is an attribute and last column is label
# attribute is the column number of desired attribute
# target is the desired label
def entropy(data, attribute, target):
    # simplification here, knowing that the x values are only 1/0 (True/False)
    one_count = 0
    one_positive = 0
    zero_count = 0
    zero_positive = 0
    for instance in data:
        if instance[attribute]:
            one_count += 1
            if instance[-1] == target:
                one_positive += 1
        else:
            zero_count += 1
            if instance[-1] == target:
                zero_positive += 1

    # if examples belong to one category, entropy is 0
    if one_count == 0 or zero_count == 0:
        return 0

    # entropy of each possible value
    one_negative = (one_count - one_positive) / one_count
    one_positive = one_positive / one_count
    if one_negative == 0 or one_positive == 0:
        h_one = 0
    else:
        h_one = (-1 * one_positive * log(one_positive, 2)) - (one_negative * log(one_negative, 2))

    zero_negative = (zero_count - zero_positive) / zero_count
    zero_positive = zero_positive / zero_count
    if zero_negative == 0 or zero_positive == 0:
        h_zero = 0
    else:
        h_zero = (-1 * zero_positive * log(zero_positive, 2)) - (zero_negative * log(zero_negative, 2))

    # expected entropy
    instances = len(data)
    return (one_count / instances * h_one) + (zero_count / instances * h_zero)

def entropy_label(data, target):
    target_count = 0
    for instance in data:
        if instance[-1] == target:
            target_count += 1

    instances = len(data)
    target_ratio = target_count / instances
    non_target_ratio = (instances - target_count) / instances
    return (-1 * target_ratio * log(target_ratio, 2)) - (non_target_ratio * log(non_target_ratio, 2))

def info_gain(data, attribute, target):
    label_ent = entropy_label(data, target)
    ent = entropy(data, attribute, target)
    return label_ent - ent

def best_attr(data, attributes, target):
    max_attr = attributes[0]
    max_gain = 0

    for attribute in attributes:
        curr_gain = info_gain(data, attribute, target)
        if curr_gain > max_gain:
            max_gain = curr_gain
            max_attr = attribute


    return max_attr

# generates set of instances where attribute is equal to value
def subset(data, attribute, value):
    set = []
    for instance in data:
        if instance[attribute] == value:
            set.append(instance)

    return set

def common_label(data):
    pos_count = 0
    for instance in data:
        if instance[-1]:
            pos_count += 1

    return pos_count > (len(data) - pos_count)

class TreeNode:
    # attribute: feature number, represented as the column in the data
    # branches: a pair where 0th element is node to go to if True and vice versa for 1st element
    # note: a leaf node will have no branches
    def __init__(self, attribute, branches):
        self.attribute = attribute
        self.branches = branches

# depth means max depth
def build_tree(data, attributes, target, depth):
    #print("Level")
    labels = set()
    for instance in data:
        labels.add(instance[-1])

    # base case: no more examples or attributes
    if not data or len(attributes) == 0:
        return TreeNode(common_label(data), None)
    # base case: all examples have same label
    if len(labels) == 1:
        return TreeNode(data[0][-1], None)
    if depth == 0:
        return TreeNode(common_label(data), None)

    # Recursive case: find best possible root node
    attr = best_attr(data, attributes, target)

    # True and False subset
    pos_set = subset(data, attr, True)
    neg_set = subset(data, attr, False)

    branches = [None, None]
    common = common_label(data)
    attributes.remove(attr) # drop the best attribute (consumed)

    # if empty, add common leaf node
    if not pos_set:
        branches[0] = TreeNode(common, None)
    else:
        branches[0] = build_tree(pos_set, attributes, target, depth -1)

    if not neg_set:
        branches[1] = TreeNode(common, None)
    else:
        branches[1] = build_tree(pos_set, attributes, target, depth - 1)

    return TreeNode(attr, branches) # root

class Tree:
    def __init__(self, data, target, depth=100):
        self.data = data
        self.target = target
        self.attributes = list(range(0, len(data[0]) - 1)) # -1 because the last column is a label not feature
        self.root = build_tree(self.data, self.attributes, self.target, depth)

    def classify(self, instance):
        curr = self.root # I hope this does not mess up the root
        while curr:
            if not curr.branches: # leaf node
                break

            next = instance[curr.attribute] # returns 0 or 1
            curr = curr.branches[next]

        return curr.attribute

    def compute_depth(self, root):
        if root.branches == None:
            return 0
        else:
            return max(self.compute_depth(root.branches[0]), self.compute_depth(root.branches[1])) + 1

    def depth(self):
        return self.compute_depth(self.root)

    def print(self):
        return 111

# Return average cross-validated error, 3 significant figures
def cross_validate(partitions, target, depth=None):
    folds = len(partitions)
    error = []
    for i in range(0, folds):
        train_data = []
        for d in range(1, folds):
            train_data += partitions[(i + d) % folds]

        if depth is None:
            tree = Tree(train_data, target)
        else:
            tree = Tree(train_data, target, depth)
        err_count = 0
        for instance in partitions[i]:
            predict = tree.classify(instance)
            if predict != instance[-1]:
                err_count += 1

        error.append(err_count / len(partitions[i]))

    print("Depth", depth, "standard deviation:", round(stdev(error), 3))

    return round(sum(error) / folds, 3)

