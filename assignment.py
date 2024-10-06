import pandas as pd
import numpy as np
from collections import Counter

# Sample dataset


df = pd.read_csv('data.csv')

def entropy(target_col):
    # Calculate the entropy of a target column
    elements, counts = np.unique(target_col, return_counts=True)
    
    entropy = sum((-count / sum(counts)) * np.log2(count / sum(counts)) for count in counts)
    return entropy

def _entropy(y):
    elements, counts = np.unique(y, return_counts=True)

    entropy = -sum((count / sum(counts)) * np.log2(count / sum(counts)) for count in counts)

    # print(entropy)
    return entropy

def information_gain(data, split_attribute_name, target_name):
    # Calculate the Information Gain of a split
    total_entropy = _entropy(data[target_name].values)
    values, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    # print(data[split_attribute_name], values, counts)
    
    weighted_entropy = sum((counts[i] / np.sum(counts)) * _entropy(data.where(data[split_attribute_name] == values[i]).dropna()[target_name]) for i in range(len(values)))
    gain = total_entropy - weighted_entropy
    # print(total_entropy, weighted_entropy, gain)
    return gain

def best_feature_to_split(data, target_name):
    # Determine the best feature to split on
    features = data.columns[:-1]  # Exclude target column
    print(f"finding information gains for {features}")
    info_gains = [information_gain(data, feature, target_name) for feature in features]
    print(info_gains)
    print(f"chosing {features[np.argmax(info_gains)]}")
    return features[np.argmax(info_gains)]

def create_decision_tree(data, target_name):
    # print(data)
    # Base case: if all target values are the same, return that value
    if len(np.unique(data[target_name])) == 1:
        return np.unique(data[target_name])[0]

    # Base case: if no more features, return the majority class
    if len(data.columns) == 1:
        print("col is 1", Counter(data[target_name], Counter(data[target_name]).most_common(1)))
        return Counter(data[target_name]).most_common(1)[0][0]

    # Find the best feature to split on
    best_feature = best_feature_to_split(data, target_name)
    tree = {best_feature: {}}

    print(f"best feature dataset:", f"{data[best_feature]}", data[best_feature][target_name])
    # Split the data on the best feature
    for value in np.unique(data[best_feature]):
        subset = data.where(data[best_feature] == value).dropna()
        # subset = subset.drop(columns=[best_feature])  # Exclude the current best feature
        print(f"subset for {value}")
        print(f"{subset}")
        # Recursively create the subtree
        subtree = create_decision_tree(subset, target_name)
        tree[best_feature][value] = subtree

    return tree

# Creating the decision tree
decision_tree = create_decision_tree(df, 'buys_computer')

# Display the decision tree
import pprint
pprint.pprint(decision_tree)
# print(best_feature_to_split(df, 'buys_computer'))
