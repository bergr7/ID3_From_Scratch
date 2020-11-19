# ID3 Algorithm implementation from scratch

import math
import numpy as np
import pandas as pd
from collections import deque


class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs =None


class DecisionTreeClassifier(object):
    def __init__(self, sample, attributes, labels):
        self.sample = sample
        self.attributes = attributes
        self.labels = labels
        self.labelCodes = list(set(labels))
        self.labelCodesCount = [labels.count(x) for x in self.labelCodes]
        self.node = None
        self.entropy = self.get_entropy([x for x in range(len(self.labels))])   # calculates the initial entropy

    def get_entropy(self, sample_ids):
        """ Calculates the entropy.
        Parameters
        __________
        :param sample_ids: list, List containing the samples ID's
        __________
        :return: entropy: float, Entropy.
        """
        labels = [self.labels[i] for i in sample_ids]
        label_count = [labels.count(x) for x in self.labelCodes]
        # calculate each term of the entropy and sum them
        entropy = sum([-x / len(sample_ids) * math.log(x / len(sample_ids), 2) if x else 0 for x in label_count])
        return entropy

    def get_information_gain(self, sample_ids, attribute_id):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param sample_ids: list, List containing the samples ID's
        :param attribute_id: int, attribute/feature ID
        __________
        :return: gain: float, the information gain for a given feature.
        """
        gain = self.get_entropy(sample_ids)
        sample_attributes = [self.sample[x][attribute_id] for x in sample_ids]
        attribute_vals = list(set(sample_attributes))
        attribute_vals_count = [sample_attributes.count(x) for x in attribute_vals]
        attribute_vals_id = [
            sample_ids[i]
            for i, x in enumerate(sample_attributes)
            if x == y
            for y in attribute_vals
        ]
        gain = gain - sum([vc / len(sample_ids) * self.get_entropy(vids)
                           for vc, vids in zip(attribute_vals_count, attribute_vals_id)])
        return gain

    def get_attribute_max_information_gain(self, sample_ids, attribute_ids):
        """Finds the attribute/feature that maximizes the information gain.
        Parameters
        __________
        :param sample_ids: list, List containing the samples ID's
        :param attribute_ids: list, List containing the attributes ID's
        __________
        :returns: string and int, attribute and attribute id.
        """
        attributes_entropy = [self.get_information_gain(sample_ids, x) for x in attribute_ids]
        max_id = attribute_ids[attributes_entropy.index(max(attributes_entropy))]
        return self.attributes[max_id], max_id

    def id3(self):
        """Initializes ID3 algorithm.

        :return: None
        """
        sample_ids = [x for x in range(len(self.sample))]
        attribute_ids = [x for x in range(len(self.attributes))]
        self.node = self.id3_recv(sample_ids, attribute_ids, self.node)

    def id3_recv(self, sample_ids, attribute_ids, node):
        if not node:
            node = Node()   # initialize current node
        labels_in_attributes = [self.labels[x] for x in sample_ids]
        # if all the example have the same class
        if len(set(labels_in_attributes)) == 1:
            node.value = self.labels[sample_ids[0]]
            return node     # return a leaf with that class
        if len(attribute_ids) == 0:     # if there are not more feature to compute
            node.value = max(set(labels_in_attributes), key=labels_in_attributes.count)     # labels mode
            return node     # return a leaf with the most probable class
        # else
        # Choose the feature that maximizes the information gain for the next node
        best_attr_name, best_attr_id = self.get_attribute_max_information_gain(sample_ids, attribute_ids)
        node.value = best_attr_name
        node.childs = []
        attributes = list(set([self.sample[x][best_attr_id] for x in sample_ids]))
        for attribute in attributes:
            child = Node()
            child.value = attribute     # add a branch from the node to each f value in out feature
            node.childs.append(child)       # append new child node to current node
            child_sample_ids = [x for x in sample_ids if self.sample[x][best_attr_id]]
            if not child_sample_ids:
                child.next = max(set(labels_in_attributes), key=labels_in_attributes.count)
            else:
                if attribute_ids and best_attr_id in attribute_ids:
                    to_remove = attribute_ids.index(best_attr_id)
                    attribute_ids.pop(to_remove)
                # recursively call the algo
                child.next = self.id3_recv(child_sample_ids, attribute_ids, child.next)
        return node

    def printTree(self):
        """It plots the decision tree.

        :return: None
        """
        if not self.node:
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = node.popleft()
            print(node.value)
            if node.childs:
                for child in node.childs:
                    print('({})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print(node.next)



