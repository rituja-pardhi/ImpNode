import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from collections import Counter


def get_degree_distribution(file_name):

    original_graph = nx.read_gml(file_name)

    degrees_original = [original_graph.degree(n) for n in original_graph.nodes()]
    degrees_normalised = [degree / max(degrees_original) for degree in degrees_original]

    degree_counts = Counter(degrees_normalised)
    probabilities = [count / len(degrees_normalised) for degree, count in degree_counts.items()]
    unique_degrees = np.array(list(degree_counts.keys()))

    cumulative_probs = np.cumsum(probabilities)

    return unique_degrees, cumulative_probs, degrees_original


def cm_model(unique_degrees, cumulative_probs, degrees_original, n_samples):
    degree_new = [unique_degrees[np.argmax(cumulative_probs >= i / n_samples)] for i in range(1, n_samples + 1)]

    mul_factor = np.mean(degrees_original) / np.mean(degree_new)
    degree_distribution = [int(mul_factor * sample) for sample in degree_new]

    if sum(degree_distribution) % 2 != 0:
        max_degree_idx = np.argmax(degree_distribution)
        degree_distribution[max_degree_idx] -= 1

    G = nx.configuration_model(degree_distribution)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    degree_weights = {int(node): degree / max_degree for node, degree in degrees.items()}
    nx.set_node_attributes(G, degree_weights, 'weight')
    return G
