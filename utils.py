import pandas as pd
import json
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jax.numpy as jnp
from graph_nets import utils_np
from graph_nets import utils_tf

def data_processing(train_file_path):
    """
    Function to process data from the given file paths.

    Args:
        train_file_path (str): Path to the training data file.
        val_file_path (str): Path to the validation data file.
        test_file_path (str): Path to the test data file.

    Returns:
        tuple: Tuple containing train_df, val_df, test_df, train_func, val_func, test_func
    """
    train_data = load_json_data(train_file_path)


    train_df = pd.DataFrame(train_data)


    train_func = extract_func_data(train_df)


    return train_df, train_func

def load_json_data(file_path):
    """
    Function to load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List containing the loaded JSON data.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Skip lines that cannot be parsed as JSON
    return data

def extract_func_data(df):
    """
    Function to extract function data from a DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        list: List containing the extracted function data.
    """
    func_data = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df.iloc[i, j] is not None:
                func_data.append(df.iloc[i, j]['func'])
    return func_data

def extract_func_embedding(func, word2vec_model):
    try:
        embedding = word2vec_model.wv[func]
        return np.array(embedding, dtype=np.float32)  # Convert to float32
    except KeyError:
        return np.zeros(word2vec_model.vector_size, dtype=np.float32)
def feature_engineering(train_func):
    """
    Function to perform feature engineering and graph construction.

    Args:
        train_func (list): List containing the function data for training.
        val_func (list): List containing the function data for validation.
        test_func (list): List containing the function data for testing.

    Returns:
        tuple: Tuple containing graphs_tuple
    """
    # Train Word2Vec model on train_func
    train_func_word2vec = Word2Vec(train_func, min_count=5)

    # Create a graph
    G = nx.Graph()

    # Extract func embeddings and filter out functions without valid embeddings
    func_embeddings = []
    valid_funcs = []
    for func in train_func:
        embedding = extract_func_embedding(func, train_func_word2vec)
        if np.any(embedding):  # Check if the embedding is not all zeros
            func_embeddings.append(embedding)
            valid_funcs.append(func)

    # Calculate cosine similarity between valid func embeddings
    cosine_sim_all = cosine_similarity(func_embeddings)



    # Add nodes and edges to the graph based on valid functions and cosine similarity
    for i, func in enumerate(valid_funcs):
        G.add_node(i, features=np.array(func_embeddings[i], dtype=np.float32))

    num_nodes = len(valid_funcs)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cosine_sim_all[i][j] > 0.01:
                G.add_edge(i, j, features=cosine_sim_all[i][j])

    # Convert the graph to a data dictionary
    data_dict = utils_np.networkx_to_data_dict(G)

    # Create a `GraphsTuple` object
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([data_dict])

    return graphs_tuple