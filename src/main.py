from utils import data_processing, feature_engineering , extract_func_embedding
from models.gcn import gcn_definition, optimize_model
import jax.numpy as jnp
import haiku as hk
from gensim.models import Word2Vec
import jraph

# Specify the file paths
train_file_path = '/home/sushen/projects/Sourcecode-GNN/data/train_data.json'
val_file_path = '/home/sushen/projects/Sourcecode-GNN/data/val_data.json'
test_file_path = '/home/sushen/projects/Sourcecode-GNN/data/test_data.json'

# Process data
train_df, train_func = data_processing(train_file_path)

# Perform feature engineering and graph construction
graphs_tuple = feature_engineering(train_func)

# Set all target labels to 1
train_targets = jnp.ones(len(train_func))

# Encapsulate the target labels in the graph
graphs_tuple = graphs_tuple._replace(globals=train_targets)

Jgraph = jraph.GraphsTuple(
    nodes=jnp.array(graphs_tuple.nodes),
    edges=jnp.array(graphs_tuple.edges),
    senders=jnp.array(graphs_tuple.senders),
    receivers=jnp.array(graphs_tuple.receivers),
    globals=jnp.array(graphs_tuple.globals),
    n_node=jnp.array(graphs_tuple.n_node),
    n_edge=jnp.array(graphs_tuple.n_edge)
)

# Define the GCN model
network = hk.without_apply_rng(hk.transform(gcn_definition))

# Train the GCN model
trained_params = optimize_model(network, Jgraph, num_steps=15)

# Function to predict vulnerability of a new function
def predict_vulnerability(func, trained_params,train_func):
    # Convert the function to a node feature vector
    train_func.append(func)
    func_vec = Word2Vec(train_func, min_count=5)
    func_embedding = extract_func_embedding(func, func_vec)

    # Create a new node with the function embedding
    new_node = jnp.array([func_embedding])

    # Create a new GraphsTuple with the new node
    new_graph = jraph.GraphsTuple(
        nodes=new_node,
        edges=jnp.zeros((0,), dtype=jnp.float32),
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals=jnp.zeros((1,), dtype=jnp.float32),
        n_node=jnp.array([1]),
        n_edge=jnp.array([0])
    )

    # Apply the trained GCN model to the new graph
    predicted_graph = network.apply(trained_params, new_graph)

    # Get the predicted vulnerability score for the new node
    vulnerability_score = predicted_graph.nodes[0, 1]

    # Threshold the score to determine if the function is vulnerable or not
    is_vulnerable = bool(vulnerability_score > 0.5)

    return is_vulnerable

# Example usage
new_func = "def some_function():"
is_vulnerable = predict_vulnerability(new_func, trained_params, train_func)
print(f"Function '{new_func}' is {'vulnerable' if is_vulnerable else 'not vulnerable'}")