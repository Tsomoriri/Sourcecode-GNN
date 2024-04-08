import pickle
import streamlit as st
from utils import data_processing, feature_engineering, extract_func_embedding
from models.gcn import gcn_definition, optimize_model
import haiku as hk
from gensim.models import Word2Vec
import jraph
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp
from graph_nets import utils_np
from graph_nets import utils_tf

import jax




# Load the trained model parameters
with open('trained_model.pkl', 'rb') as f:
    trained_params = pickle.load(f)

# Load the trained Word2Vec model
func_vec = Word2Vec.load('func_vec.model')

# Define the GCN model
network = hk.without_apply_rng(hk.transform(gcn_definition))
def visualize_graph(G):
    nx_graph = G
    pos = nx.circular_layout(nx_graph)

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121)
    nx.draw(
        nx_graph,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color='skyblue',
        font_color='white')
    ax1.title.set_text('Predicted Node Assignments with GCN')
    fig.suptitle('GCN', y=-0.01)


def predict_vulnerability(func):
    # Convert the function to a node feature vector
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
    vulnerability_score = jnp.argmax(predicted_graph.nodes, axis=1)

    # Threshold the score to determine if the function is vulnerable or not
    is_vulnerable = bool(vulnerability_score > 0.5)

    return is_vulnerable


# Streamlit app


def main():
    st.set_page_config(page_title="Function Vulnerability Prediction", page_icon=":shield:", layout="wide")

    st.markdown("<h1 style='text-align: center;'>Function Vulnerability Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Predict the vulnerability of a function using Graph Convolutional Networks (GCN).</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Function Definition")
        function_def = st.text_area("Enter the function definition:", height=300)
        if st.button("Predict"):
            if function_def:
                is_vulnerable = predict_vulnerability(function_def)
                st.subheader("Prediction Result")
                if is_vulnerable:
                    st.error(f"The function is vulnerable.")
                else:
                    st.success(f"The function is not vulnerable.")
            else:
                st.warning("Please enter a function definition.")

    with col2:
        st.subheader("Node Assignments")
        st.markdown(predict_vulnerability(function_def))


    st.markdown("---")
    st.subheader("About")
    st.markdown("This app uses a Graph Convolutional Network (GCN) to predict the vulnerability of a given function definition. The model is trained on a dataset of vulnerable functions.")
    st.markdown("Enter a function definition in the text area and click the 'Predict' button to see the vulnerability prediction and the corresponding graph visualization.")

    st.sidebar.title("Options")
    st.sidebar.checkbox("Show advanced settings", False, key="advanced_settings")
    if st.sidebar.checkbox("Show model performance metrics", False, key="show_metrics"):
        st.subheader("Model Performance Metrics")
        # model performance metrics

    st.sidebar.title("Feedback")
    st.sidebar.text_area("Please provide your feedback or suggestions:")
    if st.sidebar.button("Submit Feedback"):
        # Handle feedback submission
        st.sidebar.success("Thank you for your feedback!")


if __name__ == '__main__':
    main()