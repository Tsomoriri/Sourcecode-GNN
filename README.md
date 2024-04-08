# Function Vulnerability Prediction

- This project aims to predict the vulnerability of a given function definition using a Graph Convolutional Network (GCN) implemented in Jax and Jraph. 
- We assume that a vulnerable function shares similarities in terms of word embeddings with the training dataset of vulnerable functions.
- The GCN will find patterns that the new function definition shares with the training dataset. It encapsulates this knowledge through a relationship via embedding with  the training graph. 
- The reason for using Jax is to utilise @jit and GPU accelerated features native to Jax. 
- It includes a Streamlit application for the user interface and a Docker container for easy deployment.

## Project Overview

The project follows the following flow of logic:
```
+---------------------+
| Function Definition |
+---------------------+
            |
            v
+---------------------+
|   Word Embedding    |
+---------------------+
            |
            v
+---------------------+
|  Graph Construction |
+---------------------+
            |
            v
+---------------------+
|   GCN Model         |
+---------------------+
            |
            v
+---------------------+
| Vulnerability Score |
+---------------------+
            |
            v
+---------------------+
| Vulnerability
            Prediction|
+---------------------+
```
-    Function Definition: The user provides a function definition as input.
-    Word Embedding: The function definition is converted into a word embedding vector using the Word2Vec model from the Gensim library.
-    Graph Construction: A graph is constructed where each node represents a word in the function definition, and edges are created based on the cosine similarity between word embeddings.
-    GCN Model: The constructed graph is fed into a Graph Convolutional Network (GCN) model implemented using Jax and Jraph.
-    Vulnerability Score: The GCN model outputs a vulnerability score for each node (word) in the graph.
-    Vulnerability Prediction: The vulnerability scores are aggregated, and a threshold is applied to determine whether the function is vulnerable or not.


## Demo
![Screenshot](model/img.jpg)
## Prerequisites
```
    Docker
    Python 3.7 or higher
```
Running the Docker Container

    Build the Docker image:
```    
docker build -t function-vulnerability-prediction .
```
    Run the Docker container:
```
docker run -p 8501:8501 function-vulnerability-prediction
```
This command maps the container's port 8501 to the host's port 8501, allowing you to access the Streamlit app.

    Access the Streamlit app by opening http://localhost:8501 in your web browser.

Running the Streamlit App Locally

If you prefer to run the Streamlit app locally without Docker, follow these steps:

    Install the required Python packages:
```
pip install -r requirements.txt
```
    Run the Streamlit app:
```
streamlit run app.py
```
    Access the Streamlit app by opening the provided URL in your web browser.

## Usage

-    Enter the function definition in the text area provided in the Streamlit app.
-    Click the "Predict" button to get the vulnerability prediction for the provided function definition.
-    The app will display whether the function is vulnerable or not, along with a visualization of the constructed graph.

# Project Structure

-    app.py: Contains the Streamlit app code for the user interface and prediction logic.
-    utils.py: Contains utility functions for data processing, feature engineering, and word embedding extraction.
-    models/gcn.py: Contains the implementation of the Graph Convolutional Network (GCN) model.
-    data/train_data.json: Contains the training data for the GCN model.
-   Dockerfile: Defines the Docker image configuration for easy deployment.
    requirements.txt: Lists the required Python packages for the project.
## References
- https://doi.org/10.1145/3607199.3607242
- JAX and jraph
