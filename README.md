# GNN-Based Edge Importance Learning for Route Optimization

A graph-based route optimization framework that uses a **2-layer Graph Convolutional Network (GCN)** to learn node representations and an **edge-importance model** to identify promising connections for route construction.

## Overview

Traditional route optimization methods primarily rely on explicit distance-based heuristics. This project explores a **Graph Neural Network (GNN)-assisted approach**, where relationships between cities are represented as graph edges and the model learns which edges are more important for constructing high-quality routes.

The framework uses **100 cities** as graph nodes and generates pseudo-good routes using **Nearest Neighbor + 2-opt**. These routes provide supervision for training an edge-importance model.

### Key Idea

```text
City Coordinates
       ↓
Graph Construction
       ↓
2-Layer GCN
       ↓
Node Embeddings
       ↓
Pseudo Routes
       ↓
Edge Feature Generation
       ↓
Edge Importance Learning
       ↓
Learned Edge Scores
       ↓
Learned Cost Matrix
       ↓
Route Optimization
```

---

## Methodology

### 1. Graph Construction

Each city is represented as a graph node.

**Node features include:**

* Spatial coordinates
* Distance-related information

The graph is initially constructed using **K-nearest-neighbor (KNN) connectivity**.

For the experimental setup:

* Number of cities: **100**
* K: **2**
* Self-loops: Added before GCN normalization
* Edge weights: Based initially on Euclidean distance

The adjacency matrix is normalized before applying graph convolution.

---

### 2. Two-Layer GCN

A manually implemented two-layer GCN is used to learn graph-aware representations.

```text
Input Node Features
      3 features
          ↓
      GCN Layer 1
       3 → 32
          ↓
    Hidden Features
       100 × 32
          ↓
      GCN Layer 2
       32 → 16
          ↓
    Node Embeddings
       100 × 16
```

The propagation is:

```python
H1 = ReLU(A_hat @ X @ W1)

H2 = ReLU(A_hat @ H1 @ W2)
```

where:

* `X` = node feature matrix
* `A_hat` = normalized adjacency matrix
* `W1` = first GCN weight matrix `(3 × 32)`
* `W2` = second GCN weight matrix `(32 × 16)`
* `H2` = learned node embeddings `(100 × 16)`

---

## 3. Pseudo-Route Generation

Since labelled edge data is not directly available, pseudo-good routes are generated using:

1. **Nearest Neighbor**
2. **2-opt improvement**

The resulting routes provide supervision for the edge-importance learning stage.

For each pseudo-route:

```text
Route
 ↓
Identify consecutive city pairs
 ↓
Mark corresponding edges as 1
 ↓
Other candidate edges = 0
```

Thus, the route itself becomes the source of the **edge labels**.

---

## 4. Edge Feature Construction

For every directed edge `(i, j)`, an edge feature vector is created using:

```text
Edge Feature(i,j)
=
[ Node Embedding(i)
  Node Embedding(j)
  Distance(i,j) ]
```

Each node embedding contains **16 features**.

Therefore:

```text
16 + 16 + 1 = 33 edge features
```

So each edge is represented by a **33-dimensional feature vector**.

Example:

```text
        Source Node i
        H2[i] = 16
             │
             ├─────────────┐
             │             │
             ↓             ↓
        ┌──────────────────────┐
        │   Edge Feature(i,j)  │
        │                      │
        │ H2[i] + H2[j] + d(i,j)
        │                      │
        │       33 features    │
        └──────────────────────┘
             ↑
             │
        H2[j] = 16
        Destination Node j
```

---

## 5. Edge Importance Learning

A linear model is used to predict the importance of each edge.

```python
logits = features @ Wedge
preds = sigmoid(logits)
```

where:

```text
features : 33-dimensional edge representation
Wedge    : 33 × 1 trainable weight vector
preds    : edge importance probability
```

### Why sigmoid?

The target indicates whether an edge appears in a pseudo-route:

```text
1 → Edge used
0 → Edge not used
```

Therefore, the problem is treated as **binary classification**.

The sigmoid converts the model output into a value between **0 and 1**, which can be interpreted as the predicted likelihood/importance of an edge.

---

## 6. Training

The model uses the pseudo-route edge labels as supervision.

```text
Edge Features
     +
Edge Labels
     ↓
Linear Layer
     ↓
Logits
     ↓
Sigmoid
     ↓
Predicted Edge Importance
     ↓
Binary Cross-Entropy
     ↓
Gradient Calculation
     ↓
Update Wedge
```

The weight update follows:

```text
Wedge = Wedge + LR × gradWedge
```

The process is repeated over the available training samples.

---

## 7. Final Edge Importance Prediction

After training, the learned `Wedge` is applied to **all possible directed non-self edges**.

For 100 cities:

```text
100 × 99 = 9,900 directed edges
```

For each edge:

```text
[Source Embedding | Destination Embedding | Distance]
                         ↓
                    Wedge
                         ↓
                      Sigmoid
                         ↓
                Edge Importance Score
```

These scores form an **edge-importance matrix**.

The learned scores can then be combined with the original distance information to construct a **learned cost matrix**, which can be used by downstream route optimization algorithms.

---

## Experimental Setup

| Parameter              |                    Value |
| ---------------------- | -----------------------: |
| Number of cities       |                      100 |
| KNN neighbors          |                        2 |
| GCN layers             |                        2 |
| Input node features    |                        3 |
| Hidden dimension       |                       32 |
| Embedding dimension    |                       16 |
| Edge feature dimension |                       33 |
| Pseudo routes          |                       30 |
| Route generation       | Nearest Neighbor + 2-opt |
| Edge model             |         Linear + Sigmoid |
| Edge classification    |                   Binary |

---

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Graph Neural Networks**
* **Graph Convolutional Networks (GCN)**
* **Nearest Neighbor Search**
* **2-opt Optimization**
* **Binary Classification**

---

## Summary

This project demonstrates a **GNN-assisted route optimization framework** in which graph convolution is used to learn node representations, while a supervised edge model learns the importance of individual connections from pseudo-good routes. The resulting learned edge scores provide an additional information layer that can be incorporated into downstream route optimization.
