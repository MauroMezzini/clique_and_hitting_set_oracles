# Quantum Graph & Hypergraph Phase Encoding Oracles

A Qiskit-based framework implementing Quantum Phase Estimation (QPE) and quantum oracle circuits for graph and hypergraph problem instances, specifically designed for finding **$k$-Cliques** and **Hitting Sets**.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
  - [1. Data Representation](#1-data-representation)
  - [2. Phase Encoding Operators](#2-phase-encoding-operators)
  - [3. Oracle Circuits](#3-oracle-circuits)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This repository contains Python routines to build and simulate quantum oracle circuits that encode combinatorial structures on graphs and hypergraphs. By leveraging Quantum Phase Estimation (QPE) and arithmetic helper circuits (quantum incrementers/decrementers), the codebase generates quantum oracles capable of marking valid solutions (such as $k$-cliques or hitting sets) via phase kickback.

---

## Key Features

* **Hypergraph & Graph Data Structure:** Automatically classifies graph types based on edge cardinality and determines phase-register size requirements.
* **Quantum Phase Estimation:** Estimates topological properties into quantum phase registers using controlled phase and multi-controlled phase gates.
* **Arithmetic Circuits:** Includes custom little-endian incrementer (`+1`) and decrementer (`-1`) quantum subroutines using multi-controlled $X$ gates.
* **Problem Oracles:**
  * **Clique Oracle:** Marks vertex subsets forming $k$-cliques on ordinary graphs.
  * **Hitting Set Oracle:** Marks vertex subsets forming hitting sets of threshold size $k$ on hypergraphs.
* **State Inspection & Measurement:** Utilities to inspect non-zero state vector amplitudes, magnitudes, and phase angles in radians, as well as sample phase register outcomes.

---

## How It Works

### 1. Data Representation
The `hypergraph` class stores vertices and edges. If every edge contains exactly 2 vertices, it is classified as `"graph"`; otherwise, it is classified as `"hypergraph"`. It dynamically allocates $m = 1 + \lceil \log_2(M) \rceil$ counting qubits for the phase register, where $M$ is the number of edges.

### 2. Phase Estimation Operators
* **`graph_encoding_operator`**: Applies controlled-phase ($\text{CP}$) gates across endpoint qubits for standard graphs.
* **`hypergraph_encoding_operator`**: Applies single-qubit phase gates based on vertex degree and multi-controlled phase ($\text{MCP}$) gates across hyperedge subsets.
* **`phase_encoding_operator`**: Assembles the QPE circuit by driving controlled powers of the encoding operator followed by an inverse Quantum Fourier Transform ($\text{QFT}^\dagger$).

### 3. Oracle Circuits
Both oracles use a phase-kickback flag qubit prepared in the $\ket{-} = \frac{\ket{0} - \ket{1}}{\sqrt{2}}$ state:
* **`clique_oracle_circuit`**: Initializes a weight counter to $-k$ (using Two's Complement), increments the counter for each selected vertex in $\ket{1}$, and triggers a phase flip if the phase estimation register and weight conditions evaluate to zero.
* **`hitting_set_oracle_circuit`**: Loads threshold $k$ into a weight register, decrements it for each selected vertex, and conditionally flips the flag qubit when all control conditions are met.

---

## Dependencies

The code is written in **Python 3** and requires the following external packages:

* **[qiskit](https://qiskit.org/)** ($\ge 1.0.0$): For quantum circuit creation, gate primitives (`QFTGate`, `XGate`), and quantum state vector simulation (`qiskit.quantum_info.Statevector`).
* **[numpy](https://numpy.org/)**: For numerical routines, logarithmic register sizing, phase angle calculations, and array manipulations.
* **[matplotlib](https://matplotlib.org/)**: Optional dependency for rendering circuit diagrams (`qc.draw()`).

*(Standard library dependencies include `math`, `itertools`, and `random`.)*

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/quantum-graph-oracles.git
   cd quantum-graph-oracles
   ```

2. **Install requirements:**
   ```bash
   pip install qiskit numpy matplotlib
   ```

---

## Usage

Run the main module directly to execute demo instances for both a 5-vertex graph ($k=3$ clique oracle) and a 5-vertex hypergraph ($k=3$ hitting set oracle):

```bash
python GEO_and_HEO.py
```

### Example Code Snippet

```python
from GEO_and_HEO import hypergraph, clique_oracle_circuit, print_state_vector

# Define vertices and edges for a simple graph
vertices = [0, 1, 2, 3, 4]
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (2, 4)]

# Initialize graph object
G = hypergraph(vertices, edges)

# Construct Clique Oracle for k = 3
oracle_qc = clique_oracle_circuit(G, k=3)

# Print state vector
print_state_vector(oracle_qc, "Clique Oracle State Vector")
```
