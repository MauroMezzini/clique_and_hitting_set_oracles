#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 08:45:59 2026

@author: mauro
"""

import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
from itertools import combinations
import matplotlib.pyplot as plt
from qiskit.circuit.library import XGate
import random

def print_state_vector(qc, text, only_all_non_zero_amplitudes=True):
    """Print the state vector produced by a quantum circuit.

    Args:
        qc: The circuit to simulate. It must be compatible with
            ``Statevector.from_instruction`` (for example, a unitary circuit
            without measurements).
        text: A heading printed before the state-vector entries.
        only_all_non_zero_amplitudes: If True, omit basis states whose
            amplitudes are numerically close to zero. If False, print every
            basis state.

    Each output entry includes the computational-basis state, its complex
    amplitude, amplitude magnitude, and phase in radians. The displayed
    bitstring follows the integer indexing used by Qiskit's state vector.
    """
    print(f"\n=== {text} ===")
    # Simulate the circuit and obtain its state-vector amplitudes.
    state = Statevector.from_instruction(qc)
    n_qubits = qc.num_qubits

    # Visit each computational-basis state in the vector.
    for idx, amp in enumerate(state.data):
        # Convert the vector index to a fixed-width binary basis label.
        bitstring = format(idx, f"0{n_qubits}b")
        magnitude = np.abs(amp)

        # Optionally hide amplitudes that are effectively zero (up to
        # floating-point precision).
        if not only_all_non_zero_amplitudes or not np.isclose(magnitude, 0):
            # np.angle returns the complex amplitude's phase in radians.
            phase_angle_rad = np.angle(amp)
            print(f"|{bitstring}> : Amplitude = {amp:+.4f} | Mag = {magnitude:.4f} | Phase = {phase_angle_rad:+.4f} rad")



class hypergraph:
    """Represent a graph or hypergraph and its quantum-encoding dimensions.

    Vertices and edges are stored as provided. Each edge should be an
    iterable of vertex indices; if all the edges contains exactly  two vertices 
    then it is classified as a graph, otherwise  it is classified as a hypergraph.
    """
    def __init__(self, vertices, edges):
        """Initialize the structure and calculate its register sizes.

        Args:
            vertices: Vertex indices used by the edge definitions.
            edges: Iterable of edges, where each edge contains vertex indices.

        Attributes:
            vertices: The provided collection of vertices.
            edges: The provided collection of edges.
            N: Number of vertices.
            M: Number of edges.
            m: Number of counting qubits used by the phase register.
            type: ``"graph"`` if every edge has two vertices, otherwise
                ``"hypergraph"``.
        """
        
        self.vertices = vertices
        self.edges = edges

        
        self.N = len(vertices)
        self.M = len(edges)

        # Allocate enough counting qubits to represent the edge count,
        # retaining one extra qubit as in this project's QPE convention.
        self.m = 1 + int(np.ceil(np.log2(len(edges))))  # Number of counting qubits

        # Ordinary graph edges all have size two; otherwise classify as a
        # hypergraph.
        self.type = "graph" if all(len(edge) == 2 for edge in edges) else "hypergraph"
        
def graph_encoding_operator(G, theta):
    """Build the graph phase-encoding circuit for phase angle ``theta``.

    Each edge contributes a controlled-phase gate between its two vertex
    qubits. The circuit uses the graph's vertex indices directly as qubit
    indices and is returned without modifying ``G``.

    Args:
        G: Graph-like object with ``vertices`` and two-vertex ``edges``.
        theta: Phase angle, in radians, applied for each edge.

    Returns:
        A quantum circuit with one qubit per vertex.
    """
    # Create a target register with one qubit for each graph vertex.
    qc = QuantumCircuit(len(G.vertices), name="GraphCircuit")
    
    # Encode each edge with a controlled phase between its endpoint qubits.
    for j, h in G.edges:
        qc.cp(theta, j, h)

    return qc

def hypergraph_encoding_operator(H, theta, apply_global_phase=False):
    """Build the phase-encoding circuit for a hypergraph.

    The circuit first applies a phase to each vertex qubit proportional to
    that vertex's degree, then adds multi-controlled phase gates for each
    hyperedge. Optionally, it includes a global phase proportional to the
    number of hyperedges.
    """
    qc = QuantumCircuit(len(H.vertices), name="HyperGraphCircuit")
    num_edges = len(H.edges)

    # Apply the degree-dependent single-qubit phase to every vertex.
    for v in H.vertices:
        adj = sum(1 for e in H.edges if v in e)
        qc.p(theta * adj, v)

    # Add the controlled phase contributions for each non-singleton edge.
    for e in H.edges:
        if len(e) > 1:
            # Build the controls incrementally in sorted vertex order.
            control_list= []
            for u in sorted(e):
                for v in sorted(e):
                    if v > u:
                        # Apply -theta when the target and controls match the
                        # requested control state for this vertex pair.
                        qc.mcp(-theta, control_list+[u], v, ctrl_state="1"+str("0"*len(control_list)))
                control_list.append(u)

    if apply_global_phase:
        # Include one uniform phase offset per hyperedge when requested.
        qc.global_phase = -theta * num_edges
    return qc

def phase_encoding_operator(G, is_superposition=True):
    """Construct a quantum phase-estimation circuit for graph encoding.

    The first ``G.m`` qubits form the phase register, and the following
    ``G.N`` qubits encode vertex selections. Controlled powers of the graph
    or hypergraph encoding operator are applied before the inverse QFT.

    Args:
        G: Graph or hypergraph providing vertex/edge data and register sizes.
        is_superposition: If True, apply Hadamard gates to both registers;
            otherwise, put only the phase register into superposition.

    Returns:
        A quantum circuit containing the phase-estimation operation.
    """
    # --- QPE Circuit Construction ---
    N = G.N
    m = G.m  # Number of qubits in the phase/counting register.
    theta =  2 * np.pi  /(2**m)

    # Place the phase register first, followed by the N vertex-selection
    # qubits used as the target register for the encoding operator.
    qpe_qc = QuantumCircuit(m + N)

    # Prepare the phase register for phase estimation. Optionally prepare
    # the vertex register in a uniform superposition as well.
    if is_superposition:
        qpe_qc.h(range(m+N))
    else:
        qpe_qc.h(range(m))

    # Apply controlled powers B^(2^j), with phase-register qubit j as
    # the control and the vertex register as the target.
    for j in range(m):
        power = 2**j
        # Scale the base phase angle to construct the required power of B.
        if G.type == "graph":
            B = graph_encoding_operator(G, power * theta)
        else:
            B = hypergraph_encoding_operator(G, power * theta,apply_global_phase=True)

        # Convert B to a gate and promote it to a single-control operation.
        B_gate = B.to_gate(label=f"B^{power}")
        controlled_B = B_gate.control(1)

        # The target register begins immediately after the phase register.
        target_qubits = list(range(m, m + N))
        qpe_qc.append(controlled_B, [j] + target_qubits)

    # Transform the phase register with the inverse QFT to obtain the
    # encoded phase in the computational basis.
    iqft = QFTGate(m).inverse()
    iqft.label = "QFT†"
    qpe_qc.append(iqft, range(m))
    return qpe_qc

def add_one_circuit(weight_register_size): 
    """Create a circuit that increments a little-endian register by one.

    The register qubits are ordered from least significant to most
    significant. Higher-order bits are toggled when all lower-order bits
    are set, implementing addition modulo ``2**weight_register_size``.

    Args:
        weight_register_size: Number of qubits in the weight register.

    Returns:
        A quantum circuit implementing the increment operation.
    """
    # Create the incrementer and list register qubits from least to most
    # significant, matching the bit order used by the oracle circuits.
    qc_add_one = QuantumCircuit( weight_register_size, name="+1")
    reg_weight_indices=list(range(weight_register_size))

    # Process from the most-significant bit down. Each bit flips only when
    # every less-significant bit is 1, which propagates the carry.
    for bit_idx, b_qubit in reversed(list(enumerate(reg_weight_indices))):
        controls =  reg_weight_indices[:bit_idx]
        if len(controls) == 1:
            # A single lower bit can control the flip directly.
            qc_add_one.cx(controls[0], b_qubit)
        else:
            # With zero or multiple lower bits, use a multi-controlled X;
            # no controls means the least-significant bit flips unconditionally.
            qc_add_one.mcx(controls, b_qubit)

    # Return the reusable increment circuit.
    return qc_add_one

def subtract_one_circuit(weight_register_size):
    """Create a circuit that decrements a little-endian register by one.

    The register qubits are ordered from least significant to most
    significant. Higher-order bits toggle when all lower-order bits are zero,
    implementing subtraction modulo ``2**weight_register_size``.

    Args:
        weight_register_size: Number of qubits in the weight register.

    Returns:
        A quantum circuit implementing the decrement operation.
    """
    # Create the decrementer and list the register from least to most
    # significant, consistent with the oracle's weight-register layout.
    qc_subtract_one = QuantumCircuit(weight_register_size, name="-1")
    reg_weight_indices=list(range(weight_register_size))

    # Process from least-significant to most-significant bit. A bit toggles
    # when all lower bits are 0, propagating the borrow during subtraction.
    for bit_idx, b_qubit in list(enumerate(reg_weight_indices)):
        controls =  reg_weight_indices[:bit_idx]
        if len(controls) == 1:
            # A single lower bit can control the toggle directly.
            qc_subtract_one.cx(controls[0], b_qubit)
        else:
            # Use a multi-controlled X for multiple controls; with no
            # controls, the least-significant bit toggles unconditionally.
            qc_subtract_one.mcx(controls, b_qubit)

    # Return the reusable decrement circuit.
    return qc_subtract_one

def clique_oracle_circuit(G, k):
    """Build an oracle circuit that marks graph selections forming a k-clique.

    The circuit runs phase estimation, accumulates the selected-vertex count
    offset by ``-k`` in a weight register, then uses the phase and weight
    registers to conditionally flip the flag qubit.

    Args:
        G: A graph instance with vertex, edge, and phase-register metadata.
        k: Clique size threshold encoded by the oracle.

    Returns:
        The completed oracle circuit, or ``None`` if ``G`` is not a graph.
    """
    # This oracle uses graph-specific encoding and is not defined here for
    # general hypergraphs.
    if G.type != "graph":
        print("clique_oracle_circuit is only applicable to graphs, not hypergraphs.")
        return None

    # Reserve registers for the phase-estimation output, vertex selections,
    # an n-qubit weight counter, and one flag qubit.
    n = 1+int(np.ceil(np.log2(G.N)))
    oracle_qc = QuantumCircuit(G.N+G.m+n+1, name="CliqueOracleCircuit")

    # Apply phase estimation to the graph encoding using the leading phase
    # register and the following vertex-selection register.
    # Superposition is applied to the vertex selection just for testing porpouse.
    qpe_qc = phase_encoding_operator(G, is_superposition=True)
    oracle_qc.compose(qpe_qc, inplace=True)

    # Initialize the weight register to -k in two's-complement form. Adding
    # one for every selected vertex makes the final value depend on |S| - k.
    weight_register_indices = list(range(G.m + G.N, G.m + G.N + n))
    minus_k_2_complement= 2**n - k
    for bit_idx, b_qubit in enumerate(weight_register_indices):
        if (minus_k_2_complement >> bit_idx) & 1:
            oracle_qc.x(b_qubit)

    # Add one to the counter for each vertex qubit in state |1|.
    add_one_gate = add_one_circuit(n).to_gate(label="+1")
    targets_qubits = list(range(G.m + G.N, G.m + G.N + n))
    for g_qubit in range(G.m, G.m + G.N):
        controlled_increment = add_one_gate.control(1)
        oracle_qc.append(controlled_increment, [g_qubit] + targets_qubits)

    # Prepare the flag in |-> so a conditional X applies a phase flip
    # through phase kickback.
    flag_qubit = G.m + G.N + n
    oracle_qc.x(flag_qubit)  # Set the flag qubit to |1>
    oracle_qc.h(flag_qubit)

    # Use the phase-estimation register and the most-significant weight bit
    # as controls for the marking operation.
    control_qubits = list(range(G.m)) + [G.m + G.N + n - 1]
    target_qubit = flag_qubit

    # Trigger only when every control is |0>, corresponding to a zero
    # phase-estimation result and a zero most-significant weight bit.
    ctrl_state_str = "0" * len(control_qubits)

    # Construct a multi-controlled X with the required all-zero control state.
    mcx_gate = XGate().control(
        num_ctrl_qubits=len(control_qubits), 
        ctrl_state=ctrl_state_str,
        annotated=False # Keeps it as a concrete gate structure if needed
    )

    # Append controls first and the flag as the target, then return the oracle.
    oracle_qc.append(mcx_gate, control_qubits + [target_qubit])
    return oracle_qc
    

def hitting_set_oracle_circuit(G, k):
    """Build an oracle circuit for a hitting-set size threshold.

    The circuit applies phase estimation to the hypergraph encoding, loads
    ``k`` into a weight register, and decrements that register for each
    selected vertex. A multi-controlled X then performs the flag operation.

    Args:
        G: Hypergraph providing the vertices, edges, and register metadata.
        k: Hitting-set size threshold to encode.

    Returns:
        The oracle circuit
    """

    # Reserve the phase register, vertex register, weight counter, and flag.
    n = 1+int(np.ceil(np.log2(G.N)))
    oracle_qc = QuantumCircuit(G.N+G.m+n+1, name="HittingSetOracleCircuit")

    # Run phase estimation on the hypergraph encoding.
    # Superposition is applied to the vertex selection just for testing porpouse.
    qpe_qc = phase_encoding_operator(G, is_superposition=True)
    oracle_qc.compose(qpe_qc, inplace=True)

    # Initialize the weight register to k in binary.
    weight_register_indices = list(range(G.m + G.N, G.m + G.N + n-1))
    for bit_idx, b_qubit in enumerate(weight_register_indices):
        if (k >> bit_idx) & 1:
            oracle_qc.x(b_qubit)

    # Decrement the weight counter once for each selected vertex qubit.
    subtract_one_gate = subtract_one_circuit(n).to_gate(label="-1")
    targets_qubits = list(range(G.m + G.N, G.m + G.N + n))
    for g_qubit in range(G.m, G.m + G.N):
        controlled_increment = subtract_one_gate.control(1)
        oracle_qc.append(controlled_increment, [g_qubit] + targets_qubits)

    # Prepare the flag in $\ket{-}$ so a conditional X applies a phase flip by
    # phase kickback.
    flag_qubit = G.m + G.N + n
    oracle_qc.x(flag_qubit)  # Set the flag qubit to |1>
    oracle_qc.h(flag_qubit)
    
    # Use all phase-register qubits and the most-significant weight bit as
    # controls for the marking operation.
    control_qubits = list(range(G.m)) + [G.m + G.N + n - 1]
    target_qubit = flag_qubit

    # Activate the operation only when every control qubit is |0>.
    ctrl_state_str = "0" * len(control_qubits)

    # Construct the multi-controlled X for the all-zero control pattern.
    mcx_gate = XGate().control(
        num_ctrl_qubits=len(control_qubits), 
        ctrl_state=ctrl_state_str,
        annotated=False # Keeps it as a concrete gate structure if needed
    )

    # Append the gate with controls first and the flag as its target.
    oracle_qc.append(mcx_gate, control_qubits + [target_qubit])
    return oracle_qc


 
def measure_phase_amplitude(qpe_qc, m, N, shots=1024):
    """Sample a QPE circuit and print counts for its phase register.

    The circuit is simulated as a state vector, then sampled the requested
    number of times. Since Qiskit displays bitstrings from the highest-index
    qubit to the lowest, the rightmost ``m`` bits correspond to qubits
    ``0`` through ``m - 1``.

    Args:
        qpe_qc: Quantum phase-estimation circuit to simulate.
        m: Number of qubits in the phase register.
        N: Number of target-register qubits (retained for the function's
            interface; the phase register is selected using ``m``).
        shots: Number of samples to draw from the circuit's output state.
    """
    # Remove any terminal measurements so Statevector can simulate the
    # circuit, then sample its final state.
    qpe_without_measurements = qpe_qc.remove_final_measurements(inplace=False)
    counts = Statevector.from_instruction(
        qpe_without_measurements
    ).sample_counts(shots=shots)

    print(f"\n--- Measurement Results ({shots} shots) ---")
    first_m_counts = {}
    for bitstring, count in counts.items():
        # Extract the phase-register bits (qubits 0 through m - 1) from
        # Qiskit's right-to-left bitstring representation.
        first_m = bitstring[-m:]
        # Combine counts that share the same phase-register measurement,
        # regardless of the target-register state.
        first_m_counts[first_m] = (
            first_m_counts.get(first_m, 0) + count
        )

    # Display phase-register outcomes from most to least frequently sampled.
    print("Measurements of the phase register (displayed as q(m-1),..., q1, q0):")
    for bitstring, count in sorted(
        first_m_counts.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{bitstring}: {count} counts ({(count/shots)*100:.2f}%) ")




if __name__ == "__main__":
    # Demonstrate phase estimation and the clique oracle on the graph 
    # reported in the paper
    vertices = [0,1,2,3, 4]
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (2, 4)]  # 0-indexed vertices
    G = hypergraph(vertices, edges)
    qpe_qc = phase_encoding_operator(G, is_superposition=True)
    print_state_vector(qpe_qc, "Phase State Vector")
    oracle_qc = clique_oracle_circuit(G, k=3)
    print("--- Clique Oracle Circuit ---")
    print(oracle_qc.draw(output='text'))
    print_state_vector(oracle_qc, "Final State Vector clique oracle circuit")
    

    # Demonstrate phase estimation and the hitting-set oracle on the hypergraph
    # reported in the paper.
    vertices=[0,1,2,3,4]
    edges=[{0,3}, {0,4}, {1,3, 4}, {0,1, 2}, {2,3, 4}]
    G = hypergraph(vertices, edges)
    qpe_qc = phase_encoding_operator(G, is_superposition=True)
    print_state_vector(qpe_qc, "Phase State Vector")
    oracle_qc = hitting_set_oracle_circuit(G, k=3)
    print("--- Hitting Set Oracle Circuit ---")
    print(oracle_qc.draw(output='text'))
    print_state_vector(oracle_qc, "Final State Vector hitting set oracle circuit")

    measure_phase_amplitude(qpe_qc, G.m, G.N, shots=1024)
  
