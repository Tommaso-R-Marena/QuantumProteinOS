import os
import json
import numpy as np

try:
    from qpos.quantum.iws_qaoa import IWSQAOASolver
except ImportError as e:
    raise NotImplementedError(f"Modular dependency unavailable: {e}")

def calculate_exact_ground_state(Q):
    """Exhaustive search for ground state."""
    n_qubits = Q.shape[0]
    best_energy = float('inf')
    
    for i in range(2**n_qubits):
        vec = np.array([int(x) for x in format(i, f'0{n_qubits}b')])
        energy = vec.T @ Q @ vec
        if energy < best_energy:
            best_energy = energy
    return best_energy

def greedy_solve(Q, N, n):
    """Single-shot greedy: for each residue, pick the rotamer with lowest local energy."""
    assignment = np.zeros(N * n, dtype=int)
    for i in range(N):
        best_rot = 0
        best_e = float('inf')
        for r in range(n):
            vec = np.zeros(N * n)
            vec[i * n + r] = 1
            e = vec @ Q @ vec
            if e < best_e:
                best_e = e
                best_rot = r
        assignment[i * n + best_rot] = 1
    return assignment @ Q @ assignment

def main():
    print("Running IWS-QAOA rotamer packing benchmark...")
    
    instances = []
    solver = IWSQAOASolver()
    
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    
    # N=4 instance
    N4, n = 4, 3
    Q4 = rng.standard_normal((N4*n, N4*n))
    Q4 = (Q4 + Q4.T) / 2  # Symmetrize — QUBO matrices must be symmetric
    
    gs4_energy = calculate_exact_ground_state(Q4)
    res4 = solver.solve(Q4, N4, n)
    iws_gap4 = 0.0 if gs4_energy == 0 else abs((res4.energy - gs4_energy) / gs4_energy) * 100
    
    greedy_energy4 = greedy_solve(Q4, N4, n)
    greedy_gap4 = 0.0 if gs4_energy == 0 else abs((greedy_energy4 - gs4_energy) / gs4_energy) * 100
    
    instances.append({'name': 'N=4', 'greedy_gap_pct': greedy_gap4, 'iws_qaoa_gap_pct': iws_gap4})
    
    # N=5 instance
    N5 = 5
    Q5 = rng.standard_normal((N5*n, N5*n))
    Q5 = (Q5 + Q5.T) / 2  # Symmetrize
    
    gs5_energy = calculate_exact_ground_state(Q5)
    res5 = solver.solve(Q5, N5, n)
    iws_gap5 = 0.0 if gs5_energy == 0 else abs((res5.energy - gs5_energy) / gs5_energy) * 100
    
    greedy_energy5 = greedy_solve(Q5, N5, n)
    greedy_gap5 = 0.0 if gs5_energy == 0 else abs((greedy_energy5 - gs5_energy) / gs5_energy) * 100

    instances.append({'name': 'N=5', 'greedy_gap_pct': greedy_gap5, 'iws_qaoa_gap_pct': iws_gap5})

    os.makedirs('benchmarks/rotamer_packing/results', exist_ok=True)
    with open('benchmarks/rotamer_packing/results/comparison_table.json', 'w') as f:
        json.dump({'instances': instances}, f)

if __name__ == '__main__':
    main()
