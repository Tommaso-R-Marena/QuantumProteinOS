import numpy as np

def _kabsch_alignment(P, Q):
    """
    Computes the optimal rotation matrix using Kabsch algorithm.
    P and Q must be (N, 3) numpy arrays centered at origin.
    """
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    return U

def imfdrMSD(pred_coords: np.ndarray, ref_coords: np.ndarray, inhibitory_residues: list) -> float:
    """
    imfdRMSD computed only over the inhibitory module residues 
    after global Kabsch alignment on all Cα atoms.
    """
    N = pred_coords.shape[0]
    p_center = np.mean(pred_coords, axis=0)
    r_center = np.mean(ref_coords, axis=0)
    
    P_centered = pred_coords - p_center
    R_centered = ref_coords - r_center
    
    U = _kabsch_alignment(P_centered, R_centered)
    
    P_aligned = np.dot(P_centered, U)
    
    if len(inhibitory_residues) == 0:
        return 0.0
        
    p_inhib = P_aligned[inhibitory_residues]
    r_inhib = R_centered[inhibitory_residues]
    
    diff = p_inhib - r_inhib
    msd = np.sum(diff**2) / len(inhibitory_residues)
    return float(np.sqrt(msd))

MJ_MATRIX = {
    ('A','A'):-1.74,('A','C'):-2.00,('A','D'):-1.42,('A','E'):-1.61,
    ('A','F'):-2.03,('A','G'):-1.59,('A','H'):-1.67,('A','I'):-2.29,
    ('A','K'):-1.31,('A','L'):-2.26,('A','M'):-2.15,('A','N'):-1.44,
    ('A','P'):-1.44,('A','Q'):-1.53,('A','R'):-1.45,('A','S'):-1.55,
    ('A','T'):-1.65,('A','V'):-2.10,('A','W'):-2.01,('A','Y'):-1.87,
    ('C','C'):-3.83,('C','D'):-1.68,('C','E'):-1.89,('C','F'):-2.86,
    ('C','G'):-1.73,('C','H'):-2.26,('C','I'):-2.86,('C','K'):-1.50,
    ('C','L'):-2.86,('C','M'):-2.78,('C','N'):-1.76,('C','P'):-1.71,
    ('C','Q'):-1.90,('C','R'):-1.71,('C','S'):-1.97,('C','T'):-1.90,
    ('C','V'):-2.57,('C','W'):-2.89,('C','Y'):-2.71,('D','D'):-1.21,
    ('D','E'):-1.50,('D','F'):-1.94,('D','G'):-1.26,('D','H'):-1.75,
    ('D','I'):-1.84,('D','K'):-1.38,('D','L'):-1.83,('D','M'):-1.75,
    ('D','N'):-1.43,('D','P'):-1.26,('D','Q'):-1.55,('D','R'):-1.59,
    ('D','S'):-1.31,('D','T'):-1.37,('D','V'):-1.70,('D','W'):-1.98,
    ('D','Y'):-1.90,('E','E'):-1.61,('E','F'):-2.05,('E','G'):-1.34,
    ('E','H'):-1.84,('E','I'):-1.98,('E','K'):-1.49,('E','L'):-2.01,
    ('E','M'):-1.91,('E','N'):-1.51,('E','P'):-1.37,('E','Q'):-1.67,
    ('E','R'):-1.64,('E','S'):-1.39,('E','T'):-1.50,('E','V'):-1.84,
    ('E','W'):-2.09,('E','Y'):-2.00,('F','F'):-3.17,('F','G'):-1.87,
    ('F','H'):-2.53,('F','I'):-3.03,('F','K'):-1.71,('F','L'):-3.04,
    ('F','M'):-2.95,('F','N'):-1.91,('F','P'):-1.82,('F','Q'):-2.02,
    ('F','R'):-1.89,('F','S'):-1.94,('F','T'):-2.01,('F','V'):-2.72,
    ('F','W'):-3.26,('F','Y'):-3.05,('G','G'):-1.40,('G','H'):-1.57,
    ('G','I'):-1.81,('G','K'):-1.17,('G','L'):-1.84,('G','M'):-1.74,
    ('G','N'):-1.31,('G','P'):-1.28,('G','Q'):-1.39,('G','R'):-1.31,
    ('G','S'):-1.37,('G','T'):-1.44,('G','V'):-1.67,('G','W'):-1.87,
    ('G','Y'):-1.78,('H','H'):-2.42,('H','I'):-2.26,('H','K'):-1.59,
    ('H','L'):-2.33,('H','M'):-2.21,('H','N'):-1.82,('H','P'):-1.67,
    ('H','Q'):-1.94,('H','R'):-1.80,('H','S'):-1.72,('H','T'):-1.80,
    ('H','V'):-2.11,('H','W'):-2.57,('H','Y'):-2.42,('I','I'):-3.31,
    ('I','K'):-1.63,('I','L'):-3.32,('I','M'):-3.05,('I','N'):-1.87,
    ('I','P'):-1.74,('I','Q'):-1.98,('I','R'):-1.83,('I','S'):-1.90,
    ('I','T'):-2.02,('I','V'):-2.99,('I','W'):-3.05,('I','Y'):-2.81,
    ('K','K'):-1.29,('K','L'):-1.77,('K','M'):-1.69,('K','N'):-1.41,
    ('K','P'):-1.25,('K','Q'):-1.53,('K','R'):-1.51,('K','S'):-1.32,
    ('K','T'):-1.42,('K','V'):-1.63,('K','W'):-1.85,('K','Y'):-1.75,
    ('L','L'):-3.44,('L','M'):-3.08,('L','N'):-1.90,('L','P'):-1.80,
    ('L','Q'):-2.03,('L','R'):-1.88,('L','S'):-1.95,('L','T'):-2.05,
    ('L','V'):-3.07,('L','W'):-3.12,('L','Y'):-2.90,('M','M'):-2.95,
    ('M','N'):-1.82,('M','P'):-1.74,('M','Q'):-1.96,('M','R'):-1.80,
    ('M','S'):-1.89,('M','T'):-1.98,('M','V'):-2.78,('M','W'):-3.04,
    ('M','Y'):-2.76,('N','N'):-1.50,('N','P'):-1.30,('N','Q'):-1.58,
    ('N','R'):-1.51,('N','S'):-1.43,('N','T'):-1.51,('N','V'):-1.73,
    ('N','W'):-1.95,('N','Y'):-1.84,('P','P'):-1.37,('P','Q'):-1.44,
    ('P','R'):-1.38,('P','S'):-1.35,('P','T'):-1.43,('P','V'):-1.64,
    ('P','W'):-1.83,('P','Y'):-1.74,('Q','Q'):-1.67,('Q','R'):-1.62,
    ('Q','S'):-1.49,('Q','T'):-1.58,('Q','V'):-1.87,('Q','W'):-2.08,
    ('Q','Y'):-1.97,('R','R'):-1.59,('R','S'):-1.43,('R','T'):-1.51,
    ('R','V'):-1.71,('R','W'):-1.93,('R','Y'):-1.84,('S','S'):-1.44,
    ('S','T'):-1.54,('S','V'):-1.75,('S','W'):-1.93,('S','Y'):-1.82,
    ('T','T'):-1.61,('T','V'):-1.90,('T','W'):-2.01,('T','Y'):-1.90,
    ('V','V'):-2.83,('V','W'):-2.92,('V','Y'):-2.70,('W','W'):-3.67,
    ('W','Y'):-3.23,('Y','Y'):-2.97
}

class QICESSScorer:
    """
    QICESS Scorer for autoinhibited protein conformational ensembles.
    Composite weights:
      - quantum: 0.30
      - retrieval: 0.25
      - plddt: 0.25
      - pae: 0.15
      - rama: 0.05
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.weights = self.config.get('weights', {
            'quantum': 0.30, 'retrieval': 0.25, 'plddt': 0.25, 'pae': 0.15, 'rama': 0.05
        })
        self.contact_cutoff = self.config.get('contact_cutoff', 8.0)
        
    def ising_contact_energy(self, coords, sequence, cutoff=8.0):
        """Real Cβ-Cβ contact energy using MJ potentials."""
        n = len(sequence)
        energy = 0.0
        for i in range(n):
            for j in range(i+2, n):  # skip i+1 (sequential)
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff:
                    seq_i = sequence[i].upper()
                    seq_j = sequence[j].upper()
                    pair = tuple(sorted([seq_i, seq_j]))
                    energy += MJ_MATRIX.get(pair, -1.0)
        return energy
        
    def score_ensemble(self, structures: list, reference_active: str=None, reference_inactive: str=None) -> list:
        # Mock scores for integration pipeline
        return [{'structure': s, 'score': np.random.rand() * 100} for s in structures]
        
    def rank_ensemble(self, scores: list) -> list:
        return sorted(scores, key=lambda x: x['score'], reverse=True)
