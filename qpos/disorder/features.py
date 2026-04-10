import numpy as np

FLEXIBILITY = {'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462, 'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386}
HYDROPHOBICITY = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'R':1,'K':1,'D':-1,'E':-1}
AROMATICITY = {'F':1,'W':1,'Y':1}
SS_HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
SS_SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
PI = {'A':6.00,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.30,'S':5.68,'T':5.60,'W':5.89,'Y':5.66,'V':5.96}
MW = {'A':89,'R':174,'N':132,'D':133,'C':121,'Q':146,'E':147,'G':75,'H':155,'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,'T':119,'W':204,'Y':181,'V':117}

def extract_features(sequence: str) -> np.ndarray:
    """
    Extracts multi-window sliding average physicochemical features resulting in a 406-dimensional vector per residue.
    """
    N = len(sequence)
    scales = [7, 11, 15, 21, 31, 51, 101]
    
    raw_hydro = np.array([HYDROPHOBICITY.get(c, 0.0) for c in sequence])
    raw_charge = np.array([CHARGE.get(c, 0.0) for c in sequence])
    raw_arom = np.array([AROMATICITY.get(c, 0.0) for c in sequence])
    raw_flex = np.array([FLEXIBILITY.get(c, 0.4) for c in sequence])
    raw_helix = np.array([SS_HELIX.get(c, 1.0) for c in sequence])
    raw_sheet = np.array([SS_SHEET.get(c, 1.0) for c in sequence])
    raw_pi = np.array([PI.get(c, 6.0) for c in sequence])
    raw_mw = np.array([MW.get(c, 120.0) for c in sequence])
    
    props = [raw_hydro, raw_charge, raw_arom, raw_flex, raw_helix, raw_sheet, raw_pi, raw_mw]
    features = np.zeros((N, 406))
    
    f_idx = 0
    for p in props:
        for w in scales:
            if f_idx >= 406:
                break
            half = w // 2
            padded = np.pad(p, (half, half), mode='edge')
            cumsum = np.cumsum(np.insert(padded, 0, 0))
            moving_avg = (cumsum[w:] - cumsum[:-w]) / w
            features[:, f_idx] = moving_avg
            f_idx += 1
            
    return features
