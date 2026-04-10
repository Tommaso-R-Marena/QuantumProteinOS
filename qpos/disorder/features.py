import numpy as np

# Group 1: Core Physicochemical Scales (Already correct)
FLEXIBILITY = {'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462, 'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386}
HYDROPHOBICITY = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'R':1,'K':1,'D':-1,'E':-1}
AROMATICITY = {'F':1,'W':1,'Y':1}
SS_HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
SS_SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
PI = {'A':6.00,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.30,'S':5.68,'T':5.60,'W':5.89,'Y':5.66,'V':5.96}
MW = {'A':89,'R':174,'N':132,'D':133,'C':121,'Q':146,'E':147,'G':75,'H':155,'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,'T':119,'W':204,'Y':181,'V':117}

# Group 2: Disorder-Specific Scales (Paper implementation)
UVERSKY_DISORDER = {'A':0.0,'R':1.0,'N':0.5,'D':0.5,'C':0.0,'Q':0.5,'E':0.5,'G':0.5,'H':0.3,'I':0.0,'L':0.0,'K':1.0,'M':0.0,'F':0.0,'P':0.8,'S':0.5,'T':0.3,'W':0.0,'Y':0.2,'V':0.0}
SOLUBILITY = {'A':1.0,'R':-1.0,'N':0.5,'D':0.5,'C':-0.5,'Q':0.5,'E':0.5,'G':0.8,'H':-0.3,'I':-1.0,'L':-1.0,'K':0.8,'M':-0.5,'F':-1.0,'P':0.7,'S':0.8,'T':0.5,'W':-1.0,'Y':-0.5,'V':-0.8}
WIMLEY_WHITE = {'A':-0.17,'R':-0.81,'N':-0.42,'D':-1.23,'C':0.24,'Q':-0.58,'E':-2.02,'G':-0.01,'H':-0.96,'I':0.31,'L':0.56,'K':-0.99,'M':0.23,'F':1.13,'P':-0.45,'S':-0.13,'T':-0.11,'W':1.85,'Y':0.94,'V':-0.07}
CONTACT_NUMBER = {'A':6.1,'R':4.5,'N':5.1,'D':5.0,'C':5.5,'Q':5.4,'E':5.2,'G':4.8,'H':5.7,'I':7.0,'L':7.3,'K':4.4,'M':6.5,'F':7.8,'P':5.2,'S':5.2,'T':5.8,'W':8.0,'Y':7.3,'V':6.8}
TURN_PROPENSITY = {'A':0.77,'R':0.79,'N':1.28,'D':1.41,'C':0.81,'Q':0.97,'E':0.99,'G':1.64,'H':0.68,'I':0.51,'L':0.58,'K':0.96,'M':0.73,'F':0.59,'P':1.91,'S':1.32,'T':1.04,'W':0.76,'Y':1.05,'V':0.47}
COIL_PROPENSITY = {'A':0.06,'R':-0.23,'N':0.21,'D':0.19,'C':-0.06,'Q':-0.09,'E':-0.08,'G':0.49,'H':-0.08,'I':-0.34,'L':-0.26,'K':-0.05,'M':-0.26,'F':-0.25,'P':0.37,'S':0.20,'T':0.05,'W':-0.26,'Y':-0.13,'V':-0.27}
BFACTOR = {'A':0.984,'R':1.008,'N':1.048,'D':1.068,'C':0.906,'Q':1.037,'E':1.094,'G':1.050,'H':0.950,'I':0.931,'L':0.935,'K':1.102,'M':0.952,'F':0.915,'P':1.020,'S':1.023,'T':0.996,'W':0.909,'Y':0.997,'V':0.931}
VDW_VOLUME = {'A':1.0,'R':6.13,'N':2.95,'D':2.78,'C':2.43,'Q':3.95,'E':3.78,'G':0.0,'H':4.66,'I':4.0,'L':4.0,'K':4.77,'M':4.43,'F':5.89,'P':2.72,'S':1.60,'T':2.60,'W':8.08,'Y':6.47,'V':3.0}
NCPR = {'A':0,'R':1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,'G':0,'H':0.1,'I':0,'L':0,'K':1,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
STICKINESS = {'A':0.060,'R':-0.710,'N':-0.360,'D':-0.720,'C':0.290,'Q':-0.430,'E':-0.620,'G':0.160,'H':-0.130,'I':0.730,'L':0.550,'K':-0.570,'M':0.320,'F':0.760,'P':-0.150,'S':-0.200,'T':-0.080,'W':0.950,'Y':0.490,'V':0.600}

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

class DisorderFeatureExtractor:
    def __init__(self, use_esm=True):
        self.use_esm = use_esm
        self.pca = None  # Fitted during DisorderNetV6.fit()
        
        if use_esm:
            try:
                from qpos.disorder.esm_embeddings import ESMCPUFeatureExtractor
                self.esm = ESMCPUFeatureExtractor()
                self.esm_available = True
            except ImportError:
                self.esm_available = False
        else:
            self.esm_available = False
    
    def get_esm_features(self, sequence, n_components=20):
        """Returns (n_residues, n_components) ESM embedding reduced by PCA."""
        if not self.esm_available or self.pca is None:
            return np.zeros((len(sequence), n_components))
        try:
            embeddings = self.esm.get_embeddings(sequence)  # (n_residues, 320)
            # Ensure embeddings are numpy for PCA
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()
            return self.pca.transform(embeddings)  # (n_residues, 20)
        except Exception:
            return np.zeros((len(sequence), n_components))

    def extract_features(self, sequence: str) -> np.ndarray:
        """
        Extracts multi-window sliding average features (406 total).
        58 properties * 7 scales = 406.
        """
        N = len(sequence)
        scales = [7, 11, 15, 21, 31, 51, 101]
        
        # Collect all 58 base properties
        props = []
        
        # Group 1: Physicochemical (8)
        for scale_dict in [HYDROPHOBICITY, CHARGE, AROMATICITY, FLEXIBILITY, SS_HELIX, SS_SHEET, PI, MW]:
            props.append(np.array([scale_dict.get(c, 0.0) for c in sequence]))
            
        # Group 2: Disorder-specific (10)
        for scale_dict in [UVERSKY_DISORDER, SOLUBILITY, WIMLEY_WHITE, CONTACT_NUMBER, 
                           TURN_PROPENSITY, COIL_PROPENSITY, BFACTOR, VDW_VOLUME, NCPR, STICKINESS]:
            props.append(np.array([scale_dict.get(c, 0.0) for c in sequence]))
            
        # Group 3: One-hot AA (20)
        for aa in AA_LIST:
            props.append(np.array([1.0 if c == aa else 0.0 for c in sequence]))
            
        # Group 4: ESM PCA (20)
        esm_pca = self.get_esm_features(sequence, n_components=20)
        for i in range(20):
            props.append(esm_pca[:, i])
            
        # Assert total properties = 58
        assert len(props) == 58, f"Expected 58 properties, got {len(props)}"
        
        # Apply windowing to all 58 properties
        features = np.zeros((N, 406))
        f_idx = 0
        for p in props:
            for w in scales:
                half = w // 2
                padded = np.pad(p, (half, half), mode='reflect')
                cumsum = np.cumsum(np.insert(padded, 0, 0))
                moving_avg = (cumsum[w:] - cumsum[:-w]) / float(w)
                features[:, f_idx] = moving_avg
                f_idx += 1
                
        return features

def extract_features(sequence: str) -> np.ndarray:
    """Legacy helper for non-object-oriented calls (defaults to no ESM)."""
    extractor = DisorderFeatureExtractor(use_esm=False)
    return extractor.extract_features(sequence)
