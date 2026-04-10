try:
    import torch
    import esm
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False
    import numpy as np

class ESMCPUFeatureExtractor:
    """Uses fair-esm 8M CPU model to extract baseline representations."""
    def __init__(self):
        if HAS_ML:
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.eval()
        else:
            self.model = None

    def get_embeddings(self, sequence: str) -> np.ndarray:
        if not HAS_ML:
            raise ImportError(
                "ESM-2 embeddings require the ML extras. "
                "Install with: pip install 'qpos[ml]'"
            )
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[6], return_contacts=False)
        
        # Strip <cls> token (index 0) and <eos> token (index -1)
        # result shape: (1, seq_len+2, 320) -> (seq_len, 320)
        embeddings = results["representations"][6][0, 1:-1, :].cpu().numpy()
        
        assert embeddings.shape == (len(sequence), 320), \
            f"ESM shape mismatch: got {embeddings.shape}, expected ({len(sequence)}, 320)"
            
        return embeddings
