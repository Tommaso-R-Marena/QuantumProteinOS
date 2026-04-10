try:
    import torch
    import esm
    HAS_ML = True
except ImportError:
    HAS_ML = False

class ESMCPUFeatureExtractor:
    """Uses fair-esm 8M CPU model to extract baseline representations."""
    def __init__(self):
        if HAS_ML:
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.eval()
        else:
            self.model = None

    def get_embeddings(self, sequence: str):
        if not HAS_ML:
            raise ImportError(
                "ESM-2 embeddings require the ML extras. "
                "Install with: pip install 'qpos[ml]'"
            )
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        # Return only sequence tokens (remove start/end)
        return token_representations[0, 1 : len(sequence) + 1]
