import numpy as np
from Bio.PDB import PDBParser as BioPDBParser
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.Data.IUPACData import protein_letters_3to1
    def three_to_one(resname):
        return protein_letters_3to1[resname.capitalize()]

class PDBParser:
    """Parses PDB files into coordinate arrays and sequence information for framework use."""
    
    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path
        self._parser = BioPDBParser(QUIET=True)
        self.structure = self._parser.get_structure("struct", pdb_path)
        
    def get_coordinates(self, atom_name="CA") -> np.ndarray:
        """Extract coordinates for a specific atom type across all residues."""
        coords = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    # Exclude hetero atoms, usually we just want structural amino acids
                    if residue.id[0] != ' ':
                        continue
                        
                    if atom_name in residue:
                        coords.append(residue[atom_name].get_coord())
        return np.array(coords)
        
    def get_sequence(self) -> str:
        """Extract the 1-letter amino acid sequence from the structure."""
        seq = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    resname = residue.get_resname()
                    try:
                        seq.append(three_to_one(resname))
                    except KeyError:
                        seq.append('X')  # Unknown or non-standard
        return "".join(seq)
