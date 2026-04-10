SIMULATION_DISCLAIMER = (
    "WARNING: All quantum circuit results are CLASSICALLY SIMULATED via "
    "PennyLane default.qubit. No QPU hardware is used. Results labeled [CS] "
    "reflect exact statevector simulation."
)

print(SIMULATION_DISCLAIMER)

from qpos.quantum.iws_qaoa import IWSQAOASolver
from qpos.quantum.qadf_rubric import QADFRubric

__all__ = ["IWSQAOASolver", "QADFRubric"]
