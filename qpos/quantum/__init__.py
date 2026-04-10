SIMULATION_DISCLAIMER = (
    "WARNING: All quantum circuit results are CLASSICALLY SIMULATED via "
    "PennyLane default.qubit. No QPU hardware is used. Results labeled [CS] "
    "reflect exact statevector simulation."
)

print(SIMULATION_DISCLAIMER)

from qpos.quantum import iws_qaoa, ogp_router, qadf_rubric, qubo_builder, vqc, xy_mixer, zne

__all__ = ["iws_qaoa", "ogp_router", "qadf_rubric", "qubo_builder", "vqc", "xy_mixer", "zne"]
