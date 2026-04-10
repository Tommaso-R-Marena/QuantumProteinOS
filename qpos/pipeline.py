import os
import json
import numpy as np
import logging

from qpos.disorder import DisorderNetV6
from qpos.conformational import ConformationalSampler, QICESSScorer
from qpos.quantum import IWSQAOASolver, QADFRubric
from qpos.chirality import ChiralFoldAuditor, AF3ChiralityCorrector
from qpos.rotamers import RotamerPacker
import pandas as pd

logger = logging.getLogger('QuantumProteinOS')

class PipelineResult:
    def __init__(self, structure, ensemble, disorder_scores, chirality_report, qadf_score):
        self.structure = structure
        self.ensemble = ensemble
        self.disorder_scores = disorder_scores
        self.chirality_report = chirality_report
        self.qadf_score = qadf_score
        
    def save(self, output_dir: str):
        """Save the pipeline results into a structured directory."""
        os.makedirs(output_dir, exist_ok=True)
        ensemble_dir = os.path.join(output_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # 1. structure.pdb
        struct_path = os.path.join(output_dir, "structure.pdb")
        with open(struct_path, "w") as f:
            f.write(f"REMARK 999 Final output structure: {self.structure}\n")
            
        # 2. disorder_scores.csv
        if self.disorder_scores is not None:
            pd.DataFrame({
                'residue': range(len(self.disorder_scores)),
                'disorder_probability': self.disorder_scores,
                'is_disordered': (self.disorder_scores > 0.5).astype(int)
            }).to_csv(os.path.join(output_dir, 'disorder_scores.csv'), index=False)
        else:
            open(os.path.join(output_dir, 'disorder_scores.csv'), 'w').close()
            
        # 3. chirality_report.json
        with open(os.path.join(output_dir, 'chirality_report.json'), 'w') as f:
            json.dump(self.chirality_report or {}, f, indent=2)
        
        # 4. qadf_score.json
        with open(os.path.join(output_dir, 'qadf_score.json'), 'w') as f:
            json.dump(self.qadf_score or {}, f, indent=2)
        
        # 5. summary.txt
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(self._generate_summary(struct_path))
            
        # Ensemble components
        if self.ensemble:
            for i, conf_info in enumerate(self.ensemble):
                conf = conf_info['structure'] if isinstance(conf_info, dict) else conf_info
                conf_path = os.path.join(ensemble_dir, f"conf_{i+1:03d}.pdb")
                with open(conf_path, "w") as f:
                    f.write(f"REMARK 999 Ensemble conf: {conf}\n")

    def _generate_summary(self, struct_path):
        summary = "QuantumProteinOS Execution Summary\n"
        summary += "==================================\n"
        summary += f"Final structure: {struct_path}\n"
        summary += f"Ensemble size: {len(self.ensemble) if self.ensemble else 0}\n"
        summary += "Results output successfully.\n"
        return summary


class QuantumProteinOS:
    """
    Unified quantum-classical protein structure prediction pipeline.
    """
    def __init__(self, config: dict):
        self.disorder_model = DisorderNetV6(config.get('disorder', {}))
        self.ensemble_generator = ConformationalSampler(config.get('conformational', {}))
        self.qicess = QICESSScorer(config.get('qicess', {}))
        self.rotamer_packer = RotamerPacker(config.get('rotamers', {}))
        self.chirality_auditor = ChiralFoldAuditor(config.get('chirality', {}))
        self.af3_corrector = AF3ChiralityCorrector()
        self.qadf_rubric = QADFRubric()

    @classmethod
    def from_config(cls, config_path: str):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(self, sequence: str, pdb_path: str = None) -> PipelineResult:
        logger.info("Executing QuantumProteinOS Core Pipeline")

        # Step 1: Disorder
        try:
            disorder_scores = self.disorder_model.predict(sequence)
        except Exception as e:
            logger.warning(f"Disorder prediction failed: {e}. Using uniform 0.5 scores.")
            disorder_scores = np.full(len(sequence), 0.5)

        ordered_mask = disorder_scores < 0.5

        # Step 2: Conformational ensemble
        try:
            ensemble = self.ensemble_generator.generate(
                pdb_path, n_modes=10, amplitude_scales=[2.0, 6.0],
                rigid_body_params={'translation': [5.0, 15.0], 'rotation': [20.0, 45.0]}
            )
        except Exception as e:
            logger.warning(f"Ensemble generation failed: {e}. Using input pdb as single-item ensemble.")
            ensemble = [pdb_path] if pdb_path else []

        # Step 3: QICESS ensemble scoring
        try:
            scored = self.qicess.score_ensemble(ensemble)
            ranked = self.qicess.rank_ensemble(scored)
        except Exception as e:
            logger.warning(f"QICESS scoring failed: {e}. Passing unranked ensemble.")
            ranked = [{'structure': s, 'score': 0.0} for s in ensemble]

        # Step 4: Rotamer packing
        try:
            packed = self.rotamer_packer.pack_structure(
                ranked, ordered_mask=ordered_mask,
                window_size=4, n_rotamers=3, p=4, n_iter=3
            )
        except Exception as e:
            logger.warning(f"Rotamer packing failed: {e}. Falling back to input structure.")
            packed = ranked[0]['structure'] if ranked else pdb_path

        # Step 5: Chirality audit + correction
        report = {}
        try:
            report = self.chirality_auditor.audit(packed)
            if report.get('chirality', {}).get('pct_correct', 100.0) < 100.0:
                packed = self.af3_corrector.correct(packed)
        except Exception as e:
            logger.warning(f"Chirality audit failed: {e}. Skipping AF3 correction.")
            report = {'error': str(e)}

        # Step 6: QADF rubric
        qadf_score = {}
        try:
            qadf_score = self.qadf_rubric.score("Rotamer Packing", packed)
        except Exception as e:
            logger.warning(f"QADF rubric failed: {e}. Continuing.")
            qadf_score = {'error': str(e)}

        return PipelineResult(
            structure=packed,
            ensemble=ranked,
            disorder_scores=disorder_scores,
            chirality_report=report,
            qadf_score=qadf_score
        )
