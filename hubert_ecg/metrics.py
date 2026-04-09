from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

# Mapping conditions that may appear under slightly different names across
# datasets or in the CinC2020 weights file.
MAPPING = {
    "STE": "STE_",
    "STD": "STD_",
    "NSR": "NORM",
    "IAVB": "1AVB",
    "IIAVB": "2AVB",
    "RBBB": "CRBBB|RBBB",
    "LBBB": "CLBBB|LBBB",
    "CRBBB": "CRBBB|RBBB",
    "CLBBB": "CLBBB|LBBB",
    "SVPB": "PAC|SVPB",
    "PAC": "PAC|SVPB",
    "SVPB|PAC": "PAC|SVPB",
    "LAE": "LAH",
    "LAO": "LAH",
    "RAO": "RAH",
    "RAE": "RAH",
    "LAO/LAE": "LAH",
    "RAO/RAE": "RAH",
    "LQT": "LNGQT",
    "SB": "SBRAD",
    "ST": "STACH",
    "AFIB": "AF",
    "AFLT": "AFL",
    "TAB_": "TAB",
    "SA": "SARRH",
    "QWAVE": "QAB",
    "PVC": "VPC|VPB",
    "VPB": "VPC|VPB",
    "VPC": "VPC|VPB",
    "PVC|VPB": "VPC|VPB",
    "IIAVBII": "2AVB2",
    "CCR": "-ROT",
    "CR": "+ROT",
    "_AVB": "AVB",
    "SVTAC": "SVT",
    "LANFB": "LAFB",
    "INVT": "TINV",
    "ISCIN": "IIS",
    "ISCAN": "ANMIS",
    "DEATH": "SAMITROP-DEATH",
    "MOI": "2AVB1",
    "CHB": "3AVB",
    "CMI": "CMIS",
}


def compute_modified_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Compute the modified confusion matrix used by the CinC2020 scoring metric."""
    num_recordings, num_classes = labels.shape
    A = np.zeros((num_classes, num_classes))
    for i in range(num_recordings):
        normalization = float(
            max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1)
        )
        for j in range(num_classes):
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization
    return A


class CinC2020(Metric):
    """
    PyTorch implementation of the CinC 2020 PhysioNet Challenge metric.

    Only weighted (scored) conditions are considered; unscored conditions in
    the predictions are silently ignored.

    Args:
        conditions: List of condition names corresponding to prediction columns.
        weights_path: Path to the CinC2020 weights CSV file.
        verbose: If ``True``, log intermediate state for debugging.
    """

    def __init__(
        self,
        conditions: list,
        weights_path: str = "/path/to/weights.csv",
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.add_state("gt", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

        self.weights = self._build_weights(weights_path)
        self.conditions = self._build_conditions(conditions)
        self.scored_conditions = self.weights.columns.tolist()

    def _build_conditions(self, conditions: list) -> list:
        conditions = [c.upper() for c in conditions]
        return [MAPPING.get(c, c) for c in conditions]

    def _build_weights(self, weights_path: str) -> pd.DataFrame:
        weights = pd.read_csv(weights_path, index_col=0)
        weights.columns = weights.columns.str.upper()
        weights.index = weights.index.str.upper()
        weights.rename(columns=MAPPING, index=MAPPING, inplace=True)
        return weights

    def update(self, preds: torch.Tensor, gt: torch.Tensor) -> None:
        self.gt.append(gt)
        self.preds.append(preds)

    def compute(self) -> float:
        device = torch.device("cpu")
        gt = dim_zero_cat(self.gt).to(device)
        preds = dim_zero_cat(self.preds).to(device)

        assert len(gt) == len(preds)
        assert gt.size(1) == len(self.conditions)

        if self.verbose:
            logger.info(f"Dataset conditions: {self.conditions}")
            logger.info(f"Scored conditions: {self.scored_conditions}")

        normal_class = "NORM"
        gt_df = pd.DataFrame(gt.numpy(), columns=self.conditions)
        preds_df = pd.DataFrame(preds.numpy(), columns=self.conditions)

        conditions = [c for c in self.conditions if c in self.scored_conditions]
        gt_arr = gt_df[conditions].to_numpy(dtype=float)
        preds_arr = preds_df[conditions].to_numpy(dtype=float)

        task_weights = self.weights.loc[conditions, conditions].to_numpy(dtype=float)
        task_weights_t = torch.from_numpy(task_weights).to(device)

        observed_cm = compute_modified_confusion_matrix(gt_arr, preds_arr)
        observed_score = np.nansum(task_weights_t.numpy() * observed_cm)

        correct_cm = compute_modified_confusion_matrix(gt_arr, gt_arr)
        correct_score = np.nansum(task_weights_t.numpy() * correct_cm)

        if normal_class in conditions:
            normal_index = conditions.index(normal_class)
            inactive = np.zeros(gt_arr.shape, dtype=bool)
            inactive[:, normal_index] = 1
            inactive_cm = compute_modified_confusion_matrix(gt_arr, inactive)
            inactive_score = np.nansum(task_weights_t.numpy() * inactive_cm)
        else:
            inactive_score = 0.0

        if self.verbose:
            logger.info(f"Observed: {observed_score}, Correct: {correct_score}, Inactive: {inactive_score}")

        denom = float(correct_score - inactive_score)
        if denom == 0:
            return 0.0
        return float(observed_score - inactive_score) / denom
