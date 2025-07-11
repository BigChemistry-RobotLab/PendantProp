import pandas as pd
import pickle
import json
import numpy as np
import hydra
import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from src.dataloader import SurfProDB, DataSplit
from src.model import AttentiveFPModel
from src.converter import (
    calc_langmuir, neglog_to_raw, raw_to_neglog,
    calc_charge_from_smiles, calc_n_from_charge
)
from rdkit import Chem

torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def predict(cfg: DictConfig) -> None:
    print("PREDICT CONFIG from params.yaml")
    cfg = OmegaConf.load("./params.yaml")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)

    workdir = f"{cfg.host.workdir}/out/pendantprop"
    test_df = pd.read_csv('./data/surfactant_SMILES.csv')
    test_df = test_df.loc[:, ['surfactant', 'SMILES', 'type']]
    propnames = ["pCMC", "AW_ST_CMC",
                 "Gamma_max", "Area_min", "Pi_CMC", "pC20"]

    test_loader = DataSplit(
        smiles=test_df.SMILES,
        labels=[[np.nan]*len(propnames) for _ in test_df.SMILES],
        types=[np.nan for _ in test_df.SMILES],
        propnames=propnames,
        featurize='graph'
    ).loader(shuffle=False, num_workers=4)

    with open(f"{cfg.host.workdir}/data/{cfg.task.name}/surfpro.pkl", "rb") as f:
        surfpro = pickle.load(f)


    fold_preds = []
    for fold in range(cfg.task.n_splits):
        kwargs = {
            "props": propnames,
            "hidden_channels": cfg.model.hidden_channels,
            "out_channels": cfg.model.out_channels,
            "num_layers": cfg.model.num_layers,
            "num_timesteps": cfg.model.num_timesteps,
            "dropout": cfg.model.dropout,
        }
        model = AttentiveFPModel(**kwargs)

        model.load_state_dict(torch.load(
            f"./final/all/AttentiveFP-64d-all/models/model{fold}.pt"))

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=cfg.host.device if torch.cuda.is_available() else "auto",
            precision=32 if cfg.task.scale else "bf16-mixed",
            default_root_dir=f"{workdir}/models/",
        )

        preds = trainer.predict(model, test_loader)
        preds = torch.cat([batch.get("preds")
                          for batch in preds]).float().numpy()

        # unscale preds (in multi-property / all-property setting)
        if cfg.task.scale:
            preds = surfpro.unscale(preds)
            print("preds rescaled to original units")

        fold_preds.append(preds)

    fold_preds = np.array(fold_preds)
    for i, prop in enumerate(propnames):
        test_df[prop] = np.mean(fold_preds[:, :, i], axis=0)
        # test_df[f'{prop}_std'] = np.std(fold_preds[:, :, i], axis=0)

    test_df['Gamma_max'] = test_df['Gamma_max'] / 1e6

    test_df['CMC'] = test_df.pCMC.apply(neglog_to_raw)
    test_df['C20'] = test_df.pC20.apply(neglog_to_raw)
    test_df['charge'] = test_df.SMILES.apply(calc_charge_from_smiles)
    test_df['n'] = test_df.charge.apply(calc_n_from_charge)

    langmuir = [calc_langmuir(sft, gam, cmc, n, T=25) for (sft, gam, cmc, n) in
                zip(test_df.AW_ST_CMC, test_df.Gamma_max, test_df.CMC, test_df.n)]
    test_df['K_L'] = langmuir
    # test_df['p(K_L)'] = test_df.K_L.apply(raw_to_neglog)

    test_df = test_df.drop(columns=['Area_min', 'Pi_CMC'])
    test_df = test_df.sort_values('surfactant')

    # store per-fold predictions in test_df used in evaluate.py()
    test_df.reset_index(drop=True).to_csv(
        f"{workdir}/surfpro_exp_predicted.csv", index=False)


def canonicalize_smiles(smiles_string):
    """
    Canonicalizes a SMILES string using RDKit.
    Returns the canonical SMILES string, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None


if __name__ == "__main__":
    predict()
