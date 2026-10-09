"""
Backend of the demo UI (no Streamlit code, so it can be tested headlessly). Everything is reused from the project, nothing is re-implemented:

  * pair table / cache loading        train_img_single_frame.load_rows
  * Transformer history + inference   train_img_transformer.build_histories / predict          (frozen checkpoint, full history)
  * ConvLSTM sequences + inference    train_img_convlstm.build_sequences / predict_uint8       (frozen checkpoint, K=4: the last 4 frames)
  * metrics                           train_img_single_frame.score_pairs  (skimage RGB SSIM / PSNR + structure SSIM, as in every README table)
  * missing-observation dropping      robustness_missing_obs.drop_indices (imported unchanged)

A UI state (plant, cutoff, drop level) maps to ONE real test pair (the cutoff chooses how many of the plant's input frames are used, i.e. which
target) and, if drop > 0, a fixed random subset of that pair's input frames removed. The seed is derived from (plant, target day, drop level), so
the same state always gives the same result. CPU inference only.

Bundle layout (see scripts/export_demo_bundle.py):
    <bundle>/image_pairs_test.parquet            the 1,233 cleaned TEST pairs
    <bundle>/cache/images256.npy + images256_index.parquet   256x256 uint8 images used by those pairs (inputs incl. history, targets)
    <bundle>/checkpoints/img_transformer_best.pt, img_convlstm_k4_best.pt
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_convlstm import TemporalFiLMUNet                                   # noqa: E402
from img_transformer import TemporalTransformerUNet                         # noqa: E402
from robustness_missing_obs import drop_indices                             # noqa: E402  (validated for the diameter models; reused unchanged)
from train_img_convlstm import build_sequences, predict_uint8               # noqa: E402
from train_img_single_frame import load_rows, score_pairs                   # noqa: E402
from train_img_transformer import build_histories, predict                  # noqa: E402

DROP_LEVELS = [0, 25, 50, 75]
METRICS = ["structure_ssim", "ssim", "psnr"]
CKPT_TF, CKPT_CL = "img_transformer_best.pt", "img_convlstm_k4_best.pt"


def state_seed(plant_id, target_day, drop_pct):
    """Fixed seed per (plant, cutoff, drop level)."""
    return zlib.crc32(f"{plant_id}|{int(target_day)}|{int(drop_pct)}".encode()) & 0x7FFFFFFF


class DemoBackend:
    def __init__(self, bundle_dir, device="cpu"):
        self.device = torch.device(device)
        cache = os.path.join(bundle_dir, "cache")
        self.pairs = load_rows(os.path.join(bundle_dir, "image_pairs_test.parquet"), cache)
        assert (self.pairs.split == "test").all(), "the bundle must contain TEST pairs only"
        self.arr = np.load(os.path.join(cache, "images256.npy"), mmap_mode="r")
        self.cache_idx = pd.read_parquet(os.path.join(cache, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
        ck = os.path.join(bundle_dir, "checkpoints")
        sd = torch.load(os.path.join(ck, CKPT_TF), map_location="cpu")
        a = sd["args"]
        assert a.get("max_history", 0) == 0, "expected the full-history Transformer checkpoint"
        self.tf = TemporalTransformerUNet(d_model=a["d_model"], nhead=a["nhead"], num_layers=a["num_layers"], dim_ff=a["dim_ff"],
                                          dropout=a["dropout"], grad_checkpoint=False)
        self.tf.load_state_dict(sd["model"])
        self.tf.eval().to(self.device)
        sd = torch.load(os.path.join(ck, CKPT_CL), map_location="cpu")
        assert sd["args"]["k"] == 4, "expected the K=4 ConvLSTM checkpoint"
        self.cl_norm, self.cl_k = sd["norm"], sd["args"]["k"]
        self.cl = TemporalFiLMUNet()
        self.cl.load_state_dict(sd["model"])
        self.cl.eval().to(self.device)

    # ---- catalogue ----
    def plants(self):
        d = self.pairs.drop_duplicates("plant_id")[["plant_id", "field"]].sort_values("plant_id")
        return list(d.itertuples(index=False, name=None))

    def cutoffs(self, plant_id):
        """Available cutoffs (= real test pairs) of a plant: [(n_input_frames, target_day), ...]."""
        d = self.pairs[self.pairs.plant_id == plant_id].sort_values("n_input_frames")
        return [(int(r.n_input_frames), int(r.target_day)) for r in d.itertuples()]

    # ---- one UI state ----
    def run(self, plant_id, n_input_frames, drop_pct):
        d = self.pairs[(self.pairs.plant_id == plant_id) & (self.pairs.n_input_frames == n_input_frames)]
        assert len(d) == 1, f"no unique test pair for {plant_id} with {n_input_frames} input frames"
        row = d.iloc[0].to_dict()
        fps, days = list(row["input_filepaths"]), [int(x) for x in row["input_days"]]
        seed = state_seed(plant_id, row["target_day"], drop_pct)
        if drop_pct > 0:
            keep = drop_indices(len(days), drop_pct / 100.0, np.random.RandomState(seed))
        else:
            keep = list(range(len(days)))
        new = dict(row)
        new["input_filepaths"], new["input_days"] = [fps[i] for i in keep], [days[i] for i in keep]
        new["input_last_filepath"], new["input_last_day"] = fps[keep[-1]], days[keep[-1]]
        new["n_input_frames"] = len(keep)
        new["gap_days"] = int(row["target_day"]) - days[keep[-1]]
        new["row_in"] = self.cache_idx[fps[keep[-1]]]
        new["offset"] = float(new["gap_days"])
        df = pd.DataFrame([new])
        Hs = build_histories(df, self.cache_idx, 0)
        tf_pred, _ = predict(self.tf, self.arr, df, Hs, self.device)
        S = build_sequences(df, self.cache_idx, self.cl_k, self.cl_norm)
        cl_pred = predict_uint8(self.cl, self.arr, df, {"rows": S[0], "gaps": S[1], "mask": S[2]}, self.device)
        cf_pred = np.stack([np.asarray(self.arr[int(new["row_in"])])])
        out = {"plant_id": plant_id, "field": row["field"], "target_day": int(row["target_day"]), "n_available": len(days),
               "input_days": days, "kept": keep, "dropped": [i for i in range(len(days)) if i not in keep], "seed": seed, "drop_pct": drop_pct,
               "last_kept_day": days[keep[-1]], "offset": int(new["gap_days"]),
               "input_images": [np.asarray(self.arr[self.cache_idx[f]]) for f in fps],
               "target": np.asarray(self.arr[int(row["row_tg"])]), "pred": {}, "metrics": {}}
        for name, p in (("Transformer", tf_pred), ("ConvLSTM", cl_pred), ("Copy-forward", cf_pred)):
            out["pred"][name] = p[0]
            m = score_pairs(p, self.arr, df).iloc[0]
            out["metrics"][name] = {k: float(m[k]) for k in METRICS}
        return out
