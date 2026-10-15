"""
Sharpness / perceptual-distance scores of the EXISTING models and of copy-forward on the cleaned TEST pairs, with the same metrics as the sharpness ablation
(train_img_sharp.py): per-pair structure SSIM / RGB SSIM / PSNR, gradient-magnitude and Laplacian-variance sharpness (prediction vs real target) and the VGG16
feature distance. Inference only, frozen checkpoints of the original project directory. Writes outputs/img_sharp_existing_<model>_per_pair.csv.

Usage:
    python src/sharpness_existing.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs \
        --sc-ckpt checkpoints/img_single_frame_best.pt --cl-ckpt checkpoints/img_convlstm_k4_best.pt --tf-ckpt checkpoints/img_transformer_best.pt --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from img_convlstm import TemporalFiLMUNet
from img_film_unet import FiLMUNet
from img_sharp import VGGPerceptual, sharpness_table
from img_transformer import TemporalTransformerUNet
from train_img_convlstm import build_sequences, predict_uint8 as predict_cl
from train_img_sharp import evaluate
from train_img_single_frame import load_rows, score_pairs
from train_img_transformer import build_histories, predict as predict_tf


def vgg_dist(vgg, preds, tg, device):
    out = []
    for i in range(0, len(preds), 32):
        p = torch.from_numpy(preds[i:i + 32]).to(device).permute(0, 3, 1, 2).float() / 255
        t = torch.from_numpy(tg[i:i + 32]).to(device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            out.append(vgg(p, t, reduce=False).cpu().numpy())
    return np.concatenate(out)


def summarize(name, res):
    print(f"{name:14s} structure SSIM {res.structure_ssim.mean():.4f} | RGB SSIM {res.ssim.mean():.4f} | PSNR {res.psnr.mean():.2f} | VGG dist {res.vgg_dist.mean():.4f} | "
          f"sharpness ratio grad {res.grad_pred.mean() / res.grad_target.mean():.3f} lap {res.lap_pred.mean() / res.lap_target.mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sc-ckpt", required=True)
    ap.add_argument("--cl-ckpt", required=True)
    ap.add_argument("--tf-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pretrained-vgg", dest="pretrained_vgg", action="store_false")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = torch.device(a.device)
    pairs = load_rows(a.image_pairs, a.cache_dir)
    arr = np.load(os.path.join(a.cache_dir, "images256.npy"), mmap_mode="r")
    cache_idx = pd.read_parquet(os.path.join(a.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    te = pairs[pairs.split == "test"].sort_values("pair_id").reset_index(drop=True)
    tg = np.stack([np.asarray(arr[int(r)]) for r in te["row_tg"]])
    vgg = VGGPerceptual(pretrained=a.pretrained_vgg).to(dev)

    def finish(name, preds):
        res = score_pairs(preds, arr, te)
        sh = sharpness_table(preds, tg)
        for k in ("grad_pred", "grad_target", "lap_pred", "lap_target"):
            res[k] = sh[k]
        res["vgg_dist"] = vgg_dist(vgg, preds, tg, dev)
        res.to_csv(os.path.join(a.out_dir, f"{a.tag}img_sharp_existing_{name}_per_pair.csv"), index=False)
        summarize(name, res)

    finish("real_target", tg)                                                            # reference: the real image against itself (sharpness ratio 1, VGG 0)
    finish("copy_forward", np.stack([np.asarray(arr[int(r)]) for r in te["row_in"]]))
    sd = torch.load(a.sc_ckpt, map_location="cpu")
    m = FiLMUNet(); m.load_state_dict(sd["model"]); m.eval().to(dev)
    from train_img_single_frame import predict_uint8 as predict_sc
    finish("step_c", predict_sc(m, arr, te, sd["norm"], dev))
    sd = torch.load(a.cl_ckpt, map_location="cpu")
    m = TemporalFiLMUNet(); m.load_state_dict(sd["model"]); m.eval().to(dev)
    S = build_sequences(te, cache_idx, 4, sd["norm"])
    finish("convlstm_k4", predict_cl(m, arr, te, {"rows": S[0], "gaps": S[1], "mask": S[2]}, dev))
    sd = torch.load(a.tf_ckpt, map_location="cpu")
    ar = sd["args"]
    m = TemporalTransformerUNet(d_model=ar["d_model"], nhead=ar["nhead"], num_layers=ar["num_layers"], dim_ff=ar["dim_ff"], dropout=ar["dropout"], grad_checkpoint=False)
    m.load_state_dict(sd["model"]); m.eval().to(dev)
    p, _ = predict_tf(m, arr, te, build_histories(te, cache_idx, 0), dev)
    finish("transformer", p)


if __name__ == "__main__":
    main()
