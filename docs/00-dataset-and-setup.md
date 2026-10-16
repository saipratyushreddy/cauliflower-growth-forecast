# Dataset, repository structure and environment setup

[Back to the project overview](../README.md)

## Dataset

[GrowliFlower](https://huggingface.co/datasets/Voxel51/GrowliFlower)
(curated FiftyOne release, `Voxel51/GrowliFlower`), consolidating the
`GrowliFlowerL`/`R`/`D` subsets from the original release:

> Kierdorf, J., Junker-Frohn, L. V., Delaney, M., Olave, M. D., Burkart,
> A., Jaenicke, H., Muller, O., Rascher, U., & Roscher, R. (2022).
> GrowliFlower: An image time series dataset for GROWth analysis of
> cauLIFLOWER. *arXiv preprint arXiv:2204.00294*.

**License caveat (unresolved):** no license is stated in the PhenoRoam
catalog record, the shipped dataset card, or the source paper. This
should be confirmed with the corresponding author (Jana Kierdorf,
`jkierdorf@uni-bonn.de`) before any wider sharing, publication, or reuse
beyond this internal research checkpoint.

Runs on UNL Holland Computing Center's Swan cluster (SLURM). Local
machine is used only for lightweight metadata inspection/pipeline
construction (no GPU, no full image downloads) — anything touching the
actual images or a GPU runs on Swan.


## Repo structure

```
data/            Metadata/pairs/embeddings artifacts (gitignored; regenerate via scripts below)
src/             Pipeline scripts
scripts/         SLURM job scripts
notebooks/       Exploration only, not part of the pipeline
outputs/         Reports, plots, saved figures (gitignored; regenerate via scripts below)
checkpoints/     Trained model weights (gitignored; regenerate via scripts below)
```

`app.py` + `DEMO.md`: Streamlit demo of the image-to-image track (laptop, CPU; needs a bundle exported from Swan, see `DEMO.md`).

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Swan, `pip install torch` pulls a CUDA-enabled build automatically
(bundled `nvidia-cu12*` runtime packages — confirmed via
`torch.__version__` showing `+cu128`); no separate `module load cuda` is
needed on this cluster.
