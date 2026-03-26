# Implementation of CLIP Importance-Weighted Learning
This folder contains sources for our IWL method that can be used in OpenCLIP codebase.

Replace following sources to our python files in the github repository `mlfoundations/open_clip`:
- `src/open_clip/loss.py`
- `src/open_clip_train/train.py`
- `src/open_clip_train/params.py`

Training can be executed by following commands:
```
torchrun --nproc_per_node 8 -m open_clip_train.main \
  --train-data 'path-to-cc12m' \
  --train-num-samples 12423374  \
  --dataset-type webdataset \
  --model ViT-B-32 \
  --report-to wandb \
  --wandb-project-name project-name \
  --batch-size 1000 \
  --precision amp \
  --workers 8 \
  --epochs 10 \
  --iwl --iwl-str 'flowers' \
  --iwl-ref-model-name ViT-L-14 --iwl-ref-model-pretrained laion2b_s32b_b82k \
  --iwl-logit-scale 10 \
  --imagenet-val path-to-imagenet_1k/val
```
where `--iwl-str` specify the domain and will be embedded in the template `a photo of {args.iwl_str}`.

Evaluations of downstream task performance are done by `LAION-AI/CLIP_benchmark`.