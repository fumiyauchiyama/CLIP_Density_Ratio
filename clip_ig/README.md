# Code for qualitative results and N-gram analysis

## Preparation
### Environments
We conducted experiments at following emvironments:
- Python 3.12.9
- CUDA 11.8.0
- cuDNN 9.5.1
- NCCL 2.23.4-1

To install Python package, run following:
```
pip install -r requirements.txt
```


## Extract top & bottom Information Gain samples
CLIP
```bash
python -m apps.exp_qualitative.metrics_mscoco_2 \
    model.model_type=CLIP \
    model.model_name=ViT-L-14 \
    model.pretrained=laion2b_s32b_b82k \
    dataset.dataset_name=path-to-mscoco2017 \
    mode=MC
```
SigLIP
```bash
python -m apps.exp_qualitative.metrics_mscoco_2 \
    model.model_type=SIGLIP \
    model.model_name=ViT-B-16-SigLIP \
    model.pretrained=webli \
    dataset.dataset_name=path-to-mscoco2017 \
    mode=MC
```

# N-gram analysis
`notebooks/clipnorm_data_mscoco2.ipynb` is a notebook to analyze N-gram characteristics with the output of `apps/exp_qualitative/metrics_mscoco_2.py`.