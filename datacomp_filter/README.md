# Filtering code for DataComp
## Preparation
Replace or add sources in this folder to the original repository of `mlfoundations/datacomp`.
Download `small` datapool of DataComp and set up environments.
For detail, please refer to the github repository of DataComp.
## Extract statistics and sample embeddings from datapool
```
python baselines/embedding_mean_covariance.py \
    --metadata_dir path-to-metadata \
    --save_path path-to-l14-stats-npz \
    --arch l14

python baselines/embedding_mean_covariance.py \
    --metadata_dir path-to-metadata \
    --save_path path-to-b32-stats-npz \
    --arch b32

python baselines/sample_embeddings.py \
    --metadata_dir path-to-metadata \
    --save_path path-to-l14-sample-npz \
    --arch l14 \
    --sample_ratio 0.002

python baselines/sample_embeddings.py \
    --metadata_dir path-to-metadata \
    --save_path path-to-b32-sample-npz \
    --arch b32 \
    --sample_ratio 0.002
```
## Add each metrics to metadata
This commands add a new column to each parquet data in the metadata.
Note that the same name of the column will be added, so please save each metadata to different locations.
Also, sample embeddings are needed for `MC` and `REVERSE_MC`, and embedding statistics are required for `CENTERIZED` and `WHITEN`.
```
python baselines/information_gain.py \
    --mode MC \
    --metadata_dir path-to-metadata \
    --save_path path-to-new-metadata \
    --b32_stats_path path-to-b32-sample-npz \
    --l14_stats_path path-to-l14-sample-npz

python baselines/information_gain.py \
    --mode REVERSE_MC \
    --metadata_dir path-to-metadata \
    --save_path path-to-new-metadata \
    --b32_stats_path path-to-b32-sample-npz \
    --l14_stats_path path-to-l14-sample-npz

python baselines/information_gain.py \
    --mode CENTERIZED \
    --metadata_dir path-to-metadata \
    --save_path path-to-new-metadata \
    --b32_stats_path path-to-b32-stats-npz \
    --l14_stats_path path-to-l14-stats-npz

python baselines/information_gain.py \
    --mode WHITEN \
    --metadata_dir path-to-metadata \
    --save_path path-to-new-metadata \
    --b32_stats_path path-to-b32-stats-npz \
    --l14_stats_path path-to-l14-stats-npz
```
## Select subset by the metrics
The following command extracts the subset that have top 25% text score.
```
python baselines.py \
    --metadata_dir path-to-new-metadata \
    --save_path path-to-subsets \
    --name info_gain \
    --ig_fraction 0.25 \
    --ig_arch l14 \
    --ig_target txt
```

Once subset data is prepared, execute reshader to build subsets as explained in the DataComp repositoty.