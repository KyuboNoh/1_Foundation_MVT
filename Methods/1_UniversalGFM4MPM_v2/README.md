# Universal GFM4MPM v2 (Transformer-based aggregator DCCA overlap alignment + CLS training)

## Stage-1 Overlap Alignment Training 

The stage-1 trainer freezes encoder features and optimises only the projection
heads using a positive-only, negative-free alignment loss.

```bash
python -m Methods.1_UniversalGFM4MPM_v2.scripts.train --config ./ufm_v2_config_debug.json --debug
```

## Stage-2 nnPU for Global\Overlap using Global data
```bash
python -m Methods.1_UniversalGFM4MPM_v2.scripts.train --config ./ufm_v2_config_debug.json --read-dcca --dcca-weights-path /home/wslqubuntu24/Research/Data/1_Foundation_MVT_Result/2_UFM_v2/Ex1_TransformerDCCA/overlap_alignment_stage1_dcca.pt --train-cls --debug
```
