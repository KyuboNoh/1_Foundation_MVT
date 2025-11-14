# Universal GFM4MPM v2 (Transformer-based aggregator DCCA overlap alignment + CLS training)

## Stage-1 Overlap Alignment Training 

The stage-1 trainer freezes encoder features and optimises only the projection
heads using a positive-only, negative-free alignment loss.

```bash
python -m Methods.1_UniversalGFM4MPM_v2_module.scripts.train --config ./ufm_v2_config_debug_256.json --read-dcca --dcca-weights-path /home/qubuntu25/Desktop/Research/Data/1_Foundation_MVT_Result/2_UFM_v2/TransAggDCCA_Ex1_dim256/overlap_alignment_stage1_dcca.pt --train-cls-1 --train-cls-1-Method PN```
```
