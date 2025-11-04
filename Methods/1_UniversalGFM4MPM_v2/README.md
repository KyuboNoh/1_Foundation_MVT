# Universal GFM4MPM v2 (Transformer-based aggregator DCCA overlap alignment + CLS training)

## Stage-1 Overlap Alignment Training 

The stage-1 trainer freezes encoder features and optimises only the projection
heads using a positive-only, negative-free alignment loss.

```bash
python -m Methods.1_UniversalGFM4MPM_v2.scripts.train --config ./ufm_v2_config_debug.json --debug
```

