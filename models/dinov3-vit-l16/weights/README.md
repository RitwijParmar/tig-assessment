# DINOv3 ViT-L/16 + linear classifier weights

Best and final checkpoints are stored under `best/` and `final/` in this folder. Run:

```bash
python run.py fetch-weights
python run.py verify-weights
```

- `pretrained_encoder.safetensors`: official released pretrained encoder; not a weld-trained checkpoint [dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)
- `best/pretrained_encoder.safetensors`: best frozen encoder; byte-identical to the released pretrained encoder. [dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)
- `best/classifier.pt`: best trained classifier [dinov3-vit-l16--best--classifier.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--best--classifier.pt)
- `final/pretrained_encoder.safetensors`: final frozen encoder; byte-identical to the released pretrained encoder. [dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)
- `final/classifier.pt`: final trained classifier [dinov3-vit-l16--final--classifier.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--final--classifier.pt)
- `best/pretrained_encoder.safetensors`: Released encoder base retained with the best checkpoint. [dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)
- `final/pretrained_encoder.safetensors`: Released encoder base retained with the final checkpoint. [dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)

The downloader verifies bytes and SHA-256 and reuses identical local encoder bytes with hard links. Best is validation-selected; final is the last completed epoch of its selected configuration. Raw data, frame predictions and private optimizer states are excluded from public downloads.
