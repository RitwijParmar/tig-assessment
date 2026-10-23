# MobileNetV5-300M + linear classifier weights

Best and final checkpoints are stored under `best/` and `final/` in this folder. Run:

```bash
python run.py fetch-weights
python run.py verify-weights
```

- `pretrained_encoder.safetensors`: official released pretrained encoder; not a weld-trained checkpoint [mobilenetv5-300m--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/mobilenetv5-300m--pretrained_encoder.safetensors)
- `best/encoder.pt`: best trained supervised encoder [mobilenetv5-300m--best--encoder.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--encoder.pt)
- `best/classifier.pt`: best trained classifier [mobilenetv5-300m--best--classifier.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--classifier.pt)
- `final/encoder.pt`: final trained supervised encoder [mobilenetv5-300m--final--encoder.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--final--encoder.pt)
- `final/classifier.pt`: final trained classifier [mobilenetv5-300m--final--classifier.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--final--classifier.pt)
- `best/pretrained_encoder.safetensors`: Released encoder base retained with the best checkpoint. [mobilenetv5-300m--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/mobilenetv5-300m--pretrained_encoder.safetensors)
- `final/pretrained_encoder.safetensors`: Released encoder base retained with the final checkpoint. [mobilenetv5-300m--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/mobilenetv5-300m--pretrained_encoder.safetensors)

The downloader verifies bytes and SHA-256 and reuses identical local encoder bytes with hard links. Best is validation-selected; final is the last completed epoch of its selected configuration. Raw data, frame predictions and private optimizer states are excluded from public downloads.
