# Weights

`best/` contains the validation-selected encoder and its matching classifier used for the reported results. Both are required. The classifiers are committed in this repository; large encoder files are public release downloads.

Run from the model directory:

```bash
python run.py fetch-weights
python run.py verify-weights
```

| Local file | Bytes | Download |
| --- | ---: | --- |
| `best/encoder.pt` | 2,523,165,301 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--encoder.pt.part02) |
| `best/classifier.pt` | 18,633 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--classifier.pt) |

Part 1 and part 2 are pieces of one encoder file. The encoder is about 2.52 GB, so it was split to fit GitHub’s release-file limit. `python run.py fetch-weights` automatically joins and verifies them. [GitHub limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

In the previous layout, the `final/encoder.pt` row linked to `best` because both used identical frozen encoder weights; their classifier weights differed. Only the best checkpoint is distributed here.

[manifest.json](manifest.json) is the single source for the encoder/classifier pair, SHA-256 checksums, sizes, part checksums and download URLs. Keep the two files from the same model together.

The best inference encoder is not a replacement for the original full pretraining checkpoint when reproducing training. Training prerequisites are documented in the [model README](../README.md).
