# Weights

`best/` contains the validation-selected encoder, matching classifier and checkpoint descriptor used for the reported results.

```bash
python run.py fetch-weights
python run.py verify-weights
```

| File | Download |
| --- | --- |
| `best/encoder.pt` | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--encoder.pt.part02) |
| `best/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--classifier.pt) |

**Note:** Part 1 and part 2 are pieces of one encoder file. The encoder is about 2.52 GB, so it was split to fit GitHub’s release-file limit. `python run.py fetch-weights` automatically joins and verifies them. [GitHub limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

In the previous layout, the `final/encoder.pt` row linked to `best` because both used identical frozen encoder weights; their classifier weights differed. Only the best checkpoint is distributed here.

[manifest.json](manifest.json) records sizes, SHA-256 hashes and checkpoint pairing.
