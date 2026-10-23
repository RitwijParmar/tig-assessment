# Weld-adapted I-JEPA ViT-H/14 + linear classifier: weights

[manifest.json](manifest.json) records relative filenames, byte sizes, SHA-256 checksums and availability. Available checkpoint files are retained in this local folder. The public repository contains the inventory; checkpoint bytes are excluded from Git.

| Role | Relative file | Availability |
| --- | --- | --- |
| pretrained_encoder | `pretrained_encoder.pth.tar` | saved_locally |
| selected_encoder | `encoder.pt` | saved_locally |
| classifier | `classifier.pt` | saved_locally |
| final_epoch_encoder | — | not_available |
| final_epoch_classifier | — | not_available |

Use `encoder.pt` with `classifier.pt` for the completed adapted model. The original full checkpoint is retained as its pretraining source. The selected export contains the EMA target encoder used for classification; it is not a full optimizer/context/predictor resume checkpoint.

Future runs retain selected and final-epoch artifacts separately under their run directory. Local retention uses hard links where supported to avoid duplicate disk allocation. This is not an independent disk backup. Upstream weight licenses continue to apply.
