# DINOv3 ViT-L/16 + linear classifier: weights

[manifest.json](manifest.json) records relative filenames, byte sizes, SHA-256 checksums and availability. Available checkpoint files are retained in this local folder. The public repository contains the inventory; checkpoint bytes are excluded from Git.

| Role | Relative file | Availability |
| --- | --- | --- |
| pretrained_encoder | `pretrained_encoder.safetensors` | saved_locally |
| selected_encoder | — | not_available |
| classifier | — | not_available |
| final_epoch_encoder | — | not_available |
| final_epoch_classifier | — | not_available |

The released pretrained encoder is retained. Matched weld training has not run, so a weld classifier and selected weld model are unavailable. Do not substitute a classifier from another model or an older split.

Future runs retain selected and final-epoch artifacts separately under their run directory. Local retention uses hard links where supported to avoid duplicate disk allocation. This is not an independent disk backup. Upstream weight licenses continue to apply.
