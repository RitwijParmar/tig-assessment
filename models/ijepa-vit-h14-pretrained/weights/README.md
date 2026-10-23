# Original pretrained I-JEPA ViT-H/14 + linear classifier: weights

[manifest.json](manifest.json) records relative filenames, byte sizes, SHA-256 checksums and availability. Available checkpoint files are retained in this local folder. The public repository contains the inventory; checkpoint bytes are excluded from Git.

| Role | Relative file | Availability |
| --- | --- | --- |
| pretrained_encoder | `pretrained_encoder.pth.tar` | saved_locally |
| selected_encoder | — | not_available |
| classifier | `classifier.pt` | saved_locally |
| final_epoch_encoder | — | not_available |
| final_epoch_classifier | — | not_available |

Use `target_encoder` from `pretrained_encoder.pth.tar` with this baseline’s `classifier.pt`. This encoder has no weld JEPA adaptation. The shared loader supports `encoder_role="target_encoder"`; see the model card for a complete example.

Future runs retain selected and final-epoch artifacts separately under their run directory. Local retention uses hard links where supported to avoid duplicate disk allocation. This is not an independent disk backup. Upstream weight licenses continue to apply.
