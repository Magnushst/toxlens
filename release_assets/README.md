# Release assets

These binaries are intentionally excluded from ordinary Git history. Upload
them as assets attached to the corresponding GitHub release or deposit them in
the manuscript's archival data record. Keep the filenames unchanged because
the README commands and checksum manifest refer to them directly.

- `checkpoints/main/toxlens_single_best.ckpt`: selected single-model checkpoint.
- `checkpoints/ensemble/toxlens_seed_42.ckpt`: selected seed-42 ensemble member.
- `checkpoints/ensemble/toxlens_seed_43.ckpt`: selected seed-43 ensemble member.
- `checkpoints/ensemble/toxlens_seed_44.ckpt`: selected seed-44 ensemble member.
- `checkpoints/ensemble/toxlens_seed_45.ckpt`: selected seed-45 ensemble member.
- `checkpoints/ensemble/toxlens_seed_46.ckpt`: selected seed-46 ensemble member.
- `checkpoints/tox21/toxlens_tox21_challenge.ckpt`: selected Tox21 Challenge checkpoint.

Verify every downloaded asset against `checksums.sha256` before evaluation.

`graph_bundle/pyg_graphs_class.pkl` is the later of the two archived graph
serialisations and reproduces the selected checkpoint predictions. Its training
partition retains 5,394 legacy records with no labels for any primary endpoint;
the masked objective assigns those records no supervised loss. Use the clean
21,578-row `data/splits/train.csv` for new training, and use the graph bundle
only when exact checkpoint-era feature tensors are required.
