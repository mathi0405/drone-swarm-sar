---
name: Benchmark submission
about: Submit a result to the SwarmSAR-Bench v1 leaderboard
title: "[leaderboard] <method name>"
labels: leaderboard
---

**Method**
Name and one-line description (architecture, training budget).

**Result file**
Attach `results/<your_method>.json` produced by:

```bash
python scripts/evaluate.py --ckpt path/to/your.pt --out results/<your_method>.json
```

**Reproducibility checklist**
- [ ] At least 5 seeds
- [ ] Env dynamics, reward, and benchmark maps (seeds 200-219) unmodified
- [ ] Learned methods earn communication (no scripted auto-broadcast)
- [ ] Config file included so reviewers can re-run `evaluate.py`

**Regenerated leaderboard**
Run `python scripts/update_leaderboard.py results/*.json` and include the diff.
