# Result provenance and public claims

This project studies cooperative search and rescue in simulation. Hardware deployment is not established by the presence of a ROS2 skeleton or AirSim adapter.

## Existing artifacts

- The README reports historical heuristic and learned-policy summaries.
- `docs/model_zoo.md` and `docs/leaderboard.md` report different values for corresponding seed labels. Do not merge or compare these tables until checkpoint identity and evaluation configuration are reconciled.
- The old activity-based SIS communication table was retired. Use utility-based SIS, with its exact implementation revision recorded.
- Illustrative learning curves are not measured training evidence.

## Required evaluation record

For every new result, preserve: Git commit, checkpoint SHA-256, resolved environment configuration, controller type, training seed, evaluation map seeds, metric version, raw per-episode metrics, runtime/hardware, and evaluation command. Report means and uncertainty across independent training seeds; map variation alone is not training-seed uncertainty.

## Next controlled experiment

Evaluate the same frozen checkpoints and held-out maps with normal, disabled, and shuffled message content. Keep sensing, map seeds, policy weights and episode limits fixed. Define the shuffle intervention precisely; changing radio range can also alter peer observations and is not a content-only intervention. Report rescues, coverage, collisions and assisted rescues alongside SIS. Label a negative result as a result, not as successful learned coordination.

## Contribution statement

Describe your implemented modules, experimental decisions, and analyses separately from dependencies and upstream frameworks. Verify individual authorship before publishing a personal contribution claim.
