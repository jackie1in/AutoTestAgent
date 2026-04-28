# Manual + Auto Hybrid Cartography Acceptance

## Phase 1 - Minimal Viable Hybrid Capture

Scope:
- Manual capture entrypoints exist for graph-assisted and raw modes.
- Both manual and auto data paths produce unified transition candidates.
- Unified persistence path stores transitions with source metadata.

Acceptance checklist:
- Run manual mode with a sample events JSON and verify transitions are stored.
- Verify transition properties include `source_type` and `operator_id`.
- Verify generated checkpoints from manual capture use `origin_type=manual`.

## Phase 2 - Human-in-the-loop Orchestration

Scope:
- Automatic exploration emits intervention tasks from quality/failure signals.
- Intervention task payload is persisted in session stats for downstream tooling.

Acceptance checklist:
- Trigger low layout confidence and action failures in a test session.
- Verify session stats include:
  - `intervention_task_count`
  - `intervention_tasks`
- Verify task reasons include expected categories (`low_layout_confidence`, etc).

## Phase 3 - Merge Quality and Stability

Scope:
- Merge logic uses source priority so manual transitions are not downgraded by auto conflicts.
- Transition conflict behavior remains deterministic and traceable.

Acceptance checklist:
- Re-ingest equivalent transition pairs from auto and manual sources.
- Verify higher-priority manual source can upgrade existing transition source.
- Verify confidence adjustments follow source-priority and selector fallback rules.
