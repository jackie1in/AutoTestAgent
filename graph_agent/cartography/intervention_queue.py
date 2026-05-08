from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InterventionTriggerConfig:
    min_low_layout_hits: int = 2
    min_failed_action_count: int = 3
    min_semantic_conflicts: int = 2
    force_on_complex_context: bool = True


@dataclass(slots=True)
class InterventionTask:
    task_id: str
    reason: str
    source_url: str
    page_type: str = ""
    context: dict[str, object] = field(default_factory=dict)
    status: str = "open"


class InterventionQueue:
    def __init__(self) -> None:
        self._tasks: list[InterventionTask] = []

    def enqueue(self, task: InterventionTask) -> None:
        self._tasks.append(task)

    def list_open_tasks(self) -> list[InterventionTask]:
        return [task for task in self._tasks if task.status == "open"]

    def mark_done(self, task_id: str) -> None:
        for task in self._tasks:
            if task.task_id == task_id:
                task.status = "done"
                return


def evaluate_intervention_need(
    *,
    session_id: str,
    source_url: str,
    page_type: str,
    low_layout_confidence_hits: int,
    failed_action_count: int,
    semantic_conflict_count: int,
    has_cross_origin: bool,
    has_iframe: bool,
    has_captcha: bool,
    captcha_action_count: int = 0,
    captcha_empty_code_count: int = 0,
    captcha_manual_empty_count: int = 0,
    captcha_fill_failed_count: int = 0,
    config: InterventionTriggerConfig | None = None,
) -> list[InterventionTask]:
    cfg = config or InterventionTriggerConfig()
    reasons: list[str] = []
    if low_layout_confidence_hits >= cfg.min_low_layout_hits:
        reasons.append("low_layout_confidence")
    if failed_action_count >= cfg.min_failed_action_count:
        reasons.append("high_action_failure_rate")
    if semantic_conflict_count >= cfg.min_semantic_conflicts:
        reasons.append("semantic_conflict")
    if cfg.force_on_complex_context and has_cross_origin and has_iframe and has_captcha:
        reasons.append("complex_cross_origin_iframe_captcha")
    if has_captcha and captcha_manual_empty_count > 0:
        reasons.append("captcha_manual_required")
    if has_captcha and (
        captcha_fill_failed_count >= 2
        or (captcha_action_count >= 3 and captcha_empty_code_count >= 2)
    ):
        reasons.append("captcha_looping")
    tasks: list[InterventionTask] = []
    for idx, reason in enumerate(reasons):
        task_id = f"manual-task:{session_id}:{idx}"
        tasks.append(
            InterventionTask(
                task_id=task_id,
                reason=reason,
                source_url=source_url,
                page_type=page_type,
                context={
                    "low_layout_confidence_hits": low_layout_confidence_hits,
                    "failed_action_count": failed_action_count,
                    "semantic_conflict_count": semantic_conflict_count,
                    "has_cross_origin": has_cross_origin,
                    "has_iframe": has_iframe,
                    "has_captcha": has_captcha,
                    "captcha_action_count": captcha_action_count,
                    "captcha_empty_code_count": captcha_empty_code_count,
                    "captcha_manual_empty_count": captcha_manual_empty_count,
                    "captcha_fill_failed_count": captcha_fill_failed_count,
                },
            )
        )
    return tasks
