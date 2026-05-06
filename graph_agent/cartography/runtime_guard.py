from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class GuardSeverity(str, Enum):
    NONE = "none"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class GuardDecision:
    severity: GuardSeverity = GuardSeverity.NONE
    code: str = ""
    reason: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return self.severity == GuardSeverity.STOP


@dataclass
class RuntimeGuardConfig:
    enabled: bool = True
    max_captcha_attempts: int = 3
    max_captcha_failures: int = 2
    max_same_action_failures: int = 3
    max_blank_dom_observations: int = 3
    max_request_failures_window: int = 5
    max_5xx_window: int = 3
    max_429_window: int = 2
    network_window_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "RuntimeGuardConfig":
        def _int(name: str, default: int) -> int:
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        raw_enabled = (
            (os.getenv("CARTOGRAPHY_RUNTIME_GUARD_ENABLED") or "true").strip().lower()
        )
        return cls(
            enabled=raw_enabled not in {"0", "false", "no", "off"},
            max_captcha_attempts=_int("CARTOGRAPHY_GUARD_MAX_CAPTCHA_ATTEMPTS", 3),
            max_captcha_failures=_int("CARTOGRAPHY_GUARD_MAX_CAPTCHA_FAILURES", 2),
            max_same_action_failures=_int("CARTOGRAPHY_GUARD_MAX_ACTION_FAILURES", 3),
            max_blank_dom_observations=_int("CARTOGRAPHY_GUARD_MAX_BLANK_DOM", 3),
            max_request_failures_window=_int(
                "CARTOGRAPHY_GUARD_MAX_REQUEST_FAILURES", 5
            ),
            max_5xx_window=_int("CARTOGRAPHY_GUARD_MAX_5XX", 3),
            max_429_window=_int("CARTOGRAPHY_GUARD_MAX_429", 2),
            network_window_seconds=_float(
                "CARTOGRAPHY_GUARD_NETWORK_WINDOW_SECONDS", 15.0
            ),
        )


class RuntimeGuard:
    """Detect terminal or unhealthy runtime states during cartography exploration.

    The guard is intentionally generic: it is not a login guard. It watches page
    text, action results, blank DOM observations, and network events to decide
    whether the agent should stop before polluting the graph with failure states.
    """

    _terminal_text_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "invalid_credentials",
            re.compile(
                r"用户名或密码错误|账号或密码错误|账户或密码错误|密码错误|invalid credentials|invalid password",
                re.I,
            ),
        ),
        (
            "captcha_error",
            re.compile(
                r"验证码错误|验证码不正确|验证码已过期|验证码失效|captcha.*(invalid|wrong|expired)",
                re.I,
            ),
        ),
        (
            "account_blocked",
            re.compile(
                r"账号已锁定|账户已锁定|账号被冻结|账户被冻结|账号禁用|account (locked|disabled|blocked)",
                re.I,
            ),
        ),
        (
            "unauthorized",
            re.compile(
                r"无权限|未授权|没有权限|登录已过期|会话已过期|unauthorized|forbidden|access denied|permission denied|\b401\b|\b403\b",
                re.I,
            ),
        ),
        (
            "service_unavailable",
            re.compile(
                r"服务不可用|系统繁忙|服务器错误|内部服务器错误|网关错误|维护中|请求超时|service unavailable|bad gateway|gateway timeout|internal server error|temporarily unavailable|maintenance|\b500\b|\b502\b|\b503\b|\b504\b",
                re.I,
            ),
        ),
        (
            "rate_limited",
            re.compile(
                r"操作频繁|请求过于频繁|请稍后再试|访问受限|too many requests|rate limit|\b429\b",
                re.I,
            ),
        ),
        (
            "application_error",
            re.compile(
                r"application error|runtime error|whitelabel error page|页面异常|系统异常|加载失败",
                re.I,
            ),
        ),
    )

    _failure_result_pattern = re.compile(
        r"failed|failure|error|exception|timeout|not found|empty|失败|错误|异常|超时|不可用",
        re.I,
    )

    def __init__(self, config: RuntimeGuardConfig | None = None) -> None:
        self.config = config or RuntimeGuardConfig.from_env()
        self.blank_dom_count = 0
        self.captcha_attempts = 0
        self.captcha_failures = 0
        self.action_failures: dict[str, int] = {}
        self._network_events: deque[dict[str, object]] = deque(maxlen=100)

    def inspect_observation(
        self, *, dom_text: str, url: str, dom_len: int
    ) -> GuardDecision:
        if not self.config.enabled:
            return GuardDecision()

        if dom_len < 50:
            self.blank_dom_count += 1
            if self.blank_dom_count >= self.config.max_blank_dom_observations:
                return GuardDecision(
                    GuardSeverity.STOP,
                    "blank_dom_limit",
                    f"Page DOM stayed blank/minimal for {self.blank_dom_count} observations.",
                    {"url": url, "dom_len": dom_len},
                )
        else:
            self.blank_dom_count = 0

        sample = (dom_text or "")[:12000]
        for code, pattern in self._terminal_text_patterns:
            match = pattern.search(sample)
            if match:
                return GuardDecision(
                    GuardSeverity.STOP,
                    code,
                    f"Terminal page text detected: {match.group(0)}",
                    {"url": url, "match": match.group(0)},
                )
        return GuardDecision()

    def record_action_result(
        self, *, action_type: str, result_text: str
    ) -> GuardDecision:
        if not self.config.enabled:
            return GuardDecision()

        text = result_text or ""
        normalized = text.upper()

        if action_type == "solve_captcha":
            self.captcha_attempts += 1
            if "CAPTCHA_OK" not in normalized:
                self.captcha_failures += 1
            if self.captcha_failures >= self.config.max_captcha_failures:
                return GuardDecision(
                    GuardSeverity.STOP,
                    "captcha_failure_limit",
                    f"Captcha solve failed {self.captcha_failures} times.",
                    {"attempts": self.captcha_attempts, "result": text[:300]},
                )
            if self.captcha_attempts >= self.config.max_captcha_attempts:
                return GuardDecision(
                    GuardSeverity.STOP,
                    "captcha_attempt_limit",
                    f"Captcha solve attempted {self.captcha_attempts} times.",
                    {"attempts": self.captcha_attempts, "result": text[:300]},
                )

        is_failure = bool(self._failure_result_pattern.search(text))
        if is_failure:
            self.action_failures[action_type] = (
                self.action_failures.get(action_type, 0) + 1
            )
            if (
                self.action_failures[action_type]
                >= self.config.max_same_action_failures
            ):
                return GuardDecision(
                    GuardSeverity.STOP,
                    "repeated_action_failure",
                    f"Action '{action_type}' failed {self.action_failures[action_type]} times.",
                    {"action_type": action_type, "result": text[:300]},
                )
        else:
            self.action_failures[action_type] = 0

        return GuardDecision()

    def record_network_event(self, event: dict[str, object]) -> None:
        if not self.config.enabled:
            return
        item = dict(event)
        item.setdefault("ts", time.time())
        self._network_events.append(item)

    def inspect_network(self) -> GuardDecision:
        if not self.config.enabled:
            return GuardDecision()

        def _to_float(value: object, default: float) -> float:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _to_int(value: object, default: int = 0) -> int:
            try:
                return int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        now = time.time()
        window = [
            e
            for e in self._network_events
            if now - _to_float(e.get("ts"), now) <= self.config.network_window_seconds
        ]
        request_failures = [e for e in window if e.get("kind") == "requestfailed"]
        responses_5xx = [
            e
            for e in window
            if e.get("kind") == "response" and _to_int(e.get("status")) >= 500
        ]
        responses_429 = [
            e
            for e in window
            if e.get("kind") == "response" and _to_int(e.get("status")) == 429
        ]

        if len(responses_5xx) >= self.config.max_5xx_window:
            return GuardDecision(
                GuardSeverity.STOP,
                "http_5xx_burst",
                f"Detected {len(responses_5xx)} HTTP 5xx responses in recent window.",
                {"count": len(responses_5xx), "events": responses_5xx[-5:]},
            )
        if len(responses_429) >= self.config.max_429_window:
            return GuardDecision(
                GuardSeverity.STOP,
                "http_429_burst",
                f"Detected {len(responses_429)} HTTP 429 responses in recent window.",
                {"count": len(responses_429), "events": responses_429[-5:]},
            )
        if len(request_failures) >= self.config.max_request_failures_window:
            return GuardDecision(
                GuardSeverity.STOP,
                "request_failure_burst",
                f"Detected {len(request_failures)} request failures in recent window.",
                {"count": len(request_failures), "events": request_failures[-5:]},
            )
        return GuardDecision()


def guard_history_entry(step: object, decision: GuardDecision) -> dict[str, object]:
    return {
        "step": step,
        "action": "runtime_guard",
        "result": f"{decision.code}: {decision.reason}",
        "evidence": decision.evidence,
    }
