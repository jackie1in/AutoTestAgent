from graph_agent.cartography.captcha.config_dom import (
    CaptchaRecognitionResult,
    normalize_manual_captcha_code,
    resolve_captcha_solve_mode,
    _extract_form_ancestor_xpath,
    _in_same_form,
    _score_captcha_img_node,
    _should_keep_img_candidate,
    _xpath_proximity_score,
)
from graph_agent.cartography.captcha.recognition import (
    _build_captcha_content,
    _needs_arithmetic_retry,
    _normalize_captcha_code,
    _parse_evaluate_result,
    recognize_captcha_with_candidates,
    recognize_captcha_with_fallback,
)
from graph_agent.cartography.captcha.solver import solve_captcha_from_page
