"""
Local skills loader and skill-creator flow for UI test agent.

Implements: load/parse SKILL.md (agentskills.io spec), skill tree, prompt matching,
build extend_system_message for record, and run_skill_creator for record --skill-creator.
All logic in this single file.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Frontmatter: split on --- and parse key: value (minimal, no pyyaml)
def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter between first --- and second ---. Return (frontmatter_dict, body)."""
    if not content.strip().startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    front = parts[1]
    body = parts[2].lstrip("\n")
    fm = {}
    for line in front.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip("'\"").strip('"')
    return fm, body


@dataclass
class LocalSkill:
    """One skill from a directory containing SKILL.md (agentskills.io spec)."""
    id: str  # folder path from skills root, e.g. "login", "auth/sso"
    name: str
    description: str
    content: str  # full markdown (body)
    file_path: str


@dataclass
class SkillTreeNode:
    """Node in skill tree for display."""
    name: str
    path: str
    skill: Optional[LocalSkill] = None
    children: list["SkillTreeNode"] = field(default_factory=list)


def parse_skill_md(file_path: Path, skill_id: str) -> Optional[LocalSkill]:
    """Parse a SKILL.md file. Return LocalSkill or None if invalid."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return None
    fm, body = _parse_frontmatter(text)
    name = fm.get("name", "").strip()
    description = fm.get("description", "").strip()
    if not name or not description:
        return None
    return LocalSkill(
        id=skill_id,
        name=name,
        description=description,
        content=body.strip(),
        file_path=str(file_path),
    )


def load_skills_dir(skills_dir: str) -> list[LocalSkill]:
    """Scan skills_dir for directories containing SKILL.md; parse and return list of LocalSkill."""
    root = Path(skills_dir).resolve()
    if not root.is_dir():
        return []
    skills: list[LocalSkill] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        skill_md = d / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            rel = d.relative_to(root)
            skill_id = str(rel).replace("\\", "/")
        except ValueError:
            continue
        s = parse_skill_md(skill_md, skill_id)
        if s:
            skills.append(s)
    return skills


def build_skill_tree(skills: list[LocalSkill], skills_dir: str) -> SkillTreeNode:
    """Build a tree from flat skill list. Root has path '', children by first path segment."""
    root = SkillTreeNode(name="", path="", children=[])
    by_path: dict[str, SkillTreeNode] = {"": root}
    for s in skills:
        parts = s.id.split("/")
        for i in range(len(parts)):
            p = "/".join(parts[: i + 1])
            if p not in by_path:
                parent_path = "/".join(parts[:i]) if i else ""
                parent = by_path.get(parent_path, root)
                node = SkillTreeNode(name=parts[i], path=p, children=[])
                by_path[p] = node
                parent.children.append(node)
        node = by_path[s.id]
        node.skill = s
    return root


def _text_tokens(text: str) -> set[str]:
    """Tokenize for matching: ASCII words (a-z0-9) + CJK bigrams so 登录/药店 match."""
    if not text:
        return set()
    text_lower = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", text_lower))
    # CJK: add 2-char runs so "登录系统" -> {"登录","录系","系统"}, "登录" -> {"登录"}
    for m in re.finditer(r"[\u4e00-\u9fff]+", text):
        s = m.group()
        for i in range(len(s) - 1):
            tokens.add(s[i : i + 2])
    return tokens


def match_skills(prompt: str, skills: list[LocalSkill], top_k: int = 5) -> list[LocalSkill]:
    """Keyword match: normalize prompt and description, score by token overlap, return top_k."""
    if not prompt or not skills:
        return []
    prompt_tokens = _text_tokens(prompt)
    if not prompt_tokens:
        return []
    scored: list[tuple[float, LocalSkill]] = []
    for s in skills:
        desc_tokens = _text_tokens(s.description)
        name_tokens = _text_tokens(s.name)
        overlap = len(prompt_tokens & desc_tokens) + 0.5 * len(prompt_tokens & name_tokens)
        if overlap > 0:
            scored.append((overlap, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_k]]


def build_extend_system_message(matched_skills: list[LocalSkill]) -> str:
    """Build string to pass to Agent(extend_system_message=...)."""
    if not matched_skills:
        return ""
    parts = []
    for s in matched_skills:
        parts.append(f"## Skill: {s.name}\n{s.description}\n\n{s.content}")
    return "\n\n---\n\n".join(parts)


async def match_skills_with_llm(
    prompt: str,
    skills: list[LocalSkill],
    llm_config: Any,
    top_k: int = 5,
) -> list[LocalSkill]:
    """
    When keyword match finds nothing, ask the LLM to pick skills by task + skill descriptions.
    Returns the list of skills the LLM selected (by id), in order, up to top_k.
    """
    if not prompt or not skills:
        return []
    skills_list = "\n".join(f"- {s.id}: {s.description}" for s in skills)
    system = (
        "You select which skills apply to a user's task. Reply with only the skill id(s) to use, "
        "comma-separated (e.g. login,pharmacy-system-login), or exactly 'none' if no skill fits. "
        "Pick at most 5. Use the skill's id (the part before the colon), nothing else."
    )
    user = (
        f"User task:\n{prompt}\n\nAvailable skills (id: description):\n{skills_list}\n\n"
        "Which skill id(s) to use? Reply with comma-separated ids or 'none'."
    )
    combined = f"{system}\n\n---\n\n{user}"
    text = None
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=combined)]
    except ImportError:
        messages = [{"role": "user", "content": combined}]
    llm = llm_config.create_llm()
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = await asyncio.to_thread(llm.invoke, messages)
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        if "Unknown message type" in str(e) and getattr(llm_config, "base_url", None) and getattr(llm_config, "api_key", None):
            text = await _llm_chat_http(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
                system=system,
                user=user,
            )
    if not text:
        return []
    raw = text.strip().lower().replace(" ", "")
    if "none" in raw or not raw:
        return []
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    by_id = {s.id: s for s in skills}
    result: list[LocalSkill] = []
    for sid in ids[:top_k]:
        if sid in by_id and by_id[sid] not in result:
            result.append(by_id[sid])
    return result


# --- Skill-creator flow ---

SKILL_CREATOR_ID = "skill-creator"
AGENTSKILLS_SPEC_SUMMARY = """
Agent Skills format (agentskills.io): SKILL.md must have YAML frontmatter with required fields:
- name: 1-64 chars, lowercase letters, numbers, hyphens only; must match directory name.
- description: 1-1024 chars; describe what the skill does and when to use it (used for matching).
Optional: license, compatibility, metadata, allowed-tools.
Body: Markdown instructions. Optional dirs: scripts/, references/, assets/.
"""


def _load_skill_content(skill_dir: Path) -> str:
    """Load full content of a skill: SKILL.md + optional references (for context)."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return ""
    parts = [skill_md.read_text(encoding="utf-8")]
    refs = skill_dir / "references"
    if refs.is_dir():
        for f in sorted(refs.iterdir()):
            if f.suffix.lower() in (".md", ".txt"):
                try:
                    parts.append(f"\n\n--- {f.name} ---\n\n{f.read_text(encoding='utf-8')}")
                except Exception:
                    pass
    return "\n".join(parts)


def _slug_from_task(task: str) -> str:
    """Derive a kebab-case skill name from task (fallback)."""
    s = re.sub(r"[^a-z0-9\s-]", "", task.lower())
    s = re.sub(r"\s+", "-", s).strip("-")[:64]
    return s or "generated-skill"


def _parse_skill_creator_response(response: str, fallback_name: str) -> dict[str, Any]:
    """
    Parse LLM response to extract skill name, frontmatter, body, and optional files.
    Expects either: (1) a full SKILL.md in a markdown code block, or (2) structured sections.
    Returns dict: name, frontmatter_str, body, extra_files (list of {path, content}).
    """
    name = fallback_name
    frontmatter_str = ""
    body = ""
    extra_files: list[dict[str, str]] = []

    # Try to find ```markdown or ``` SKILL.md block
    block_pat = re.compile(r"```(?:markdown|md)?\s*\n(.*?)```", re.DOTALL)
    blocks = block_pat.findall(response)
    if blocks:
        full_content = blocks[0].strip()
        if "---" in full_content:
            fm, body = _parse_frontmatter(full_content)
            name = fm.get("name", fallback_name).strip()
            # Rebuild frontmatter for writing
            frontmatter_str = f"---\nname: {name}\ndescription: {fm.get('description', '')}\n---"
        else:
            body = full_content
    else:
        # No block: use whole response as body, derive name from first line or fallback
        body = response.strip()
        first_line = body.split("\n")[0].strip()
        if first_line.startswith("#"):
            name = re.sub(r"[^a-z0-9-]", "", first_line.replace(" ", "-").lower())[:64] or fallback_name

    # Normalize name to agentskills spec
    name = re.sub(r"[^a-z0-9-]", "", name.lower()).strip("-") or fallback_name
    name = name[:64]

    return {
        "name": name,
        "frontmatter_str": frontmatter_str or f"---\nname: {name}\ndescription: (Generated skill)\n---",
        "body": body,
        "extra_files": extra_files,
    }


async def _llm_chat_http(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
) -> str:
    """Call OpenAI-compatible chat API via HTTP. Returns assistant content."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })

    def _do_request():
        with urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out

    resp = await asyncio.to_thread(_do_request)
    choice = (resp.get("choices") or [None])[0]
    if not choice:
        raise RuntimeError("No choices in LLM response")
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def _format_recorded_actions_for_prompt(test_case_dict: dict) -> str:
    """Format recorded test case (id, name, description, start_url, actions) for skill-creator prompt."""
    lines = []
    lines.append(f"Test case name: {test_case_dict.get('name', '')}")
    lines.append(f"Description: {test_case_dict.get('description', '')}")
    lines.append(f"Start URL: {test_case_dict.get('start_url', '') or '(none)'}")
    lines.append("")
    actions = test_case_dict.get("actions") or []
    for i, a in enumerate(actions):
        if isinstance(a, dict):
            step = a.get("step_number", i + 1)
            atype = a.get("action_type", "")
            aname = a.get("action_name", "")
            elem = a.get("element_description") or ""
            goal = a.get("goal") or ""
            params = a.get("parameters") or {}
            # Omit internal or huge keys for cleaner prompt
            params_clean = {k: v for k, v in params.items() if k != "_element" and not (isinstance(v, str) and len(v) > 200)}
            lines.append(f"Step {step}: [{atype}] {aname}")
            if elem:
                lines.append(f"  element: {elem}")
            if goal:
                lines.append(f"  goal: {goal}")
            if params_clean:
                lines.append(f"  parameters: {json.dumps(params_clean, ensure_ascii=False)}")
            lines.append("")
    return "\n".join(lines).strip()


async def run_skill_creator_from_record(
    test_case_dict: dict,
    skills_dir: str,
    llm_config: Any,
) -> str:
    """
    Generate a parameterized skill from a **recorded** test case (actions captured during record).
    Uses the same skill-creator content and LLM flow as run_skill_creator, but the prompt is
    grounded in the actual recorded actions (types, element descriptions, goals, parameters).
    """
    root = Path(skills_dir).resolve()
    creator_dir = root / SKILL_CREATOR_ID
    if not creator_dir.is_dir() or not (creator_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill-creator not found at {creator_dir}")

    skill_creator_content = _load_skill_content(creator_dir)
    system = (
        skill_creator_content
        + "\n\n"
        + AGENTSKILLS_SPEC_SUMMARY
        + "\n\nOutput the complete SKILL.md content inside a markdown code block (```markdown ... ```). "
        + "Use placeholders like {{base_url}}, {{username}}, {{password}}, {{button_selector}} so the skill is reusable."
    )
    recorded_summary = _format_recorded_actions_for_prompt(test_case_dict)
    user_prompt = (
        "Create a **parameterized** skill from the following **recorded** UI test case. "
        "The data below is the actual recording (actions, element descriptions, goals) captured during a record run.\n\n"
        "Recorded test case:\n"
        "---\n"
        f"{recorded_summary}\n"
        "---\n\n"
        "Follow the skill-creator guidance and the agentskills.io specification. "
        "Base the skill on the real actions above: use their types, element descriptions, and flow. "
        "Parameterize URLs, credentials, selectors, and any site-specific values "
        "(e.g. {{base_url}}, {{username}}, {{password}}, {{button_selector}}). "
        "Output the full SKILL.md inside a markdown code block."
    )

    combined = f"{system}\n\n---\n\n{user_prompt}"
    text = None
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=combined)]
    except ImportError:
        messages = [{"role": "user", "content": combined}]
    llm = llm_config.create_llm()
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = await asyncio.to_thread(llm.invoke, messages)
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        if "Unknown message type" in str(e) and getattr(llm_config, "base_url", None) and getattr(llm_config, "api_key", None):
            text = await _llm_chat_http(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
                system=system,
                user=user_prompt,
            )
        if text is None:
            raise

    task = test_case_dict.get("description") or test_case_dict.get("name") or "recorded"
    fallback_name = _slug_from_task(task)
    parsed = _parse_skill_creator_response(text, fallback_name)
    name = parsed["name"]
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    skill_md_path = out_dir / "SKILL.md"
    full_content = parsed["frontmatter_str"] + "\n\n" + parsed["body"]
    skill_md_path.write_text(full_content, encoding="utf-8")

    for f in parsed.get("extra_files", []):
        p = out_dir / f.get("path", "")
        if p and p.resolve().is_relative_to(out_dir.resolve()):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f.get("content", ""), encoding="utf-8")

    return str(out_dir)


async def run_skill_creator(
    task: str,
    start_url: str,
    skills_dir: str,
    llm_config: Any,
) -> str:
    """
    Load skill-creator from skills_dir/skill-creator, call LLM to generate a parameterized skill
    for the record topic (task + start_url). Write result to skills/<name>/ and return skill path.
    """
    root = Path(skills_dir).resolve()
    creator_dir = root / SKILL_CREATOR_ID
    if not creator_dir.is_dir() or not (creator_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill-creator not found at {creator_dir}")

    skill_creator_content = _load_skill_content(creator_dir)
    system = (
        skill_creator_content
        + "\n\n"
        + AGENTSKILLS_SPEC_SUMMARY
        + "\n\nOutput the complete SKILL.md content inside a markdown code block (```markdown ... ```). "
        + "Use placeholders like {{base_url}}, {{username}}, {{password}}, {{button_selector}} so the skill is reusable."
    )
    user_prompt = (
        f"Create a **parameterized** skill for the following UI record topic.\n\n"
        f"Record task: {task}\n"
        f"Start URL: {start_url or '(none)'}\n\n"
        "Follow the skill-creator guidance and the agentskills.io specification. "
        "The skill must be parameterized: use placeholders for URLs, credentials, selectors, and any site-specific values "
        "(e.g. {{base_url}}, {{username}}, {{password}}, {{button_selector}}) so the skill can be reused across environments. "
        "Output the full SKILL.md inside a markdown code block."
    )

    llm = llm_config.create_llm()
    # Some backends (e.g. OpenRouter/Gemini) accept only HumanMessage, not SystemMessage or dict
    combined = f"{system}\n\n---\n\n{user_prompt}"
    text = None
    # Try LangChain LLM first
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=combined)]
    except ImportError:
        messages = [{"role": "user", "content": combined}]
    try:
        llm = llm_config.create_llm()
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = await asyncio.to_thread(llm.invoke, messages)
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        if "Unknown message type" in str(e) and getattr(llm_config, "base_url", None) and getattr(llm_config, "api_key", None):
            # Fallback: call OpenAI-compatible API directly via HTTP
            text = await _llm_chat_http(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
                system=system,
                user=user_prompt,
            )
        if text is None:
            raise

    fallback_name = _slug_from_task(task) or "generated-skill"
    parsed = _parse_skill_creator_response(text, fallback_name)
    name = parsed["name"]
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    skill_md_path = out_dir / "SKILL.md"
    full_content = parsed["frontmatter_str"] + "\n\n" + parsed["body"]
    skill_md_path.write_text(full_content, encoding="utf-8")

    for f in parsed.get("extra_files", []):
        p = out_dir / f.get("path", "")
        if p and p.resolve().is_relative_to(out_dir.resolve()):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f.get("content", ""), encoding="utf-8")

    return str(out_dir)
