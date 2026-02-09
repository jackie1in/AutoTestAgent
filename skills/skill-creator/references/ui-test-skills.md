# UI test skills (recording / replay)

Use this when creating or generating skills for the UI test agent: skills that guide **browser recording** or **replay** (e.g. login, form fill, search flows).

## Parameterization (required)

Skills for UI flows must use **placeholders** so they work across environments and credentials:

| Placeholder | Use for |
|-------------|--------|
| `{{base_url}}` | Start URL / site root |
| `{{username}}` | Login username or email |
| `{{password}}` | Login password |
| `{{submit_selector}}` | Button/link selector (id, data-testid, or text) |
| `{{input_username_selector}}` | Username field selector |
| `{{input_password_selector}}` | Password field selector |

Example in body:

- "Navigate to `{{base_url}}` then enter `{{username}}` and `{{password}}` in the login form; click the button matching `{{submit_selector}}`."

## Description (matching)

In this project, skills are **matched by keywords** from the user's prompt against the skill's `description`. Put clear trigger words in the frontmatter `description`:

- **Include:** action words (login, sign-in, 登录, 表单, 搜索, submit, register) and the type of flow (authentication, form, search).
- **Example:** "Login and authentication flows. Use when the user wants to record or test sign-in, login, 登录, or credential entry. Parameterized with {{base_url}}, {{username}}, {{password}}."

So when the user says "record login to example.com" or "录制登录流程", the skill is selected and its body is injected into the agent.

## Body structure (recommended)

Keep the body short and scannable:

1. **Brief purpose** – When this skill applies (e.g. "Use for login/sign-in flows.")
2. **Steps** – Ordered list: navigate → locate fields → fill (with placeholders) → submit → optional verification.
3. **Selectors** – Prefer id, name, or visible text; mention that placeholders like `{{submit_selector}}` can override.
4. **Parameterization** – List placeholders and what they replace (e.g. `{{base_url}}` = start URL).

Avoid long prose; use bullet points and short sentences so the agent can follow quickly during record or replay.
