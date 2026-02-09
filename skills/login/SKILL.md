---
name: login
description: Automate login flows; fill username/password, handle OAuth or form-based authentication, and verify successful login. Use when recording or testing sign-in, login, or authentication.
---

# Login skill

Use this when the user wants to record or test login, sign-in, or authentication flows.

## Steps

1. Navigate to the login page (or use start URL).
2. Locate username/email and password fields (by id, name, or placeholder).
3. Enter credentials; use placeholders like `{{username}}` and `{{password}}` when parameterizing.
4. Click the submit/sign-in button (e.g. by text "Log in", "Sign in", or selector).
5. Verify successful login (URL change, visible user menu, or success message).

## Parameterization

For reusable skills, use variables: `{{base_url}}`, `{{username}}`, `{{password}}`, `{{submit_selector}}`.
