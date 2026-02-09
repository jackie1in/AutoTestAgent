---
name: pharmacy-system-login
description: A skill for logging into the pharmacy management system (药店系统). Handles username, password, and captcha entry. Use for UI test recording or replay of login, sign-in, or 登录 flows. Parameterized with {{base_url}}, {{username}}, {{password}}.
---

# Pharmacy System Login

This skill guides the agent through the login process for a typical Chinese management system that requires a username, password, and a visual captcha. It includes logic for retrying on captcha failure.

## Parameters

This skill is parameterized to be reusable across different environments and with different credentials.

| Placeholder                 | Description                                       | Default from Recording |
| --------------------------- | ------------------------------------------------- | ---------------------- |
| `{{base_url}}`              | The starting URL for the login page.              | `http://172.20.22.202:2209/mis-ui/login.html` |
| `{{username}}`              | The login username.                               | `zjl8888`              |
| `{{password}}`              | The login password.                               | `zjl88888`             |
| `{{input_username_selector}}` | The CSS selector for the username input field.    | `#username`            |
| `{{input_password_selector}}` | The CSS selector for the password input field.    | `#password`            |
| `{{input_captcha_selector}}`  | The CSS selector for the captcha input field.     | `#checkCode`           |
| `{{submit_selector}}`       | The CSS selector for the login submission button. | `#sbbtn`               |

## Login Workflow

Follow these steps to perform the login.

1.  **Navigate**: Go to the login page at `{{base_url}}`.
2.  **Enter Username**: Locate the username field using the selector `{{input_username_selector}}` and type the `{{username}}`.
3.  **Enter Password**: Locate the password field using the selector `{{input_password_selector}}` and type the `{{password}}`.
4.  **Handle Captcha**:
    *   Visually locate the captcha image on the page. It is typically next to the captcha input field.
    *   Use visual recognition capabilities to read the characters from the image.
    *   Locate the captcha input field using the selector `{{input_captcha_selector}}` and type the recognized characters.
5.  **Submit**: Click the login button identified by the selector `{{submit_selector}}`.

## Error Handling and Verification

*   **Verification**: After submitting, wait for the page to load and confirm that the text "已成功登录系统。" is visible to verify a successful login.
*   **Captcha Retry**: If a captcha error is detected after submission, repeat the workflow from **Step 4 (Handle Captcha)**. Attempt this a maximum of three times before considering the login to have failed.