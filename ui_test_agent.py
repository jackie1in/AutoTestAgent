"""
UI Automation Testing Agent based on Browser-Use

Features:
- Record user interactions via AI agent
- Save test cases to JSON files
- Replay recorded test cases with auto-correction
- Manage test cases (list, view, delete)
- Support custom LLM providers (OpenRouter, etc.)
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserProfile, ChatOpenAI

from local_skills import (
    load_skills_dir,
    build_skill_tree,
    match_skills,
    match_skills_with_llm,
    build_extend_system_message,
    run_skill_creator,
    run_skill_creator_from_record,
    SkillTreeNode,
)

load_dotenv()

# URL pattern for extracting from prompt (http/https)
_URL_PATTERN = re.compile(r"https?://[^\s\]\)\"\'<>]+", re.IGNORECASE)


def extract_url_from_prompt(prompt: str) -> Optional[str]:
    """Extract the first http(s) URL from a prompt string. Returns None if none found."""
    if not prompt or not prompt.strip():
        return None
    m = _URL_PATTERN.search(prompt)
    return m.group(0).rstrip(".,;:") if m else None


async def infer_url_from_task(task: str, llm_config: "LLMConfig") -> Optional[str]:
    """
    Use LLM to infer a start URL from the task description (e.g. "在百度搜索" -> https://www.baidu.com).
    Returns the first http(s) URL found in the response, or None.
    """
    if not task or not task.strip():
        return None
    system = (
        "You are a helper. Given a short task description for a browser automation, "
        "reply with the single most likely start URL the user wants to open, or reply 'none' if unclear. "
        "Reply with only the URL (e.g. https://www.example.com) or the word none, no other text."
    )
    user = f"Task: {task}"
    llm = llm_config.create_llm()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            response = await asyncio.to_thread(llm.invoke, messages)
    except Exception:
        return None
    text = (getattr(response, "content", None) or str(response)).strip().lower()
    if "none" in text and not _URL_PATTERN.search(text):
        return None
    m = _URL_PATTERN.search(text)
    return m.group(0).rstrip(".,;:") if m else None


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables"""
        return cls(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        )
    
    @classmethod
    def openrouter(
        cls,
        model: str = "google/gemini-2.5-pro-preview",
        api_key: Optional[str] = None,
    ) -> "LLMConfig":
        """Create OpenRouter configuration"""
        return cls(
            model=model,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    
    def create_llm(self) -> Any:
        """Create LLM instance based on configuration"""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        return ChatOpenAI(**kwargs)


class ActionType(str, Enum):
    """Supported action types for recording"""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    SEND_KEYS = "send_keys"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    REFRESH = "refresh"
    SELECT = "select"
    UPLOAD = "upload"
    UNKNOWN = "unknown"


@dataclass
class RecordedAction:
    """Represents a single recorded action"""
    action_type: str
    action_name: str
    parameters: dict
    timestamp: str
    step_number: int
    url: Optional[str] = None
    screenshot_path: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    # AI thinking process for replay correction
    thinking: Optional[str] = None
    goal: Optional[str] = None
    memory: Optional[str] = None
    element_description: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RecordedAction":
        # Handle legacy data without new fields
        for field_name in ["thinking", "goal", "memory", "element_description"]:
            if field_name not in data:
                data[field_name] = None
        return cls(**data)


@dataclass
class ReplayResult:
    """Result of a replay step"""
    step_number: int
    success: bool
    original_action: dict
    executed_action: Optional[dict] = None
    error: Optional[str] = None
    correction_attempted: bool = False
    correction_success: bool = False
    thinking: Optional[str] = None


@dataclass 
class TestCase:
    """Represents a complete test case"""
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    start_url: str
    actions: list[RecordedAction] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["actions"] = [a.to_dict() if isinstance(a, RecordedAction) else a for a in self.actions]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        actions = [RecordedAction.from_dict(a) if isinstance(a, dict) else a for a in data.get("actions", [])]
        data["actions"] = actions
        return cls(**data)
    
    def save(self, directory: str = "test_cases") -> str:
        """Save test case to JSON file"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(directory, f"{self.id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "TestCase":
        """Load test case from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class TestCaseManager:
    """Manages test case storage and retrieval"""
    
    def __init__(self, directory: str = "test_cases"):
        self.directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    def list_test_cases(self) -> list[TestCase]:
        """List all saved test cases"""
        test_cases = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                filepath = os.path.join(self.directory, filename)
                try:
                    test_cases.append(TestCase.load(filepath))
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
        return sorted(test_cases, key=lambda x: x.created_at, reverse=True)
    
    def get_test_case(self, test_id: str) -> Optional[TestCase]:
        """Get a specific test case by ID"""
        filepath = os.path.join(self.directory, f"{test_id}.json")
        if os.path.exists(filepath):
            return TestCase.load(filepath)
        return None
    
    def delete_test_case(self, test_id: str) -> bool:
        """Delete a test case by ID"""
        filepath = os.path.join(self.directory, f"{test_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def search_test_cases(self, query: str) -> list[TestCase]:
        """Search test cases by name or description"""
        query = query.lower()
        return [
            tc for tc in self.list_test_cases()
            if query in tc.name.lower() or query in tc.description.lower()
        ]


class UITestRecorder:
    """Records UI interactions using Browser-Use agent"""
    
    def __init__(
        self,
        headless: bool = False,
        test_cases_dir: str = "test_cases",
        llm_config: Optional[LLMConfig] = None,
        skills_dir: str = "skills",
        auto_skills: bool = True,
    ):
        self.headless = headless
        self.test_cases_dir = test_cases_dir
        self.llm_config = llm_config or LLMConfig.from_env()
        self.skills_dir = skills_dir
        self.auto_skills = auto_skills
        self.recorded_actions: list[RecordedAction] = []
        self.current_step = 0
        self.current_url = ""
        self.screenshots_dir = "screenshots"
        Path(self.screenshots_dir).mkdir(parents=True, exist_ok=True)
    
    def _parse_action(self, action) -> tuple[str, str, dict]:
        """Parse an action object to extract type, name and parameters"""
        # Map action names to our action types
        action_type_map = {
            "navigate": ActionType.NAVIGATE,
            "click": ActionType.CLICK,
            "input": ActionType.TYPE,
            "scroll": ActionType.SCROLL,
            "scroll_to_text": ActionType.SCROLL,
            "wait": ActionType.WAIT,
            "screenshot": ActionType.SCREENSHOT,
            "extract": ActionType.EXTRACT,
            "extract_content": ActionType.EXTRACT,
            "send_keys": ActionType.SEND_KEYS,
            "go_back": ActionType.GO_BACK,
            "go_forward": ActionType.GO_FORWARD,
            "refresh": ActionType.REFRESH,
            "select": ActionType.SELECT,
            "select_option": ActionType.SELECT,
            "upload": ActionType.UPLOAD,
            "upload_file": ActionType.UPLOAD,
            "done": ActionType.EXTRACT,
        }
        
        # Handle dict objects (from model_actions())
        if isinstance(action, dict):
            # Look for action keys in the dict
            for key in action_type_map.keys():
                if key in action:
                    action_name = key
                    action_params = action[key] if isinstance(action[key], dict) else {}
                    # Include interacted_element info if available
                    if "interacted_element" in action and action["interacted_element"]:
                        action_params["_element"] = action["interacted_element"]
                    action_type = action_type_map.get(action_name, ActionType.UNKNOWN).value
                    return action_type, action_name, action_params
            
            # Fallback
            for key, value in action.items():
                if key != "interacted_element" and isinstance(value, dict):
                    action_name = key
                    action_type = action_type_map.get(action_name.lower(), ActionType.UNKNOWN).value
                    return action_type, action_name, value
            
            return ActionType.UNKNOWN.value, "unknown", action
        
        # Handle Pydantic model objects
        action_name = type(action).__name__
        class_type_map = {
            "NavigateToUrlEvent": ActionType.NAVIGATE,
            "ClickElementEvent": ActionType.CLICK,
            "ClickCoordinateEvent": ActionType.CLICK,
            "TypeTextEvent": ActionType.TYPE,
            "InputTextEvent": ActionType.TYPE,
            "ScrollEvent": ActionType.SCROLL,
            "ScrollToTextEvent": ActionType.SCROLL,
            "WaitEvent": ActionType.WAIT,
            "ScreenshotEvent": ActionType.SCREENSHOT,
            "ExtractContentEvent": ActionType.EXTRACT,
            "SendKeysEvent": ActionType.SEND_KEYS,
            "GoBackEvent": ActionType.GO_BACK,
            "GoForwardEvent": ActionType.GO_FORWARD,
            "RefreshEvent": ActionType.REFRESH,
            "SelectDropdownOptionEvent": ActionType.SELECT,
            "UploadFileEvent": ActionType.UPLOAD,
            "DoneEvent": ActionType.EXTRACT,
        }
        
        action_type = class_type_map.get(action_name, ActionType.UNKNOWN).value
        
        params = {}
        try:
            if hasattr(action, "model_dump"):
                params = action.model_dump()
            elif hasattr(action, "__dict__"):
                params = {k: v for k, v in action.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
        
        return action_type, action_name, params
    
    def _extract_element_description(self, params: dict) -> Optional[str]:
        """Extract human-readable element description from parameters"""
        element = params.get("_element")
        if not element:
            return None
        
        descriptions = []
        
        try:
            # Handle both dict and object types
            def get_attr(obj, key, default=""):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            
            # Get element type
            node_name = get_attr(element, "node_name", "")
            if node_name:
                descriptions.append(node_name.lower() if isinstance(node_name, str) else str(node_name))
            
            # Get attributes
            attrs = get_attr(element, "attributes", {})
            if attrs:
                if isinstance(attrs, dict):
                    if attrs.get("id"):
                        descriptions.append(f"id='{attrs['id']}'")
                    if attrs.get("name"):
                        descriptions.append(f"name='{attrs['name']}'")
                    if attrs.get("class"):
                        descriptions.append(f"class='{attrs['class']}'")
                    if attrs.get("value"):
                        descriptions.append(f"value='{attrs['value']}'")
                    if attrs.get("type"):
                        descriptions.append(f"type='{attrs['type']}'")
                else:
                    # attrs might be an object
                    for attr_name in ["id", "name", "class", "value", "type"]:
                        val = getattr(attrs, attr_name, None)
                        if val:
                            descriptions.append(f"{attr_name}='{val}'")
            
            # Get accessible name (button text, etc.)
            ax_name = get_attr(element, "ax_name", "")
            if ax_name:
                descriptions.append(f"text='{ax_name}'")
            
            # Get XPath
            xpath = get_attr(element, "x_path", "")
            if xpath:
                descriptions.append(f"xpath='{xpath}'")
        
        except Exception as e:
            # If extraction fails, try to get string representation
            try:
                return str(element)[:200]
            except:
                return None
        
        return " ".join(descriptions) if descriptions else None
    
    async def _on_step_end(self, agent: Agent):
        """Hook called at the end of each agent step to record actions with thinking"""
        try:
            history = agent.history
            if not history or len(history) == 0:
                return
            
            # Get all model actions and thoughts
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()
            
            if not model_actions:
                return
            
            # Get current URL
            current_url = ""
            try:
                urls = history.urls()
                if urls:
                    current_url = urls[-1] if urls else ""
            except Exception:
                pass
            
            # Record actions we haven't recorded yet
            for i, action in enumerate(model_actions):
                step_num = i + 1
                if step_num <= len(self.recorded_actions):
                    continue
                
                self.current_step = step_num
                action_type, action_name, params = self._parse_action(action)
                
                # Extract AI thinking from thoughts
                thinking = None
                goal = None
                memory = None
                if model_thoughts and i < len(model_thoughts):
                    thought = model_thoughts[i]
                    if thought:
                        # Extract thinking fields from AgentBrain
                        if hasattr(thought, "evaluation_previous_goal"):
                            thinking = str(thought.evaluation_previous_goal) if thought.evaluation_previous_goal else None
                        if hasattr(thought, "next_goal"):
                            goal = thought.next_goal
                        if hasattr(thought, "memory"):
                            memory = thought.memory
                
                # Extract element description
                element_desc = self._extract_element_description(params)
                
                recorded_action = RecordedAction(
                    action_type=action_type,
                    action_name=action_name,
                    parameters=params,
                    timestamp=datetime.now().isoformat(),
                    step_number=self.current_step,
                    url=current_url,
                    thinking=thinking,
                    goal=goal,
                    memory=memory,
                    element_description=element_desc,
                )
                
                self.recorded_actions.append(recorded_action)
                print(f"  [Step {self.current_step}] Recorded: {action_name}")
        
        except Exception as e:
            print(f"  Warning: Failed to record action: {e}")
    
    async def record(
        self,
        task: str,
        start_url: str = "",
        test_name: str = "",
        description: str = "",
        max_steps: int = 50,
        force_skill_id: Optional[str] = None,
    ) -> TestCase:
        """Record a test case by running an AI agent task"""
        self.recorded_actions = []
        self.current_step = 0
        
        print(f"\n{'='*60}")
        print(f"Recording Test Case")
        print(f"{'='*60}")
        print(f"Task: {task}")
        if start_url:
            print(f"Start URL: {start_url}")
        print(f"{'='*60}\n")
        
        # Create browser - use headless parameter directly
        browser = Browser(
            headless=self.headless,
        )
        
        # If start_url is set, open it via initial_actions so the browser is not blank
        initial_actions = []
        if start_url:
            initial_actions = [{"navigate": {"url": start_url, "new_tab": False}}]
        full_task = task if initial_actions else (f"First, navigate to {start_url}. Then, {task}" if start_url else task)
        
        llm = self.llm_config.create_llm()
        print(f"Using LLM: {self.llm_config.model}")
        if self.llm_config.base_url:
            print(f"Base URL: {self.llm_config.base_url}")
        
        extend_system_message = ""
        if self.auto_skills and self.skills_dir:
            skills = load_skills_dir(self.skills_dir)
            prompt_preview = (task[:80] + "…") if len(task) > 80 else task
            print(f"[Skills] Matching skills for prompt: {prompt_preview!r}")
            matched = match_skills(task, skills)
            if force_skill_id:
                for s in skills:
                    if s.id == force_skill_id:
                        if s not in matched:
                            matched.insert(0, s)
                        break
            if matched:
                extend_system_message = build_extend_system_message(matched)
                print(f"[Skills] Using skills: {[s.name for s in matched]}")
            else:
                # No keyword match: ask LLM to pick by task + skill descriptions
                try:
                    print("[Skills] No keyword match; asking LLM to select skills from descriptions...")
                    matched = await match_skills_with_llm(task, skills, self.llm_config, top_k=5)
                    if matched:
                        extend_system_message = build_extend_system_message(matched)
                        print(f"[Skills] LLM selected: {[s.name for s in matched]}")
                    else:
                        print("[Skills] No skills matched (keyword or LLM).")
                except Exception as e:
                    print(f"[Skills] LLM selection failed: {e}; no skills will be used.")
        
        agent_kw: dict = dict(
            task=full_task,
            llm=llm,
            browser=browser,
            generate_gif=True,
        )
        if extend_system_message:
            agent_kw["extend_system_message"] = extend_system_message
        if initial_actions:
            agent_kw["initial_actions"] = initial_actions
        agent = Agent(**agent_kw)
        
        try:
            print("Starting recording...")
            history = await agent.run(
                max_steps=max_steps,
                on_step_end=self._on_step_end,
            )
            
            # Create test case
            test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_case = TestCase(
                id=test_id,
                name=test_name or f"Test_{test_id}",
                description=description or task,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                start_url=start_url,
                actions=self.recorded_actions,
                metadata={
                    "task": task,
                    "max_steps": max_steps,
                    "total_duration": history.total_duration_seconds() if history else 0,
                    "is_successful": history.is_successful() if history else False,
                    "final_result": history.final_result() if history else None,
                }
            )
            
            filepath = test_case.save(self.test_cases_dir)
            
            print(f"\n{'='*60}")
            print(f"Recording Complete!")
            print(f"{'='*60}")
            print(f"Test ID: {test_case.id}")
            print(f"Actions recorded: {len(self.recorded_actions)}")
            print(f"Saved to: {filepath}")
            print(f"{'='*60}\n")
            
            return test_case
            
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Save partial recording so we can still generate skill from it
            if self.recorded_actions:
                test_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_partial"
                self._partial_test_case = TestCase(
                    id=test_id,
                    name=test_name or f"Test_{test_id}",
                    description=description or task,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    start_url=start_url,
                    actions=list(self.recorded_actions),
                    metadata={"task": task, "max_steps": max_steps, "partial": True},
                )
                self._partial_test_case.save(self.test_cases_dir)
                print(f"\n[Interrupted] Partial recording saved ({len(self.recorded_actions)} actions).")
            else:
                self._partial_test_case = None
            raise
        finally:
            await browser.stop()


class UITestPlayer:
    """Replays recorded test cases with auto-correction"""
    
    def __init__(
        self,
        headless: bool = False,
        test_cases_dir: str = "test_cases",
        llm_config: Optional[LLMConfig] = None,
        max_correction_attempts: int = 3,
    ):
        self.headless = headless
        self.test_cases_dir = test_cases_dir
        self.llm_config = llm_config or LLMConfig.from_env()
        self.manager = TestCaseManager(test_cases_dir)
        self.max_correction_attempts = max_correction_attempts
        self.replay_results: list[ReplayResult] = []
    
    def _generate_replay_task(self, test_case: TestCase, with_context: bool = True) -> str:
        """Generate a task description from recorded actions for replay"""
        steps = []
        
        for i, action in enumerate(test_case.actions, 1):
            action_desc = self._describe_action(action, with_context)
            if action_desc:
                steps.append(f"{i}. {action_desc}")
        
        if test_case.start_url:
            steps.insert(0, f"0. Navigate to {test_case.start_url}")
        
        task = f"""Replay the following test case step by step:

Test Name: {test_case.name}
Description: {test_case.description}

Steps to perform:
{chr(10).join(steps)}

IMPORTANT: 
- Execute each step carefully
- If an element's position has changed, try to find it by its attributes (id, name, class, text)
- Report success or failure for each step"""
        
        return task
    
    def _generate_correction_task(
        self,
        action: RecordedAction,
        error: str,
        current_url: str,
    ) -> str:
        """Generate a task to correct a failed action"""
        task = f"""The previous action failed. Please try to complete the same goal using alternative methods.

FAILED ACTION:
- Type: {action.action_type}
- Name: {action.action_name}
- Original Parameters: {json.dumps(action.parameters, ensure_ascii=False)}
- Element Description: {action.element_description or 'N/A'}

ORIGINAL GOAL: {action.goal or 'Complete the action'}

ORIGINAL THINKING: {action.thinking or 'N/A'}

ERROR: {error}

CURRENT URL: {current_url}

INSTRUCTIONS:
1. The element position may have changed but the functionality should be the same
2. Try to find the element by:
   - Text content / label
   - ID or name attribute
   - Class name
   - XPath pattern
   - Visual position relative to other elements
3. If it's a click action, find and click the equivalent button/link
4. If it's an input action, find and fill the equivalent input field
5. Report whether the correction was successful"""
        
        return task
    
    def _describe_action(self, action: RecordedAction, with_context: bool = True) -> str:
        """Generate a human-readable description of an action"""
        params = action.parameters
        base_desc = ""
        
        if action.action_type == ActionType.NAVIGATE.value:
            url = params.get("url", "")
            base_desc = f"Navigate to {url}"
        
        elif action.action_type == ActionType.CLICK.value:
            if action.element_description:
                base_desc = f"Click on element: {action.element_description}"
            else:
                index = params.get("index", "")
                base_desc = f"Click on element (index {index})" if index else "Click on element"
        
        elif action.action_type == ActionType.TYPE.value:
            text = params.get("text", "")
            if action.element_description:
                base_desc = f"Type '{text}' into: {action.element_description}"
            else:
                base_desc = f"Type '{text}' into input field"
        
        elif action.action_type == ActionType.SCROLL.value:
            direction = params.get("direction", "down")
            base_desc = f"Scroll {direction}"
        
        elif action.action_type == ActionType.SEND_KEYS.value:
            keys = params.get("keys", "")
            base_desc = f"Send keys: {keys}"
        
        elif action.action_type == ActionType.SELECT.value:
            option = params.get("option", params.get("value", ""))
            base_desc = f"Select option '{option}'"
        
        elif action.action_type == ActionType.WAIT.value:
            seconds = params.get("seconds", 1)
            base_desc = f"Wait for {seconds} seconds"
        
        elif action.action_type == ActionType.EXTRACT.value:
            base_desc = "Extract/verify content"
        
        else:
            base_desc = f"{action.action_name}: {params}"
        
        # Add context from AI thinking if available
        if with_context and action.goal:
            base_desc += f" (Goal: {action.goal})"
        
        return base_desc
    
    async def replay(
        self,
        test_id: str,
        max_steps: int = 100,
        auto_correct: bool = True,
    ) -> dict:
        """Replay a recorded test case with auto-correction"""
        test_case = self.manager.get_test_case(test_id)
        if not test_case:
            raise ValueError(f"Test case not found: {test_id}")
        
        print(f"\n{'='*60}")
        print(f"Replaying Test Case")
        print(f"{'='*60}")
        print(f"Test ID: {test_case.id}")
        print(f"Name: {test_case.name}")
        print(f"Description: {test_case.description}")
        print(f"Actions to replay: {len(test_case.actions)}")
        print(f"Auto-correction: {'Enabled' if auto_correct else 'Disabled'}")
        print(f"{'='*60}\n")
        
        self.replay_results = []
        
        # Create browser
        browser = Browser(headless=self.headless)
        llm = self.llm_config.create_llm()
        print(f"Using LLM: {self.llm_config.model}")
        
        results = {
            "test_id": test_id,
            "test_name": test_case.name,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "success": False,
            "steps_executed": 0,
            "steps_corrected": 0,
            "errors": [],
            "corrections": [],
            "final_result": None,
        }
        
        try:
            # Generate replay task
            replay_task = self._generate_replay_task(test_case)
            
            # If test case has start_url, open it first so the browser is not blank
            initial_actions = []
            if test_case.start_url:
                initial_actions = [{"navigate": {"url": test_case.start_url, "new_tab": False}}]
            
            agent_kw: dict = dict(
                task=replay_task,
                llm=llm,
                browser=browser,
                generate_gif=True,
            )
            if initial_actions:
                agent_kw["initial_actions"] = initial_actions
            agent = Agent(**agent_kw)
            
            print("Starting replay...")
            
            # Track correction attempts
            correction_log = []
            
            async def on_step_end_with_correction(agent: Agent):
                """Monitor steps and attempt corrections if needed"""
                nonlocal correction_log
                
                history = agent.history
                if not history:
                    return
                
                # Check for errors in the last step
                errors = history.errors()
                if errors and errors[-1]:
                    error_msg = str(errors[-1])
                    step_num = len(errors)
                    
                    print(f"  [Step {step_num}] Error detected: {error_msg[:100]}...")
                    
                    if auto_correct and step_num <= len(test_case.actions):
                        # Get the original action that failed
                        original_action = test_case.actions[step_num - 1]
                        
                        # Log the correction attempt
                        correction_info = {
                            "step": step_num,
                            "original_action": original_action.action_name,
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat(),
                            "thinking": original_action.thinking,
                            "goal": original_action.goal,
                        }
                        correction_log.append(correction_info)
                        print(f"  [Step {step_num}] Correction info logged")
            
            history = await agent.run(
                max_steps=max_steps,
                on_step_end=on_step_end_with_correction,
            )
            
            results["completed_at"] = datetime.now().isoformat()
            results["success"] = history.is_successful() if history else False
            results["steps_executed"] = history.number_of_steps() if history else 0
            results["final_result"] = history.final_result() if history else None
            results["errors"] = [str(e) for e in (history.errors() if history else []) if e]
            results["corrections"] = correction_log
            results["steps_corrected"] = len([c for c in correction_log if "success" in str(c).lower()])
            
            # If replay failed, save failure report
            if not results["success"] and correction_log:
                self._save_failure_report(test_case, results, correction_log)
            
            print(f"\n{'='*60}")
            print(f"Replay {'Complete' if results['success'] else 'Failed'}!")
            print(f"{'='*60}")
            print(f"Success: {results['success']}")
            print(f"Steps executed: {results['steps_executed']}")
            if correction_log:
                print(f"Correction attempts: {len(correction_log)}")
            if results["errors"]:
                print(f"Errors: {len(results['errors'])}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            results["completed_at"] = datetime.now().isoformat()
            results["errors"].append(str(e))
            print(f"\nReplay failed with error: {e}")
            
        finally:
            await browser.stop()
        
        return results
    
    def _save_failure_report(
        self,
        test_case: TestCase,
        results: dict,
        correction_log: list,
    ):
        """Save detailed failure report for debugging"""
        report_dir = Path("replay_reports")
        report_dir.mkdir(exist_ok=True)
        
        report = {
            "test_case_id": test_case.id,
            "test_case_name": test_case.name,
            "replay_time": results["started_at"],
            "success": results["success"],
            "errors": results["errors"],
            "correction_attempts": correction_log,
            "original_actions": [
                {
                    "step": a.step_number,
                    "type": a.action_type,
                    "name": a.action_name,
                    "goal": a.goal,
                    "thinking": a.thinking,
                    "memory": a.memory,
                    "element": a.element_description,
                    "params": a.parameters,
                }
                for a in test_case.actions
            ],
        }
        
        report_path = report_dir / f"failure_{test_case.id}_{datetime.now().strftime('%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Failure report saved to: {report_path}")
    
    async def batch_replay(
        self,
        test_ids: list[str],
        max_steps: int = 100,
        auto_correct: bool = True,
    ) -> list[dict]:
        """Replay multiple test cases"""
        results = []
        for test_id in test_ids:
            try:
                result = await self.replay(test_id, max_steps, auto_correct)
                results.append(result)
            except Exception as e:
                results.append({
                    "test_id": test_id,
                    "success": False,
                    "errors": [str(e)],
                })
        return results


async def resolve_record_inputs(record_args: dict) -> dict:
    """Resolve task, start_url, test_name from record_args (prompts/inference as needed). Returns dict for both skill-creator and record."""
    task = record_args.get("task")
    if not task:
        task = input("Enter the task description: ").strip()
    if not task:
        raise ValueError("Task is required.")
    llm_config = LLMConfig.from_env()
    start_url = record_args.get("start_url")
    if not (start_url and str(start_url).strip()):
        extracted = extract_url_from_prompt(task)
        if extracted:
            start_url = extracted
            print(f"Using URL from prompt: {start_url}")
        else:
            inferred = await infer_url_from_task(task, llm_config)
            if inferred:
                start_url = inferred
                print(f"Using URL from task (AI): {start_url}")
            elif start_url is None:
                start_url = input("Enter start URL (optional): ").strip()
            else:
                start_url = ""
    if start_url is None:
        start_url = ""
    test_name = record_args.get("test_name")
    if test_name is None:
        test_name = input("Enter test name (optional): ").strip()
    if test_name is None:
        test_name = ""
    return {"task": task, "start_url": (start_url or "").strip(), "test_name": test_name or "", "llm_config": llm_config}


async def run_skill_creator_flow(record_args: dict, resolved: Optional[dict] = None):
    """Run skill-creator to generate a parameterized skill for the record topic. If resolved is provided, use it and do not prompt."""
    if resolved:
        task = resolved["task"]
        start_url = resolved["start_url"]
        llm_config = resolved["llm_config"]
    else:
        task = record_args.get("task")
        if not task:
            task = input("Enter the record topic (task description): ").strip()
        if not task:
            print("Task is required for --skill-creator.")
            return
        start_url = record_args.get("start_url") or ""
        if start_url is None:
            start_url = input("Enter start URL (optional): ").strip() or ""
        llm_config = LLMConfig.from_env()
    skills_dir = record_args.get("skills_dir") or "skills"
    print(f"Using LLM: {llm_config.model}")
    print(f"Generating parameterized skill for: {task}")
    try:
        path = await run_skill_creator(task, start_url, skills_dir, llm_config)
        print(f"\nSkill written to: {path}")
    except Exception as e:
        print(f"Skill-creator failed: {e}")
        raise


async def do_record(
    resolved: dict,
    headless: bool = False,
    skills_dir: str = "skills",
    no_auto_skills: bool = False,
    force_skill: Optional[str] = None,
):
    """Run browser record with already-resolved task/start_url/test_name. Shared by record-only and skill-creator+record."""
    task = resolved["task"]
    start_url = resolved["start_url"]
    test_name = resolved["test_name"]
    llm_config = resolved["llm_config"]
    print("\n" + "="*60)
    print("UI Test Recorder")
    print("="*60 + "\n")
    print(f"Using LLM: {llm_config.model}")
    recorder = UITestRecorder(
        headless=headless,
        llm_config=llm_config,
        skills_dir=skills_dir or "skills",
        auto_skills=not no_auto_skills,
    )
    try:
        test_case = await recorder.record(
            task=task,
            start_url=start_url,
            test_name=test_name,
            description=task,
            force_skill_id=force_skill,
        )
        print(f"\nTest case saved with ID: {test_case.id}")
        return test_case
    except (asyncio.CancelledError, KeyboardInterrupt):
        # Return partial so caller can still run skill-creator from it
        partial = getattr(recorder, "_partial_test_case", None)
        if partial is not None:
            return partial
        raise


async def interactive_record(
    task: Optional[str] = None,
    start_url: Optional[str] = None,
    test_name: Optional[str] = None,
    headless: bool = False,
    skills_dir: str = "skills",
    no_auto_skills: bool = False,
    force_skill: Optional[str] = None,
):
    """Resolve inputs (if needed) then run record. Uses resolve_record_inputs + do_record for reuse."""
    record_args = {
        "task": task,
        "start_url": start_url,
        "test_name": test_name,
        "headless": headless,
        "skills_dir": skills_dir,
        "no_auto_skills": no_auto_skills,
        "force_skill": force_skill,
    }
    try:
        resolved = await resolve_record_inputs(record_args)
    except ValueError as e:
        print(f"{e}")
        return None
    return await do_record(
        resolved,
        headless=headless,
        skills_dir=skills_dir,
        no_auto_skills=no_auto_skills,
        force_skill=force_skill,
    )


async def interactive_replay(test_id: Optional[str] = None, headless: bool = False):
    """Interactive replay session"""
    print("\n" + "="*60)
    print("UI Test Player")
    print("="*60 + "\n")
    
    manager = TestCaseManager()
    test_cases = manager.list_test_cases()
    
    if not test_cases:
        print("No test cases found!")
        return
    
    if not test_id:
        print("Available test cases:")
        for i, tc in enumerate(test_cases, 1):
            print(f"  {i}. [{tc.id}] {tc.name} - {len(tc.actions)} actions")
        
        selection = input("\nEnter test ID or number: ").strip()
        
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(test_cases):
                test_id = test_cases[idx].id
            else:
                test_id = selection
        except ValueError:
            test_id = selection
    
    llm_config = LLMConfig.from_env()
    print(f"Using LLM: {llm_config.model}")
    
    player = UITestPlayer(headless=headless, llm_config=llm_config)
    results = await player.replay(test_id, auto_correct=True)
    
    return results


def list_skills(skills_dir: str = "skills"):
    """List all local skills (tree) and their name/description."""
    skills = load_skills_dir(skills_dir)
    if not skills:
        print(f"\nNo skills found in {skills_dir}/")
        return
    tree = build_skill_tree(skills, skills_dir)
    print(f"\n{'='*60}\nSkills ({skills_dir}/)\n{'='*60}")
    def _walk(node: SkillTreeNode, prefix: str):
        for i, child in enumerate(node.children):
            is_last = i == len(node.children) - 1
            branch = "└── " if is_last else "├── "
            skill_info = f" ({child.skill.name})" if child.skill else ""
            print(f"{prefix}{branch}{child.name}{skill_info}")
            if child.skill and child.skill.description:
                desc = (child.skill.description[:60] + "...") if len(child.skill.description) > 60 else child.skill.description
                print(f"{prefix}    {'    ' if is_last else '│   '}  {desc}")
            _walk(child, prefix + ("    " if is_last else "│   "))
    _walk(tree, "")
    print(f"\nTotal: {len(skills)} skill(s)\n")


def list_test_cases():
    """List all test cases"""
    manager = TestCaseManager()
    test_cases = manager.list_test_cases()
    
    if not test_cases:
        print("\nNo test cases found!")
        return
    
    print("\n" + "="*60)
    print("Saved Test Cases")
    print("="*60)
    
    for tc in test_cases:
        print(f"\n  ID: {tc.id}")
        print(f"  Name: {tc.name}")
        print(f"  Description: {tc.description[:50]}..." if len(tc.description) > 50 else f"  Description: {tc.description}")
        print(f"  Actions: {len(tc.actions)}")
        print(f"  Created: {tc.created_at}")
        print(f"  Start URL: {tc.start_url}")
        print("-"*40)
    
    print(f"\nTotal: {len(test_cases)} test case(s)\n")


def view_test_case(test_id: str):
    """View details of a specific test case"""
    manager = TestCaseManager()
    tc = manager.get_test_case(test_id)
    
    if not tc:
        print(f"\nTest case not found: {test_id}")
        return
    
    print("\n" + "="*60)
    print(f"Test Case: {tc.name}")
    print("="*60)
    print(f"ID: {tc.id}")
    print(f"Description: {tc.description}")
    print(f"Start URL: {tc.start_url}")
    print(f"Created: {tc.created_at}")
    print(f"Updated: {tc.updated_at}")
    
    print(f"\nActions ({len(tc.actions)}):")
    print("-"*40)
    
    for action in tc.actions:
        action_type = action.action_type
        action_name = action.action_name
        params = action.parameters
        
        # Try to extract action from nested parameters
        if action_type == "unknown" and isinstance(params, dict):
            for key in ["navigate", "click", "input", "scroll", "wait", "done", "send_keys", "extract"]:
                if key in params:
                    action_name = key
                    action_type = {
                        "navigate": "navigate", "click": "click", "input": "type",
                        "scroll": "scroll", "wait": "wait", "done": "extract",
                        "send_keys": "send_keys", "extract": "extract"
                    }.get(key, "unknown")
                    params = params[key]
                    break
        
        print(f"\n  Step {action.step_number}: [{action_type}] {action_name}")
        
        # Show parameters
        if params and isinstance(params, dict):
            display_params = {k: v for k, v in params.items() if not k.startswith("_")}
            if display_params:
                params_str = json.dumps(display_params, ensure_ascii=False)
                if len(params_str) > 80:
                    params_str = params_str[:80] + "..."
                print(f"    Parameters: {params_str}")
        
        # Show element description
        if action.element_description:
            print(f"    Element: {action.element_description[:60]}...")
        
        # Show AI thinking
        if action.goal:
            print(f"    Goal: {action.goal}")
        if action.thinking:
            thinking_preview = action.thinking[:60] + "..." if len(str(action.thinking)) > 60 else action.thinking
            print(f"    Thinking: {thinking_preview}")
        
        if action.url:
            url_display = action.url[:60] + "..." if len(action.url) > 60 else action.url
            print(f"    URL: {url_display}")
        
        if action.error:
            print(f"    Error: {action.error}")
    
    print("\n" + "="*60 + "\n")


def delete_test_case(test_id: str):
    """Delete a test case"""
    manager = TestCaseManager()
    
    if manager.delete_test_case(test_id):
        print(f"\nTest case deleted: {test_id}")
    else:
        print(f"\nTest case not found: {test_id}")


def print_help():
    """Print help information"""
    print("""
UI Automation Testing Agent
===========================

Commands:
  record [options] [prompt]  - Record a new test case
  replay [test_id]          - Replay a saved test case (with auto-correction)
  list                      - List all saved test cases
  list-skills [--skills-dir] - List local skills (tree)
  view <test_id>            - View details of a test case
  delete <test_id>          - Delete a test case
  help                      - Show this help message

Record Options:
  -p, --prompt <text>   Task description / prompt
  -u, --url <url>       Start URL
  -n, --name <name>     Test case name
  --headless            Run in headless mode (no browser window)
  --skill-creator       Generate a parameterized skill for this topic, then run record (skill first, then browser)
  --skills-dir <path>   Skills directory (default: skills)
  --no-auto-skills      Disable auto-matching of skills from prompt
  -s, --skill <id>      Force-include skill by id (e.g. login)

Examples:
  # Record with prompt
  python main.py record "在百度搜索 ai skills"
  python main.py record -p "点击登录按钮" -u "https://example.com"
  python main.py record --prompt "填写表单" --url "https://example.com" --name "表单测试"
  
  # Interactive record
  python main.py record
  
  # Generate skill for this topic, then record (browser opens after skill is created)
  python main.py record --skill-creator -p "login to example.com" -u "https://example.com"
  
  # Replay (with auto-correction)
  python main.py replay                    # Interactive selection
  python main.py replay 20240101_120000    # Replay specific test
  
  # Management
  python main.py list
  python main.py view 20240101_120000
  python main.py delete 20240101_120000

Environment Variables (.env):
  LLM_MODEL     - Model name (e.g., google/gemini-2.5-pro-preview)
  LLM_BASE_URL  - API base URL (e.g., https://openrouter.ai/api/v1)
  LLM_API_KEY   - API key for the LLM provider

Auto-Correction:
  When replay fails (e.g., element position changed), the agent will:
  1. Use recorded AI thinking to understand the original goal
  2. Try to find the element by attributes (id, name, class, text)
  3. Log correction attempts for debugging
  4. Save failure report if correction fails
""")


def parse_record_args(args: list[str]) -> dict:
    """Parse record command arguments"""
    result = {
        "task": None,
        "start_url": None,
        "test_name": None,
        "headless": False,
        "skill_creator": False,
        "skills_dir": "skills",
        "no_auto_skills": False,
        "force_skill": None,
    }
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg in ("-p", "--prompt", "-t", "--task"):
            if i + 1 < len(args):
                result["task"] = args[i + 1]
                i += 2
                continue
        elif arg in ("-u", "--url"):
            if i + 1 < len(args):
                result["start_url"] = args[i + 1]
                i += 2
                continue
        elif arg in ("-n", "--name"):
            if i + 1 < len(args):
                result["test_name"] = args[i + 1]
                i += 2
                continue
        elif arg in ("--headless",):
            result["headless"] = True
            i += 1
            continue
        elif arg in ("--skill-creator",):
            result["skill_creator"] = True
            i += 1
            continue
        elif arg in ("--skills-dir",):
            if i + 1 < len(args):
                result["skills_dir"] = args[i + 1]
                i += 2
                continue
        elif arg in ("--no-auto-skills",):
            result["no_auto_skills"] = True
            i += 1
            continue
        elif arg in ("-s", "--skill",):
            if i + 1 < len(args):
                result["force_skill"] = args[i + 1]
                i += 2
                continue
        else:
            if result["task"] is None and not arg.startswith("-"):
                result["task"] = arg
        i += 1
    
    return result


async def main_async(args: list[str]):
    """Main async entry point"""
    if not args or args[0] == "help":
        print_help()
        return
    
    command = args[0].lower()
    
    if command == "record":
        record_args = parse_record_args(args[1:])
        try:
            resolved = await resolve_record_inputs(record_args)
        except ValueError as e:
            print(f"{e}")
            return
        headless = record_args.get("headless", False)
        skills_dir = record_args.get("skills_dir") or "skills"
        no_auto_skills = record_args.get("no_auto_skills", False)
        force_skill = record_args.get("force_skill")
        do_record_kw = dict(
            resolved=resolved,
            headless=headless,
            skills_dir=skills_dir,
            no_auto_skills=no_auto_skills,
            force_skill=force_skill,
        )
        if record_args.get("skill_creator"):
            print("\n" + "="*60)
            print("Record first, then generate skill from recording")
            print("="*60 + "\n")
            test_case = await do_record(**do_record_kw)
            if test_case:
                if not test_case.actions:
                    print("\nNo actions recorded; skipping skill generation (run record until steps are captured).")
                else:
                    print(f"\n[Skill-creator] Generating skill from {len(test_case.actions)} recorded actions (LLM: {resolved['llm_config'].model})...")
                    try:
                        path = await run_skill_creator_from_record(
                            test_case.to_dict(),
                            skills_dir=skills_dir,
                            llm_config=resolved["llm_config"],
                        )
                        print(f"\nSkill (from recording) written to: {path}")
                    except Exception as e:
                        print(f"Skill-creator from record failed: {e}")
                        raise
        else:
            await do_record(**do_record_kw)
    
    elif command == "replay":
        test_id = args[1] if len(args) > 1 else None
        await interactive_replay(test_id=test_id)
    
    elif command == "list":
        list_test_cases()
    
    elif command == "list-skills":
        skills_dir = "skills"
        rest = args[1:]
        if len(rest) >= 2 and rest[0] == "--skills-dir":
            skills_dir = rest[1]
        list_skills(skills_dir)
    
    elif command == "view":
        if len(args) < 2:
            print("Usage: view <test_id>")
            return
        view_test_case(args[1])
    
    elif command == "delete":
        if len(args) < 2:
            print("Usage: delete <test_id>")
            return
        confirm = input(f"Are you sure you want to delete '{args[1]}'? (y/N): ").strip().lower()
        if confirm == "y":
            delete_test_case(args[1])
    
    else:
        print(f"Unknown command: {command}")
        print_help()


def main():
    """Main entry point"""
    import sys
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
