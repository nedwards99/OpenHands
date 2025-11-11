import os
import re
import sys
from collections import deque
from typing import TYPE_CHECKING, Any

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.clarify_agent.function_calling as codeact_function_calling
from openhands.agenthub.clarify_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.clarify_agent.tools.browser import BrowserTool
from openhands.agenthub.clarify_agent.tools.clarify import ClarifyTool
from openhands.agenthub.clarify_agent.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.clarify_agent.tools.finish import FinishTool
from openhands.agenthub.clarify_agent.tools.ipython import IPythonTool
from openhands.agenthub.clarify_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.clarify_agent.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.clarify_agent.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.clarify_agent.tools.think import ThinkTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent

from openhands.events.action import AgentFinishAction, MessageAction, Action, AgentDelegateAction, AgentThinkAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.observation import Observation
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager

REMINDER_MESSAGE = "Carefully check whether all key information is provided. If there's any ambiguity or missing details that could impact the main agent's work you should return `True` for `needs_clarification`. Only skip asking questions when you are absolutely sure all relevant information is complete."
SAFE_TYPES = (MessageAction, AgentThinkAction)

class ClarifyAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self._awaiting_intent: bool = False
        self._intent_verdict: dict[str, Any] | None = None
        self._intent_delegate_initialized: bool = False
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        # Override with router if needed
        self.llm = self.llm_registry.get_router(
            self.config, agent_name=self.name, service_id=self.service_id
        )

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self.config.resolved_system_prompt_filename,
            )

        return self._prompt_manager

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # For these models, we use short tool descriptions ( < 1024 tokens)
        # to avoid hitting the OpenAI token limit for tool descriptions.
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-4', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        if self.llm is not None:
            # For historical reasons, previously OpenAI enforces max function description length of 1k characters
            # https://community.openai.com/t/function-call-description-max-length/529902
            # But it no longer seems to be an issue recently
            # https://community.openai.com/t/was-the-character-limit-for-schema-descriptions-upgraded/1225975
            # Tested on GPT-5 and longer description still works. But we still keep the logic to be safe for older models.
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
        #TODO: add clarify flag
        tools.append(ClarifyTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing:
            if sys.platform == 'win32':
                logger.warning('Windows runtime does not support browsing yet')
            else:
                tools.append(BrowserTool)
        #TODO: for scratchpad???
        # if self.config.enable_jupyter:
        #     tools.append(IPythonTool)
        if self.config.enable_plan_mode:
            # In plan mode, we use the task_tracker tool for task management
            tools.append(create_task_tracker_tool(use_short_tool_desc))
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc,
                    runtime_type=self.config.runtime,
                )
            )
        return tools

    def reset(self) -> None:
        """Resets the CodeAct Agent's internal state."""
        super().reset()
        # Only clear pending actions, not LLM metrics
        self.pending_actions.clear()

    def step(self, state: State) -> 'Action':
        """Performs one step using the CodeAct Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        - CondensationAction(...) - condense conversation history by forgetting specified events and optionally providing a summary
        - FileReadAction(path, ...) - read file content from specified path
        - FileEditAction(path, ...) - edit file using LLM-based (deprecated) or ACI-based editing
        - AgentThinkAction(thought) - log agent's thought/reasoning process
        - CondensationRequestAction() - request condensation of conversation history
        - BrowseInteractiveAction(browser_actions) - interact with browser using specified actions
        - MCPAction(name, arguments) - interact with MCP server tools
        """
        # Continue with pending actions if any
        if self.pending_actions:
            if self._awaiting_intent and not self._intent_verdict:
                return self.pending_actions.popleft()  # still waiting, keep draining
            if self._intent_verdict and self._intent_verdict.get('needs_clarification'):
                self._prune_pending_actions_for_clarify()
                # fall through to enqueue clarify action below
            else:
                return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        try:
            initial_user_message = self._get_initial_user_message(state.history)
        except ValueError:
            initial_user_message = state.get_last_user_message()
            if initial_user_message is None:
                raise

        # Delegate to IntentAgent for ambiguity.
        # Skip delegation on the very first turn so the main agent can process the initial instructions.
        if not self._intent_delegate_initialized:
            self._intent_delegate_initialized = True
        elif not self._awaiting_intent:
            self._intent_verdict = None
            # Ask IntentAgent for a verdict
            latest_user_message = state.get_last_user_message()
            # prompt_text = (
            #     latest_user_message.content.strip()
            #     if latest_user_message and latest_user_message.content
            #     else '(no new user message)'
            # )
            self._awaiting_intent = True
            return AgentDelegateAction(
                    agent='IntentAgent',
                    # Pass latest user context and mark delegate as persistent
                    inputs={
                        'prompt': f'{REMINDER_MESSAGE}',
                    },
                )

        else:

            self._intent_verdict = self._read_intent_verdict(state.history)
            if self._intent_verdict is None:
                logger.warning('Waiting for IntentAgent verdict…')
            else:
                self._awaiting_intent = False

        logger.warning(f"Intent verdict: {self._intent_verdict}")

        if self._intent_verdict and self._intent_verdict.get('needs_clarification'):
            reason = self._intent_verdict.get('reasons', 'No reasons provided.')
            if reason:
                reminder = (
                    "Intent agent flagged open ambiguities in the current context; call the clarify tool next.\n"
                    f"Reason: {reason.strip()}"
                )
            else:
                reminder = "Intent agent flagged open ambiguities in the current context; call the clarify tool next."
            messages = self._get_messages(condensed_history, initial_user_message)
            messages.append(Message(role='system', content=[TextContent(text=reminder)]))

            params: dict = {
                'messages': self.llm.format_messages_for_llm(messages),
            }
            # Restrict tools to Clarify for this turn.
            params['tools'] = check_tools([ClarifyTool], self.llm.config)
            params['extra_body'] = {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model, agent_name=self.name
                )
            }
            response = self.llm.completion(**params)
            logger.debug(f'Response from LLM: {response}')

        else:
            if self._intent_verdict is None:
                logger.warning('IntentAgent returned no verdict; continuing normally.')

            messages = self._get_messages(condensed_history, initial_user_message)
            params: dict = {
                'messages': messages,
            }
            params['tools'] = check_tools(self.tools, self.llm.config)
            params['extra_body'] = {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model, agent_name=self.name
                )
            }
            response = self.llm.completion(**params)
            logger.debug(f'Response from LLM: {response}')

        actions = self.response_to_actions(response)
        logger.debug(f'Actions after response_to_actions: {actions}')

        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _prune_pending_actions_for_clarify(self) -> None:
        kept: deque[Action] = deque()
        while self.pending_actions:
            action = self.pending_actions.popleft()
            if isinstance(action, SAFE_TYPES):
                kept.append(action)
            # else drop FileReadAction, FileEditAction, AgentFinishAction, etc.
        self.pending_actions = kept

    def _has_real_user_message(self, history: list[Event]) -> bool:
        example = self.prompt_manager.get_in_context_example(tools=self.tools) or ''
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                text = event.content.strip()
                if text and (not example or not text.startswith(example)):
                    return True
        return False

    def _find_first_real_user_message(self, history: list[Event]) -> MessageAction | None:
        example = self.prompt_manager.get_in_context_example(tools=self.tools) or ''
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                text = event.content.strip()
                if example and text.startswith(example):
                    continue
                if text:
                    return event
        return None

    def _build_intent_context(self, state: State) -> str:
        initial = self._find_first_real_user_message(state.history).content.strip()
        if initial is None:
            initial = '(no user message found)'
        try:
            latest = state.get_last_user_message().content.strip()
        except AttributeError:
            latest = '(no user message found)'
        summary = '\n'.join(
            ev.content.strip()
            for ev in state.history
            if isinstance(ev, (MessageAction, Observation))
        ) or '(no recent events)'

        if initial == latest:
            return(
                f"User message:\n{initial or '(none)'}\n\n"
                f"Recent context:\n{summary}"
            )
        else:
            return (
                f"Initial user message:\n{initial or '(none)'}\n\n"
                f"Latest user message:\n{latest or '(none)'}\n\n"
                f"Recent context:\n{summary}"
            )

    def _summarize_event_for_intent(self, event: Event) -> str:
        if isinstance(event, MessageAction):
            return self._summarize_agent_event(event) or 'Agent message.'
        if isinstance(event, Action):
            return self._summarize_agent_event(event) or event.__class__.__name__
        if isinstance(event, Observation):
            return self._summarize_environment_observation(event) or event.__class__.__name__
        return event.__class__.__name__

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Finds the initial user message action from the full history."""
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break

        if initial_user_message is None:
            # This should not happen in a valid conversation
            logger.error(
                f'CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history.'
            )
            # Depending on desired robustness, could raise error or create a dummy action
            # and log the error
            raise ValueError(
                'Initial user message not found in history. Please report this issue.'
            )
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Checks for SystemMessageAction in events, adds one if missing (legacy support)
        2. Processes events (Actions and Observations) into messages, including SystemMessageAction
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            events: The list of events to convert to messages

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt (from SystemMessageAction)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process events (including SystemMessageAction)
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        example = self.prompt_manager.get_in_context_example(tools=self.tools)
        if example:
            for msg in messages:
                if msg.role == 'user':
                    for content in msg.content:
                        if isinstance(content, TextContent):
                            content.text = f"{example}\n\n{content.text}"
                            break
                break

        return messages

    def _read_intent_verdict(self, history: list[Event]) -> dict[str, Any] | None:
        """Scan the newest events for the delegate's result and return it as a dict."""
        intent_outputs: dict[str, Any] | None = None
        for event in reversed(history):
            if isinstance(event, AgentDelegateObservation):
                intent_outputs = event.outputs
                break

        if intent_outputs is None:
            logger.warning('Intent agent verdict not yet available; still waiting.')
            return None

        return intent_outputs

    def _summarize_agent_event(self, event: Event | None) -> str | None:
        """Create a single-line summary of the most recent agent-originated event."""
        if event is None:
            return None

        if isinstance(event, MessageAction):
            prefix = 'Agent question' if event.wait_for_response else 'Agent message'
            summary = self._truncate_for_prompt(event.content, condense=True)
            return f'{prefix}: {summary}'

        command = getattr(event, 'command', '')
        if isinstance(command, str) and command:
            summary = self._truncate_for_prompt(command, condense=True)
            return f'Agent command: `{summary}`'

        code = getattr(event, 'code', '')
        if isinstance(code, str) and code:
            summary = self._truncate_for_prompt(code, condense=True)
            return f'Agent code cell: {summary}'

        path = getattr(event, 'path', '')
        if isinstance(path, str) and path:
            summary = self._truncate_for_prompt(path, condense=True)
            return f'Agent action on `{summary}`'

        thought = getattr(event, 'thought', '')
        if isinstance(thought, str) and thought:
            summary = self._truncate_for_prompt(thought, condense=True)
            return f'Agent thought: {summary}'

        return event.__class__.__name__

    def _summarize_environment_observation(self, observation: Observation) -> str | None:
        """Create a single-line summary describing a recent environment observation."""
        if hasattr(observation, 'command') and hasattr(observation, 'exit_code'):
            command = getattr(observation, 'command', '')
            exit_code = getattr(observation, 'exit_code', None)
            status = 'ok'
            if isinstance(exit_code, int) and exit_code != 0:
                status = f'exit {exit_code}'
            command_summary = (
                f' `{self._truncate_for_prompt(command, limit=80, condense=True)}`'
                if isinstance(command, str) and command
                else ''
            )
            output_summary = self._truncate_for_prompt(
                getattr(observation, 'content', ''), condense=True
            )
            return f'Command output{command_summary} ({status}): {output_summary}'

        if hasattr(observation, 'path'):
            path = getattr(observation, 'path', '')
            path_summary = self._truncate_for_prompt(path, limit=120, condense=True)
            content_summary = self._truncate_for_prompt(
                getattr(observation, 'content', ''), condense=True
            )
            return f'File `{path_summary}` observation: {content_summary}'

        if hasattr(observation, 'task_list'):
            task_list = getattr(observation, 'task_list', [])
            pending: list[str] = []
            if isinstance(task_list, list):
                for task in task_list:
                    if not isinstance(task, dict):
                        continue
                    status = str(task.get('status', '')).lower()
                    if status in {'done', 'complete', 'completed'}:
                        continue
                    title = task.get('title') or task.get('id') or 'unnamed task'
                    pending.append(str(title))
            if pending:
                preview = ', '.join(pending[:3])
                if len(pending) > 3:
                    preview += ', ...'
                return f'Task tracker pending items: {preview}'
            return 'Task tracker: all tasks currently marked complete.'

        if hasattr(observation, 'error_id') or observation.__class__.__name__.lower().startswith(
            'error'
        ):
            error_summary = self._truncate_for_prompt(
                getattr(observation, 'content', ''), condense=True
            )
            return f'Error reported: {error_summary}'

        content = getattr(observation, 'content', '')
        if isinstance(content, str) and content.strip():
            content_summary = self._truncate_for_prompt(content, condense=True)
            return f'{observation.__class__.__name__}: {content_summary}'

        return observation.__class__.__name__

    @staticmethod
    def _truncate_for_prompt(
        text: str, limit: int = 1000, condense: bool = False
    ) -> str:
        """Trim and optionally condense text so reminders stay compact."""
        if not isinstance(text, str):
            return ''

        processed = text.strip()
        if condense:
            processed = ' '.join(processed.split())

        if len(processed) <= limit:
            return processed

        return processed[: limit - 3].rstrip() + '...'

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
