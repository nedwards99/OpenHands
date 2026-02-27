import os
import re
import sys
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.clarify_agent_v2.function_calling as codeact_function_calling
from openhands.agenthub.clarify_agent_v2.tools.bash import create_cmd_run_tool
from openhands.agenthub.clarify_agent_v2.tools.browser import BrowserTool
from openhands.agenthub.clarify_agent_v2.tools.clarify import ClarifyTool
from openhands.agenthub.clarify_agent_v2.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.clarify_agent_v2.tools.finish import FinishTool
from openhands.agenthub.clarify_agent_v2.tools.ipython import IPythonTool
from openhands.agenthub.clarify_agent_v2.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.clarify_agent_v2.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.clarify_agent_v2.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.clarify_agent_v2.tools.think import ThinkTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.core.message import Message, TextContent

from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.events.event import Event, EventSource
from openhands.events.observation.observation import Observation
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


class ClarifyAgentV2(Agent):
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
        #self._next_reminder_message: str | None = None
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

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
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
        # # Add ambiguity reminder
        # reminder_message = """Before selecting your next tool or drafting a reply, first reason explicitly about whether you have enough clarity. After stating that reasoning, write a line formatted as `Ambiguity assessment: <clear|ambiguous>.` If the assessment is `ambiguous`, immediately call the `clarify` tool with questions for the user. If it is `clear`, proceed normally.'
        # """
        # #Use latest_user_message to add user context

        # #if self._next_reminder_message:
        # reminder_action = MessageAction(
        #         content=reminder_message,
        #     )
        # #reminder_action._source = EventSource.SYSTEM
        # self.pending_actions.appendleft(reminder_action)
            #self._next_reminder_message = None
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

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

        # example = self.prompt_manager.get_in_context_example(tools=self.tools)
        # if example:
        #     for msg in messages:
        #         if msg.role == 'user':
        #             for content in msg.content:
        #                 if isinstance(content, TextContent):
        #                     content.text = f"{example}\n\n{content.text}"
        #                     break
        #         break

        # Add ambiguity check at each turn
        reminder_message = """Before selecting your next tool or drafting a reply, first reason explicitly about whether you have enough information. After stating that reasoning, write a line formatted as `Ambiguity assessment: <clear|ambiguous>.` If the assessment is `ambiguous`, immediately call the `clarify` tool with questions for the user. If it is `clear`, proceed normally.'
        """

        messages.append(
            Message(
                role='user',
                content=[TextContent(text=reminder_message)],
            )
        )

        return messages

    # def _build_ambiguity_check_message(
    #     self,
    #     messages: list[Message],
    #     events: list[Event],
    #     in_context_example: str | None = None,
    # ) -> str | None:
    #     """Compose the per-turn ambiguity reminder including environment context."""
    #     latest_user_text = self._extract_latest_user_text(
    #         messages, in_context_example=in_context_example
    #     )

    #     # To add summarized context of history
    #     #context_notes = self._collect_recent_context_notes(events, max_notes=3)
    #     context_notes = []

    #     if not latest_user_text and not context_notes:
    #         return None

    #     lines: list[str] = ['AMBIGUITY CHECK REMINDER:']

    #     if latest_user_text:
    #         lines.append('Latest user guidance snapshot:')
    #         lines.append(f'"""\n{latest_user_text}\n"""')

    #     if context_notes:
    #         lines.append('')
    #         lines.append('Recent environment context to review:')
    #         for note in context_notes:
    #             lines.append(f'- {note}')

    #     lines.append('')
    #     lines.append(
    #         'Before selecting your next tool or drafting a reply, first reason explicitly about whether you have enough clarity.'
    #     )
    #     lines.append(
    #         'After stating that reasoning, write a line formatted as `Ambiguity assessment: <clear|ambiguous> - <short justification derived from your reasoning>`.'
    #     )
    #     lines.append(
    #         'Base this decision on both the user guidance and the environment signals above.'
    #     )
    #     lines.append(
    #         'If the assessment is `ambiguous`, immediately call the clarify tool with focused questions '
    #         'and wait for the user response before any other action. If it is `clear`, proceed normally.'
    #     )

    #     return '\n'.join(lines)

    # def _extract_latest_user_text(
    #     self, messages: list[Message], in_context_example: str | None = None
    # ) -> str:
    #     """Return the latest user-provided text, removing in-context learning example for brevity."""
    #     for msg in reversed(messages):
    #         if msg.role != 'user':
    #             continue
    #         for content in msg.content:
    #             if isinstance(content, TextContent):
    #                 text = content.text.strip()
    #                 if text:
    #                     if in_context_example:
    #                         prefix = f'{in_context_example}\n\n'
    #                         if text.startswith(prefix):
    #                             text = text[len(prefix):]
    #                     text = self._strip_prompt_scaffolding(text)
    #                     return self._truncate_for_prompt(text, limit=1200)
    #     return ''

    # def _collect_recent_context_notes(
    #     self, events: list[Event], max_notes: int = 3
    # ) -> list[str]:
    #     """Gather a short summary of the most recent agent step and environment feedback."""
    #     if not events or max_notes <= 0:
    #         return []

    #     latest_agent_event: Event | None = None
    #     environment_events: list[Observation] = []

    #     for event in reversed(events):
    #         source = event.source
    #         if latest_agent_event is None and source == EventSource.AGENT:
    #             latest_agent_event = event
    #             continue

    #         if (
    #             isinstance(event, Observation)
    #             and source in {EventSource.ENVIRONMENT, None}
    #             and len(environment_events) < max_notes
    #         ):
    #             environment_events.append(event)

    #         if latest_agent_event and len(environment_events) >= max_notes:
    #             break

    #     notes: list[str] = []
    #     seen: set[str] = set()

    #     agent_note = self._summarize_agent_event(latest_agent_event)
    #     if agent_note and agent_note not in seen:
    #         notes.append(agent_note)
    #         seen.add(agent_note)

    #     for observation in environment_events:
    #         obs_note = self._summarize_environment_observation(observation)
    #         if obs_note and obs_note not in seen:
    #             notes.append(obs_note)
    #             seen.add(obs_note)
    #         if len(notes) >= max_notes:
    #             break

    #     return notes

    # def _summarize_agent_event(self, event: Event | None) -> str | None:
    #     """Create a single-line summary of the most recent agent-originated event."""
    #     if event is None:
    #         return None

    #     if isinstance(event, MessageAction):
    #         prefix = 'Agent question' if event.wait_for_response else 'Agent message'
    #         summary = self._truncate_for_prompt(event.content, condense=True)
    #         return f'{prefix}: {summary}'

    #     command = getattr(event, 'command', '')
    #     if isinstance(command, str) and command:
    #         summary = self._truncate_for_prompt(command, condense=True)
    #         return f'Agent command: `{summary}`'

    #     code = getattr(event, 'code', '')
    #     if isinstance(code, str) and code:
    #         summary = self._truncate_for_prompt(code, condense=True)
    #         return f'Agent code cell: {summary}'

    #     path = getattr(event, 'path', '')
    #     if isinstance(path, str) and path:
    #         summary = self._truncate_for_prompt(path, condense=True)
    #         return f'Agent action on `{summary}`'

    #     thought = getattr(event, 'thought', '')
    #     if isinstance(thought, str) and thought:
    #         summary = self._truncate_for_prompt(thought, condense=True)
    #         return f'Agent thought: {summary}'

    #     return event.__class__.__name__

    # def _summarize_environment_observation(self, observation: Observation) -> str | None:
    #     """Create a single-line summary describing a recent environment observation."""
    #     if hasattr(observation, 'command') and hasattr(observation, 'exit_code'):
    #         command = getattr(observation, 'command', '')
    #         exit_code = getattr(observation, 'exit_code', None)
    #         status = 'ok'
    #         if isinstance(exit_code, int) and exit_code != 0:
    #             status = f'exit {exit_code}'
    #         command_summary = (
    #             f' `{self._truncate_for_prompt(command, limit=80, condense=True)}`'
    #             if isinstance(command, str) and command
    #             else ''
    #         )
    #         output_summary = self._truncate_for_prompt(
    #             getattr(observation, 'content', ''), condense=True
    #         )
    #         return f'Command output{command_summary} ({status}): {output_summary}'

    #     if hasattr(observation, 'path'):
    #         path = getattr(observation, 'path', '')
    #         path_summary = self._truncate_for_prompt(path, limit=120, condense=True)
    #         content_summary = self._truncate_for_prompt(
    #             getattr(observation, 'content', ''), condense=True
    #         )
    #         return f'File `{path_summary}` observation: {content_summary}'

    #     if hasattr(observation, 'task_list'):
    #         task_list = getattr(observation, 'task_list', [])
    #         pending: list[str] = []
    #         if isinstance(task_list, list):
    #             for task in task_list:
    #                 if not isinstance(task, dict):
    #                     continue
    #                 status = str(task.get('status', '')).lower()
    #                 if status in {'done', 'complete', 'completed'}:
    #                     continue
    #                 title = task.get('title') or task.get('id') or 'unnamed task'
    #                 pending.append(str(title))
    #         if pending:
    #             preview = ', '.join(pending[:3])
    #             if len(pending) > 3:
    #                 preview += ', ...'
    #             return f'Task tracker pending items: {preview}'
    #         return 'Task tracker: all tasks currently marked complete.'

    #     if hasattr(observation, 'error_id') or observation.__class__.__name__.lower().startswith(
    #         'error'
    #     ):
    #         error_summary = self._truncate_for_prompt(
    #             getattr(observation, 'content', ''), condense=True
    #         )
    #         return f'Error reported: {error_summary}'

    #     content = getattr(observation, 'content', '')
    #     if isinstance(content, str) and content.strip():
    #         content_summary = self._truncate_for_prompt(content, condense=True)
    #         return f'{observation.__class__.__name__}: {content_summary}'

    #     return observation.__class__.__name__

    # @staticmethod
    # def _truncate_for_prompt(
    #     text: str, limit: int = 240, condense: bool = False
    # ) -> str:
    #     """Trim and optionally condense text so reminders stay compact."""
    #     if not isinstance(text, str):
    #         return ''

    #     processed = text.strip()
    #     if condense:
    #         processed = ' '.join(processed.split())

    #     if len(processed) <= limit:
    #         return processed

    #     return processed[: limit - 3].rstrip() + '...'

    # @staticmethod
    # def _strip_prompt_scaffolding(text: str) -> str:
    #     """Remove additional-info blocks and other prompt scaffolding from user text."""
    #     if not isinstance(text, str):
    #         return ''

    #     cleaned = text
    #     # Remove templated sections enclosed in angle-bracket tags (e.g., <REPOSITORY_INFO> ... </REPOSITORY_INFO>)
    #     cleaned = re.sub(
    #         r'<[A-Z0-9_]+>.*?</[A-Z0-9_]+>\s*', '', cleaned, flags=re.DOTALL
    #     )
    #     # Remove leading labels like "Additional context:" or "Extra information:"
    #     cleaned = re.sub(
    #         r'^\s*(additional|extra)\s+(context|information)\s*:?\s*',
    #         '',
    #         cleaned,
    #         flags=re.IGNORECASE,
    #     )
    #     return cleaned.strip()

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
