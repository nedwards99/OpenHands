from openhands.agenthub.intent_agent.intent_agent import IntentAgent, IntentLiteAgent
from openhands.controller.agent import Agent

Agent.register('IntentAgent', IntentAgent)
Agent.register('IntentLiteAgent', IntentLiteAgent)
