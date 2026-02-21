# Building Agents with Azure AI Foundry: A Comprehensive Technical Guide

This document is a deep-dive technical guide derived from the [foundry-samples](https://github.com/xue-cai/foundry-samples) repository. It covers writing agent code with multiple frameworks, tool integration, memory management, testing, deployment, and invocation — with links to actual sample code and hypotheses about underlying technology where details are not publicly available.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Multiple Ways of Writing Agent Code](#2-multiple-ways-of-writing-agent-code)
   - [2.1 Microsoft Agent Framework (Python)](#21-microsoft-agent-framework-python)
   - [2.2 LangGraph Agents (Python)](#22-langgraph-agents-python)
   - [2.3 Custom Framework — Raw OpenAI Tool-Calling Loop (Python)](#23-custom-framework--raw-openai-tool-calling-loop-python)
   - [2.4 C# / .NET Agents](#24-c--net-agents)
   - [2.5 Prompt Agents (No Custom Code)](#25-prompt-agents-no-custom-code)
3. [Agent Powerfulness Dimensions](#3-agent-powerfulness-dimensions)
   - [3.1 Simple Chat (Echo Agent)](#31-simple-chat-echo-agent)
   - [3.2 Agents with Tools](#32-agents-with-tools)
   - [3.3 Agents with RAG / Grounded Search](#33-agents-with-rag--grounded-search)
   - [3.4 Multi-Agent Workflows](#34-multi-agent-workflows)
   - [3.5 Human-in-the-Loop (HITL)](#35-human-in-the-loop-hitl)
   - [3.6 Agents with Memory](#36-agents-with-memory)
4. [Deep Dive: Tools — SDK, Protocols, and Auth](#4-deep-dive-tools--sdk-protocols-and-auth)
   - [4.1 Tool Types and How They Are Registered](#41-tool-types-and-how-they-are-registered)
   - [4.2 Foundry Tools Communication Protocol](#42-foundry-tools-communication-protocol)
   - [4.3 MCP (Model Context Protocol) Tools — Technical Details](#43-mcp-model-context-protocol-tools--technical-details)
   - [4.4 Authentication Between Agent Code and Tools](#44-authentication-between-agent-code-and-tools)
   - [4.5 Tool Execution Flow (End to End)](#45-tool-execution-flow-end-to-end)
5. [Deep Dive: Memory — Types, Holistic Example, and Underlying Technology](#5-deep-dive-memory--types-holistic-example-and-underlying-technology)
   - [5.1 Memory Types in Foundry Agents](#51-memory-types-in-foundry-agents)
   - [5.2 Holistic Agent Example Using Various Memories](#52-holistic-agent-example-using-various-memories)
   - [5.3 Underlying Technology Hypothesis](#53-underlying-technology-hypothesis)
6. [Build, Test, Deploy, Invoke Lifecycle](#6-build-test-deploy-invoke-lifecycle)
   - [6.1 Writing the Code](#61-writing-the-code)
   - [6.2 Testing Locally](#62-testing-locally)
   - [6.3 Deploying to Azure AI Foundry](#63-deploying-to-azure-ai-foundry)
   - [6.4 Invoking the Deployed Agent](#64-invoking-the-deployed-agent)
7. [Other Important Dimensions](#7-other-important-dimensions)
   - [7.1 Observability and Tracing](#71-observability-and-tracing)
   - [7.2 Infrastructure as Code](#72-infrastructure-as-code)
   - [7.3 Security and Network Isolation](#73-security-and-network-isolation)

---

## 1. Architecture Overview

Azure AI Foundry provides a **hosted agent platform** that lets you write agent code in any framework, containerize it, and deploy it as a managed service. The high-level architecture looks like this:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Azure AI Foundry                              │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │  Client App  │───▶│  Agent Server    │───▶│  Your Agent Code   │  │
│  │  (REST API)  │    │  (Responses API) │    │  (Container)       │  │
│  └─────────────┘    └──────────────────┘    └────────┬───────────┘  │
│                                                       │              │
│                      ┌────────────────────────────────┤              │
│                      ▼              ▼                  ▼              │
│              ┌──────────┐  ┌──────────────┐  ┌───────────────┐      │
│              │ Azure    │  │ MCP Servers  │  │ Local Python  │      │
│              │ OpenAI   │  │ (Hosted)     │  │ Tool Funcs    │      │
│              │ (LLM)    │  │              │  │               │      │
│              └──────────┘  └──────────────┘  └───────────────┘      │
│                                                                      │
│   Storage Layer: Cosmos DB / Azure Storage / AI Search (threads,     │
│                  files, vector stores, checkpoints)                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key components:**
- **Agent Server SDK** (`azure-ai-agentserver-agentframework` or `azure-ai-agentserver-langgraph`): Wraps your agent into a REST server compatible with the OpenAI Responses API protocol.
- **Agent Framework** (`agent_framework`): Microsoft's Python SDK for building agents with `BaseAgent`, `ChatAgent`, tools, context providers, and workflows.
- **Agent YAML** (`agent.yaml`): Declarative configuration for deployment — defines environment variables, model resources, and protocol bindings.
- **Dockerfile**: Standardized containerization for deployment to Azure Container Registry.

---

## 2. Multiple Ways of Writing Agent Code

### 2.1 Microsoft Agent Framework (Python)

The **Microsoft Agent Framework** is a first-party Python SDK that provides high-level abstractions for building agents. It offers two main patterns:

#### Pattern A: Extend `BaseAgent` (Full Control)

You subclass `BaseAgent` and implement `run()` and `run_stream()` methods. This gives you complete control over the agent's behavior.

📁 **Sample**: [`samples/python/hosted-agents/agent-framework/echo-agent/main.py`](../samples/python/hosted-agents/agent-framework/echo-agent/main.py)

```python
from agent_framework import (
    AgentRunResponse, AgentRunResponseUpdate, AgentThread,
    BaseAgent, ChatMessage, Role, TextContent,
)
from azure.ai.agentserver.agentframework import from_agent_framework

class EchoAgent(BaseAgent):
    async def run(self, messages=None, *, thread=None, **kwargs):
        normalized = self._normalize_messages(messages)
        last = normalized[-1]
        echo_text = f"🔊 Echo: {last.text}"
        response = ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=echo_text)])
        if thread:
            await self._notify_thread_of_new_messages(thread, normalized, response)
        return AgentRunResponse(messages=[response])

    async def run_stream(self, messages=None, *, thread=None, **kwargs):
        # Yield word-by-word for streaming
        for word in response_text.split():
            yield AgentRunResponseUpdate(contents=[TextContent(text=word)], role=Role.ASSISTANT)

def create_agent():
    return EchoAgent(name="EchoBot", echo_prefix="🔊 Echo: ")

if __name__ == "__main__":
    from_agent_framework(create_agent()).run()
```

**Key SDK classes:**
- `BaseAgent` — abstract base, implement `run()` / `run_stream()`
- `AgentThread` — conversation context management
- `ChatMessage`, `TextContent`, `Role` — message primitives
- `from_agent_framework()` — adapter that wraps the agent into a REST server on port 8088

#### Pattern B: Use `AzureOpenAIChatClient` / `AzureAIAgentClient` (Faster Setup)

For most agents, you don't need to subclass `BaseAgent`. Instead, use a chat client to create an agent declaratively:

📁 **Sample**: [`samples/python/hosted-agents/agent-framework/agent-with-local-tools/main.py`](../samples/python/hosted-agents/agent-framework/agent-with-local-tools/main.py)

```python
from agent_framework.azure import AzureAIAgentClient
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.identity.aio import DefaultAzureCredential

async def main():
    async with DefaultAzureCredential() as credential, \
               AzureAIAgentClient(
                   project_endpoint=PROJECT_ENDPOINT,
                   model_deployment_name=MODEL_DEPLOYMENT_NAME,
                   credential=credential,
               ) as client:
        agent = client.create_agent(
            name="SeattleHotelAgent",
            instructions="You are a helpful travel assistant...",
            tools=[get_available_hotels],   # Python functions as tools
        )
        server = from_agent_framework(agent)
        await server.run_async()
```

Or with `AzureOpenAIChatClient` for direct Azure OpenAI integration:

📁 **Sample**: [`samples/python/hosted-agents/agent-framework/agent-with-foundry-tools/main.py`](../samples/python/hosted-agents/agent-framework/agent-with-foundry-tools/main.py)

```python
from agent_framework.azure import AzureOpenAIChatClient
from azure.ai.agentserver.agentframework import from_agent_framework, FoundryToolsChatMiddleware

chat_client = AzureOpenAIChatClient(
    ad_token_provider=_token_provider,
    middleware=FoundryToolsChatMiddleware(tools)
)
agent = chat_client.create_agent(
    name="FoundryToolAgent",
    instructions="You are a helpful assistant with access to various tools."
)
from_agent_framework(agent).run()
```

**Underlying technology:** The `AzureOpenAIChatClient` wraps the Azure OpenAI REST API. Under the hood, it sends chat completion requests with tool definitions to the model, processes tool_calls in the response, executes tools, and feeds results back in a loop until the model produces a final text response. The `from_agent_framework()` adapter creates a FastAPI/ASGI server that exposes `/responses` endpoint compatible with the OpenAI Responses protocol.

---

### 2.2 LangGraph Agents (Python)

[LangGraph](https://github.com/langchain-ai/langgraph) is LangChain's framework for building stateful, graph-based agents. Foundry provides a **LangGraph adapter** (`azure-ai-agentserver-langgraph`) that wraps any LangGraph `CompiledGraph` into a hosted agent.

📁 **Sample**: [`samples/python/hosted-agents/langgraph/calculator-agent/main.py`](../samples/python/hosted-agents/langgraph/calculator-agent/main.py)

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from azure.ai.agentserver.langgraph import from_langgraph

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

tools = [add, multiply]
tools_by_name = {t.name: t for t in tools}

def llm_call(state: MessagesState):
    return {"messages": [llm_with_tools().invoke(state["messages"])]}

def tool_node(state: dict):
    result = []
    for tc in state["messages"][-1].tool_calls:
        observation = tools_by_name[tc["name"]].invoke(tc["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tc["id"]))
    return {"messages": result}

# Build the graph
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
agent_builder.add_edge("environment", "llm_call")
agent = agent_builder.compile()

# Host as a Foundry agent
from_langgraph(agent).run()
```

**How LangGraph differs from Agent Framework:**
| Dimension | Agent Framework | LangGraph |
|-----------|----------------|-----------|
| **State model** | `AgentThread` + `ChatMessage` list | `MessagesState` (TypedDict with `messages` key) |
| **Control flow** | Imperative (`run()` method) or `WorkflowBuilder` | Declarative graph (`StateGraph` + edges) |
| **Checkpointing** | `FileCheckpointRepository`, custom stores | `MemorySaver`, `InMemorySaver`, or custom `BaseCheckpointSaver` |
| **Tool definition** | Python functions passed to `create_agent(tools=[...])` | `@tool` decorator from `langchain_core.tools` |
| **Server adapter** | `from_agent_framework(agent)` | `from_langgraph(agent)` |
| **SDK package** | `azure-ai-agentserver-agentframework` | `azure-ai-agentserver-langgraph` |

**Underlying technology:** LangGraph compiles a `StateGraph` into a `CompiledGraph` which is essentially a state machine. Each node is a function that takes state and returns state updates. The `from_langgraph()` adapter translates OpenAI Responses protocol requests into LangGraph invocations, mapping input messages to `MessagesState`, running the graph, and converting the output back to the Responses protocol format.

---

### 2.3 Custom Framework — Raw OpenAI Tool-Calling Loop (Python)

For maximum control, you can implement the entire agent loop yourself using the raw Azure OpenAI SDK and the lower-level `FoundryCBAgent` base class.

📁 **Sample**: [`samples/python/hosted-agents/custom/system-utility-agent/main.py`](../samples/python/hosted-agents/custom/system-utility-agent/main.py)

```python
from azure.ai.agentserver.core import AgentRunContext, FoundryCBAgent
from azure.ai.agentserver.core.models import Response as OpenAIResponse, ResponseStreamEvent
from openai import AzureOpenAI

class SystemUtilityAgent(FoundryCBAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = AgentConfig()
        # Direct OpenAI client creation
        self.project_client = AIProjectClient(
            endpoint=self.cfg.project_endpoint,
            credential=DefaultAzureCredential(),
        )
        self.client = self.project_client.get_openai_client()

    async def run_agent(self, context: AgentRunContext) -> AsyncGenerator[ResponseStreamEvent, None]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(context.get_conversation_messages())

        for turn in range(self.cfg.max_turns):
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                tools=TOOLS,   # OpenAI function-calling format
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                # Execute each tool call locally
                for tc in choice.message.tool_calls:
                    result = TOOL_IMPL[tc.function.name](json.loads(tc.function.arguments))
                    messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
            else:
                # Final response - stream it back
                yield from self._stream_final_text(choice.message.content, context)
                return
```

**Underlying technology:** `FoundryCBAgent` is the lowest-level base class in the Agent Server SDK. It directly implements the Responses protocol contract. You manage the entire conversation loop yourself — calling `chat.completions.create()`, checking for `tool_calls` in the response, executing tools, appending tool results, and looping until the model returns text. This is essentially the same loop that Agent Framework's `ChatAgent` automates.

---

### 2.4 C# / .NET Agents

The same hosting model works in C#. The SDK provides `ChatClientAgent` and `UseFoundryTools()` extension methods.

📁 **Sample**: [`samples/csharp/hosted-agents/AgentFramework/AgentWithTools/Program.cs`](../samples/csharp/hosted-agents/AgentFramework/AgentWithTools/Program.cs)

```csharp
using Azure.AI.AgentServer.AgentFramework.Extensions;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Azure.AI.OpenAI;
using Azure.Identity;

var chatClient = new AzureOpenAIClient(new Uri(openAiEndpoint), credential)
    .GetChatClient(deploymentName)
    .AsIChatClient()
    .AsBuilder()
    .UseFoundryTools(
        new { type = "mcp", project_connection_id = toolConnectionId },
        new { type = "code_interpreter" }
    )
    .Build();

var agent = new ChatClientAgent(chatClient,
    name: "AgentWithTools",
    instructions: "You are a helpful assistant with access to tools...")
    .AsBuilder()
    .Build();

await agent.RunAIAgentAsync();
```

**C# also supports hosted MCP:**

📁 **Sample**: [`samples/csharp/hosted-agents/AgentFramework/AgentWithHostedMCP/Program.cs`](../samples/csharp/hosted-agents/AgentFramework/AgentWithHostedMCP/Program.cs)

```csharp
var mcpTool = new HostedMcpServerTool("microsoft_docs_search",
    projectConnectionId: toolConnectionId,
    approvalMode: ToolApprovalMode.NeverRequire);

var agent = new ChatClientAgent(chatClient,
    name: "AgentWithHostedMCP",
    instructions: "...",
    tools: [mcpTool]);
```

---

### 2.5 Prompt Agents (No Custom Code)

For the simplest agents, you don't need to write code at all. Use the `AIProjectClient` to create a **prompt agent** — an agent defined entirely by a model, instructions, and tool configuration:

📁 **Sample**: [`samples/python/hosted-agents/code-interpreter-custom/main.py`](../samples/python/hosted-agents/code-interpreter-custom/main.py)

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, MCPTool

project_client = AIProjectClient(
    endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
openai_client = project_client.get_openai_client()

tools = [MCPTool(
    server_url="https://localhost",
    server_label="python_tool",
    require_approval="never",
    allowed_tools=["launchShell", "runPythonCodeInRemoteEnvironment"],
    project_connection_id=os.environ["AZURE_AI_CONNECTION_ID"],
)]

agent = project_client.agents.create_version(
    agent_name="MyAgent",
    definition=PromptAgentDefinition(
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        instructions="You are a helpful agent that can use a Python code interpreter...",
        tools=tools,
    ),
)

# Invoke directly — no server needed
response = openai_client.responses.create(
    input="Please analyze the CSV file...",
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)
```

**Underlying technology:** Prompt agents are entirely server-side. The `AIProjectClient.agents.create_version()` call registers an agent definition in the Foundry backend. When you invoke it via `responses.create()` with an `agent` reference, the Foundry service itself runs the agent loop — you never write the loop code. This is the "no code" option. *(Hypothesis: the Foundry backend likely runs the same kind of tool-calling loop as the custom framework pattern, but server-side, with the agent definition stored in a database.)*

---

## 3. Agent Powerfulness Dimensions

### 3.1 Simple Chat (Echo Agent)

The simplest possible agent — no LLM, no tools, just echoes user input.

📁 [`samples/python/hosted-agents/agent-framework/echo-agent/main.py`](../samples/python/hosted-agents/agent-framework/echo-agent/main.py)

**Capabilities:** Text in → text out. Demonstrates the minimal contract for a hosted agent.

---

### 3.2 Agents with Tools

Tools are the core mechanism that make agents powerful. The repo demonstrates three categories:

#### A. Local Python Tools
Python functions that run inside the agent container. The LLM generates function call arguments, and the agent framework executes the function locally.

📁 [`samples/python/hosted-agents/agent-framework/agent-with-local-tools/main.py`](../samples/python/hosted-agents/agent-framework/agent-with-local-tools/main.py)

```python
def get_available_hotels(
    check_in_date: Annotated[str, "Check-in date in YYYY-MM-DD format"],
    check_out_date: Annotated[str, "Check-out date in YYYY-MM-DD format"],
    max_price: Annotated[int, "Maximum price per night in USD"] = 500,
) -> str:
    """Get available hotels in Seattle for the specified dates."""
    # ... local business logic ...

agent = client.create_agent(
    name="SeattleHotelAgent",
    tools=[get_available_hotels],  # Registered as a tool
)
```

#### B. Foundry-Hosted Tools (Web Search, Code Interpreter)
Managed tools that run as Foundry services. The agent code declares them; Foundry handles execution.

📁 [`samples/python/hosted-agents/agent-framework/agent-with-foundry-tools/main.py`](../samples/python/hosted-agents/agent-framework/agent-with-foundry-tools/main.py)

```python
tools = [
    {"type": "web_search_preview"},                                           # Bing web search
    {"type": "mcp", "project_connection_id": project_tool_connection_id},     # MCP server
]
chat_client = AzureOpenAIChatClient(
    ad_token_provider=_token_provider,
    middleware=FoundryToolsChatMiddleware(tools)   # Middleware intercepts tool calls
)
```

#### C. MCP (Model Context Protocol) Tools
Remote tool servers accessed via the MCP protocol. See [Section 4.3](#43-mcp-model-context-protocol-tools--technical-details) for deep technical details.

📁 [`samples/python/hosted-agents/langgraph/react-agent-with-foundry-tools/main.py`](../samples/python/hosted-agents/langgraph/react-agent-with-foundry-tools/main.py)

---

### 3.3 Agents with RAG / Grounded Search

#### Text Search RAG via Context Providers

📁 [`samples/python/hosted-agents/agent-framework/agent-with-text-search-rag/main.py`](../samples/python/hosted-agents/agent-framework/agent-with-text-search-rag/main.py)

The `ContextProvider` pattern injects retrieved context before the LLM call:

```python
class TextSearchContextProvider(ContextProvider):
    async def invoking(self, messages, **kwargs) -> Context:
        query = messages[-1].text.lower()
        results = []
        if "return" in query and "refund" in query:
            results.append(TextSearchResult(
                source_name="Return Policy",
                source_link="https://contoso.com/policies/returns",
                text="Customers may return any item within 30 days..."
            ))
        return Context(messages=[ChatMessage(role=Role.USER, text=json.dumps(...))])

agent = AzureOpenAIChatClient(ad_token_provider=_token_provider).create_agent(
    name="SupportSpecialist",
    instructions="Answer questions using the provided context...",
    context_providers=TextSearchContextProvider(),
)
```

**Underlying technology:** The `ContextProvider` is called *before* the LLM invocation. It acts like a middleware that can prepend, append, or modify messages before they reach the model. For real RAG, you'd replace the simulated search with Azure AI Search vector queries. *(Hypothesis: in production, you'd use the `azure-search-documents` SDK to query an Azure AI Search index, compute embeddings with Azure OpenAI, and inject top-k results as context messages.)*

#### Web Search (Bing Grounding)

📁 [`samples/python/hosted-agents/agent-framework/web-search-agent/main.py`](../samples/python/hosted-agents/agent-framework/web-search-agent/main.py)

```python
from agent_framework import ChatAgent, HostedWebSearchTool

bing_search_tool = HostedWebSearchTool(
    name="Bing Grounding Search",
    connection_id=os.environ["BING_GROUNDING_CONNECTION_ID"],
)
agent = ChatAgent(chat_client=chat_client, tools=bing_search_tool)
```

---

### 3.4 Multi-Agent Workflows

The Agent Framework supports orchestrating multiple agents in concurrent or sequential workflows.

📁 [`samples/python/hosted-agents/agent-framework/agents-in-workflow/main.py`](../samples/python/hosted-agents/agent-framework/agents-in-workflow/main.py)

```python
from agent_framework import ConcurrentBuilder

researcher = AzureOpenAIChatClient(ad_token_provider=_token_provider).create_agent(
    instructions="You're an expert market researcher...", name="researcher")
marketer = AzureOpenAIChatClient(ad_token_provider=_token_provider).create_agent(
    instructions="You're a creative marketing strategist...", name="marketer")
legal = AzureOpenAIChatClient(ad_token_provider=_token_provider).create_agent(
    instructions="You're a cautious legal reviewer...", name="legal")

workflow = ConcurrentBuilder().participants([researcher, marketer, legal]).build
from_agent_framework(workflow).run()
```

**Underlying technology:** `ConcurrentBuilder` creates a workflow that fans out the user's message to all participant agents in parallel, collects their responses, and combines them into a single response. *(Hypothesis: internally, this likely uses `asyncio.gather()` to run all agent `.run()` calls concurrently, then concatenates or merges the responses. For sequential workflows, `WorkflowBuilder` with edges defines execution order.)*

---

### 3.5 Human-in-the-Loop (HITL)

Two patterns for requiring human approval before tool execution:

#### Pattern A: Agent Framework — Thread-Based HITL

📁 [`samples/python/hosted-agents/agent-framework/human-in-the-loop/agent-with-thread-and-hitl/main.py`](../samples/python/hosted-agents/agent-framework/human-in-the-loop/agent-with-thread-and-hitl/main.py)

```python
from agent_framework import ai_function

@ai_function(approval_mode="always_require")
def add_to_calendar(event_name: str, date: str) -> str:
    """Add an event to the calendar (requires approval)."""
    return f"Added '{event_name}' to calendar on {date}"

agent = ChatAgent(
    chat_client=AzureOpenAIChatClient(ad_token_provider=_token_provider),
    name="CalendarAgent",
    tools=[add_to_calendar],
    chat_message_store_factory=CustomChatMessageStore,
)

# Persist thread state across requests
thread_repository = JsonLocalFileAgentThreadRepository(agent=agent, storage_path="./thread_storage")
from_agent_framework(agent, thread_repository=thread_repository).run_async()
```

**How it works:** The `@ai_function(approval_mode="always_require")` decorator tells the framework to pause before executing this tool. The agent returns a response asking for approval. The client must send a follow-up request with the approval, and the agent resumes execution using the persisted thread state.

#### Pattern B: LangGraph — Checkpoint-Based HITL with `interrupt()`

📁 [`samples/python/hosted-agents/langgraph/human-in-the-loop/main.py`](../samples/python/hosted-agents/langgraph/human-in-the-loop/main.py)

```python
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

def ask_human(state: MessagesState):
    last_message = state["messages"][-1]
    ask = AskHuman.model_validate(last_message.tool_calls[0]["args"])
    location = interrupt(ask.question)  # Suspends graph execution
    return {"messages": [ToolMessage(tool_call_id=..., content=location)]}

workflow = StateGraph(MessagesState)
workflow.add_node("ask_human", ask_human)
app = workflow.compile(checkpointer=InMemorySaver())
```

**Underlying technology:** LangGraph's `interrupt()` raises a special exception that serializes the graph's current state into the checkpointer. When the client sends a continuation request with the human's input, the graph is restored from the checkpoint and resumes from exactly where it left off. *(Hypothesis: the Foundry adapter translates this interrupt into a Responses protocol response that signals the client to collect human input, then on the next request, it injects the human's response and resumes the graph.)*

---

### 3.6 Agents with Memory

See [Section 5](#5-deep-dive-memory--types-holistic-example-and-underlying-technology) for a comprehensive deep dive.

---

## 4. Deep Dive: Tools — SDK, Protocols, and Auth

### 4.1 Tool Types and How They Are Registered

| Tool Type | Registration | Execution Location | SDK |
|-----------|-------------|-------------------|-----|
| **Local Python functions** | `tools=[my_function]` on `create_agent()` | Inside agent container | `agent_framework` / `langchain_core.tools` |
| **Foundry web_search_preview** | `{"type": "web_search_preview"}` in tools list | Foundry-managed service | `FoundryToolsChatMiddleware` |
| **Foundry code_interpreter** | `{"type": "code_interpreter"}` in tools list | Foundry-managed sandbox | `FoundryToolsChatMiddleware` / `use_foundry_tools()` |
| **MCP tools** | `{"type": "mcp", "project_connection_id": "..."}` | Hosted MCP server | `FoundryToolsChatMiddleware` / `MCPTool` |
| **Bing Grounding** | `HostedWebSearchTool(connection_id=...)` | Foundry-managed Bing service | `agent_framework.HostedWebSearchTool` |

### 4.2 Foundry Tools Communication Protocol

**The Foundry tools are NOT plain REST APIs you call directly.** Instead, they are integrated via a **middleware pattern**:

```
Agent Code                    Middleware                         Foundry Backend
    │                            │                                    │
    │  chat.completions.create() │                                    │
    │  with tool definitions     │                                    │
    │───────────────────────────▶│                                    │
    │                            │  POST /openai/deployments/gpt-4o/  │
    │                            │  chat/completions                  │
    │                            │  (includes Foundry tool schemas)   │
    │                            │───────────────────────────────────▶│
    │                            │                                    │
    │                            │  Response: tool_calls              │
    │                            │◀───────────────────────────────────│
    │                            │                                    │
    │                            │  [Middleware intercepts tool_calls │
    │                            │   for Foundry tools and executes   │
    │                            │   them via Foundry internal APIs]  │
    │                            │                                    │
    │                            │  POST /tool-execution endpoint     │
    │                            │  (internal Foundry API)            │
    │                            │───────────────────────────────────▶│
    │                            │                                    │
    │                            │  Tool results                     │
    │                            │◀───────────────────────────────────│
    │                            │                                    │
    │  Final response            │                                    │
    │◀───────────────────────────│                                    │
```

**Key insight:** The `FoundryToolsChatMiddleware` (Python) and `UseFoundryTools()` (C#) are middleware layers that sit between your agent code and the LLM. When the LLM returns a tool_call for a Foundry-hosted tool (like `web_search_preview` or `code_interpreter`), the middleware:

1. Intercepts the tool_call
2. Executes it by calling Foundry's internal tool execution API (not a public REST endpoint you'd call yourself)
3. Feeds the result back as a tool message
4. Continues the loop

For **local tools**, the middleware (or the framework) calls the Python function directly in-process.

*(Hypothesis: the Foundry tool execution API is likely a private REST endpoint within the Foundry platform, authenticated via the same Azure AD token used for the OpenAI calls. The middleware likely makes an HTTP POST to something like `https://<project>.services.ai.azure.com/tools/execute` with the tool type, parameters, and credentials. This is not publicly documented.)*

### 4.3 MCP (Model Context Protocol) Tools — Technical Details

**MCP** (Model Context Protocol) is an [open standard](https://modelcontextprotocol.io/) for exposing tools to LLMs. Foundry supports **hosted MCP servers** — remote MCP servers that run as managed services in the Foundry platform.

**How MCP tools are registered:**

```python
# Python — via FoundryToolsChatMiddleware
tools = [{"type": "mcp", "project_connection_id": "MicrosoftLearn"}]
middleware = FoundryToolsChatMiddleware(tools)

# Python — via MCPTool class (for prompt agents)
MCPTool(
    server_url="https://localhost",       # Placeholder — actual URL from connection
    server_label="python_tool",
    require_approval="never",
    allowed_tools=["launchShell", "runPythonCodeInRemoteEnvironment"],
    project_connection_id=os.environ["AZURE_AI_CONNECTION_ID"],
)
```

**The `project_connection_id` is the key.** It refers to a **connection** configured in your Azure AI Foundry project. This connection stores:
- The actual MCP server endpoint URL
- Authentication credentials (API key, managed identity, etc.)
- Metadata about available tools

**Communication flow for MCP tools:**

```
1. Agent starts → Middleware queries Foundry for MCP server capabilities
   (GET tools/list from the MCP server via Foundry proxy)

2. LLM receives tool schemas from MCP server
   (e.g., microsoft_docs_search, microsoft_docs_fetch)

3. LLM generates tool_call → Middleware routes to MCP server
   (POST tools/call to MCP server via Foundry proxy)

4. MCP server executes and returns result → fed back to LLM
```

**MCP protocol uses JSON-RPC 2.0 over HTTP (SSE or stdio).** The hosted MCP servers in Foundry likely use HTTP+SSE transport. The `project_connection_id` abstracts away the actual server URL and auth — the Foundry middleware handles the connection setup.

*(Hypothesis: When you specify `"type": "mcp"`, the middleware calls a Foundry API like `GET /connections/{connection_id}/mcp/tools/list` to discover available tools, converts them to OpenAI function-calling format for the LLM, and when the LLM calls one, the middleware calls `POST /connections/{connection_id}/mcp/tools/call` with the tool name and arguments. The Foundry platform proxies these calls to the actual MCP server, handling auth transparently.)*

### 4.4 Authentication Between Agent Code and Tools

Authentication in Foundry agents uses **Azure AD (Entra ID) tokens** throughout. There is no separate auth between agent code and tools — everything flows through the same token:

```python
# Token provider — refreshes automatically for long-running servers
_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)

# Same token used for LLM calls AND tool execution
chat_client = AzureOpenAIChatClient(ad_token_provider=_token_provider, ...)
```

**Auth flow:**

```
1. Agent container starts with a managed identity (or DefaultAzureCredential)
2. Container requests Azure AD token for scope: cognitiveservices.azure.com
3. Token is used for:
   a. LLM calls (Azure OpenAI chat completions)
   b. Foundry tool execution (web search, code interpreter, MCP)
   c. Project client calls (AIProjectClient)
4. Token auto-refreshes via get_bearer_token_provider() — critical for
   long-running agent servers that outlive the 1-hour token lifetime
```

**For local tools:** No auth needed — they execute in-process as Python function calls.

**For MCP tools:** Auth is handled by the Foundry platform via the `project_connection_id`. The connection stores credentials securely, and the middleware injects them when proxying MCP calls.

**For Bing Grounding:** Uses a `connection_id` that maps to a Bing API subscription configured as a Foundry connection. The `HostedWebSearchTool` passes this connection ID, and Foundry handles the Bing API key injection.

### 4.5 Tool Execution Flow (End to End)

Here's the complete flow when an agent with multiple tool types processes a user message:

```
User: "Search for the latest AI news and calculate 15% of the total articles found"

Step 1: Agent receives message via POST /responses

Step 2: LLM receives message + tool schemas for:
        - web_search_preview (Foundry tool)
        - code_interpreter (Foundry tool)
        - local_calculator (local Python function, if defined)

Step 3: LLM returns: tool_calls: [{name: "web_search_preview", args: {query: "latest AI news"}}]

Step 4: FoundryToolsChatMiddleware intercepts "web_search_preview" tool_call
        → Calls Foundry's internal tool execution API
        → Bing search happens server-side
        → Returns: "Found 47 articles about..."

Step 5: Tool result appended to messages, sent back to LLM

Step 6: LLM returns: tool_calls: [{name: "code_interpreter", args: {code: "print(47 * 0.15)"}}]

Step 7: Middleware intercepts "code_interpreter" tool_call
        → Calls Foundry's sandboxed code execution service
        → Returns: "7.05"

Step 8: Tool result appended to messages, sent back to LLM

Step 9: LLM returns final text: "I found 47 articles about AI news. 15% of 47 is 7.05."

Step 10: Agent Server returns response to client via Responses protocol
```

---

## 5. Deep Dive: Memory — Types, Holistic Example, and Underlying Technology

### 5.1 Memory Types in Foundry Agents

Foundry agents can leverage several types of memory, each serving different purposes:

| Memory Type | Scope | Duration | Technology | Use Case |
|-------------|-------|----------|------------|----------|
| **Conversation history** | Single thread | Session | `AgentThread` / `MessagesState` | Multi-turn dialogue |
| **Thread persistence** | Across requests | Persistent | `JsonLocalFileAgentThreadRepository` / database | Resume conversations |
| **Checkpoints** | Graph state | Persistent | `FileCheckpointRepository` / `MemorySaver` | HITL, resume workflows |
| **Chat message store** | Per thread | Persistent | `ChatMessageStoreProtocol` implementation | Custom conversation storage |
| **RAG / Vector memory** | Cross-session | Persistent | Azure AI Search / vector DB | Knowledge retrieval |
| **Semantic memory** | Cross-session | Persistent | Embeddings + vector store | User preferences, facts |

### 5.2 Holistic Agent Example Using Various Memories

Below is a **hypothetical** comprehensive agent that demonstrates all memory types working together. This combines patterns from the repo samples with reasonable extensions:

```python
"""
Holistic Memory Agent — demonstrates all memory types in a single agent.
NOTE: This is a synthesized example combining patterns from the repo. 
Some components are hypothetical extensions of existing patterns.
"""

import json
from typing import Any, Annotated
from collections.abc import MutableSequence
from dataclasses import dataclass

from agent_framework import (
    BaseAgent, ChatAgent, ChatMessage, ChatMessageStoreProtocol,
    Context, ContextProvider, Role, TextContent, AgentRunResponse,
    AgentThread, ai_function,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.ai.agentserver.agentframework.persistence import (
    JsonLocalFileAgentThreadRepository,
)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ─── MEMORY TYPE 1: Conversation History (in-memory, per thread) ─────────
# This is automatically managed by AgentThread. Every message in the current
# conversation is stored in a ChatMessageStore and passed to the LLM.

class PersistentChatMessageStore(ChatMessageStoreProtocol):
    """
    MEMORY TYPE 2: Persistent Chat Message Store.
    Stores conversation history across HTTP requests for thread continuity.
    In production, back this with a database (Cosmos DB, Redis, etc.).
    """

    def __init__(self):
        self._messages: list[ChatMessage] = []

    async def add_messages(self, messages):
        self._messages.extend(messages)

    async def list_messages(self):
        return self._messages

    async def serialize(self, **kwargs):
        return {"messages": [m.to_dict() for m in self._messages]}

    @classmethod
    async def deserialize(cls, state, **kwargs):
        store = cls()
        if state and "messages" in state:
            for m in state["messages"]:
                store._messages.append(ChatMessage.from_dict(m))
        return store


# ─── MEMORY TYPE 3: Semantic Memory (long-term facts via vector search) ──
# Hypothesis: In production, this would use Azure AI Search with embeddings.

class SemanticMemoryProvider(ContextProvider):
    """
    MEMORY TYPE 3: Long-term semantic memory via RAG.
    Retrieves relevant facts/preferences from a vector store based on the
    current query. This simulates Azure AI Search integration.
    
    Underlying technology (hypothesis):
    - User facts are stored as embeddings in Azure AI Search
    - On each query, we compute an embedding and do a vector similarity search
    - Top-k results are injected as context before the LLM call
    """

    def __init__(self):
        # In production: azure.search.documents.SearchClient
        self._facts_db = {
            "preferences": "User prefers concise answers. Favorite color: blue.",
            "past_interactions": "User asked about Seattle hotels on 2025-01-15.",
            "domain_knowledge": "User works in fintech, interested in AI regulation.",
        }

    async def invoking(self, messages, **kwargs) -> Context:
        query = messages[-1].text.lower() if isinstance(messages, list) else messages.text.lower()

        # Simulate vector search — in production, use embedding similarity
        relevant_facts = []
        for key, fact in self._facts_db.items():
            if any(word in query for word in ["preference", "remember", "history", "past"]):
                relevant_facts.append(fact)

        if not relevant_facts:
            return Context()

        context_text = "Relevant memories:\n" + "\n".join(f"- {f}" for f in relevant_facts)
        return Context(messages=[
            ChatMessage(role=Role.SYSTEM, text=context_text)
        ])


# ─── MEMORY TYPE 4: Episodic Memory (summarized past conversations) ─────
# Hypothesis: Summarize old conversations and store summaries for context.

class EpisodicMemoryProvider(ContextProvider):
    """
    MEMORY TYPE 4: Episodic memory — summaries of past conversations.
    
    Underlying technology (hypothesis):
    - After each conversation ends, an LLM summarizes the key points
    - Summaries are stored in Cosmos DB indexed by user ID and timestamp
    - On new conversations, recent summaries are retrieved and injected
    """

    async def invoking(self, messages, **kwargs) -> Context:
        # In production: query Cosmos DB for recent conversation summaries
        summaries = [
            "Previous session (Jan 15): User booked Alpine Ski House in Seattle for Feb 1-3.",
            "Previous session (Jan 10): User asked about return policies for outdoor gear.",
        ]
        context_text = "Past conversation summaries:\n" + "\n".join(f"- {s}" for s in summaries)
        return Context(messages=[
            ChatMessage(role=Role.SYSTEM, text=context_text)
        ])


# ─── MEMORY TYPE 5: Working Memory (scratchpad for current task) ─────────

@ai_function(approval_mode="never")
def save_to_scratchpad(
    key: Annotated[str, "Key to store the note under"],
    value: Annotated[str, "The note or intermediate result to save"],
) -> str:
    """Save a note or intermediate result to the agent's scratchpad."""
    # In production: store in Redis or an in-memory dict scoped to the thread
    return f"Saved '{key}': '{value}' to scratchpad."


@ai_function(approval_mode="never")
def read_from_scratchpad(
    key: Annotated[str, "Key to retrieve"],
) -> str:
    """Read a note from the agent's scratchpad."""
    return f"Retrieved value for '{key}' from scratchpad."


# ─── ASSEMBLE THE HOLISTIC AGENT ─────────────────────────────────────────

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)

def create_holistic_agent():
    agent = AzureOpenAIChatClient(ad_token_provider=_token_provider).create_agent(
        name="HolisticMemoryAgent",
        instructions="""You are a helpful assistant with rich memory capabilities.
        
You have access to:
1. Conversation history — the current thread's messages (automatic)
2. Semantic memory — long-term facts about the user (via context)
3. Episodic memory — summaries of past conversations (via context)
4. Scratchpad — temporary notes for the current task (via tools)

Use your memories to provide personalized, context-aware responses.
When you learn something new about the user, save it to the scratchpad.
""",
        context_providers=[
            SemanticMemoryProvider(),
            EpisodicMemoryProvider(),
        ],
        tools=[save_to_scratchpad, read_from_scratchpad],
        chat_message_store_factory=PersistentChatMessageStore,
    )
    return agent


async def main():
    agent = create_holistic_agent()
    thread_repo = JsonLocalFileAgentThreadRepository(
        agent=agent, storage_path="./thread_storage"
    )
    await from_agent_framework(agent, thread_repository=thread_repo).run_async()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 5.3 Underlying Technology Hypothesis

Here's how each memory type is likely implemented under the hood:

#### Conversation History (Short-term)
- **Technology:** In-memory list of `ChatMessage` objects
- **Implementation:** `AgentThread` holds a reference to a `ChatMessageStoreProtocol` implementation. On each `run()` call, the thread provides all stored messages to the LLM.
- **Limitation:** Limited by LLM context window. For very long conversations, older messages are likely truncated or summarized.

#### Thread Persistence
- **Technology:** JSON file storage (dev) → Cosmos DB or Azure Storage (production)
- **Implementation:** `JsonLocalFileAgentThreadRepository` serializes thread state (all messages + metadata) to JSON files in a local directory. The thread ID maps to a file path. On each request, the thread is deserialized, updated, and re-serialized.
- **Production hypothesis:** In a deployed Foundry agent, thread state is likely stored in **Azure Cosmos DB** (based on the infrastructure templates that provision Cosmos DB for "Standard" agent setups). The `conversation` field in the Responses protocol likely maps to a Cosmos DB document ID.

#### Checkpoints (LangGraph)
- **Technology:** `MemorySaver` (in-memory dict) or `FileCheckpointRepository` (file-based)
- **Implementation:** LangGraph serializes the entire graph state (all node states, pending edges, current position) into a checkpoint. For `interrupt()`, the checkpoint captures the exact point of suspension.
- **Production hypothesis:** For deployed agents, Foundry likely provides a server-side checkpoint store backed by Azure Storage or Cosmos DB. The `FileCheckpointRepository` in the samples is a development convenience.

#### RAG / Vector Memory (Long-term)
- **Technology:** Azure AI Search with vector indexes
- **Implementation:** The `ContextProvider` pattern is the integration point. Before each LLM call, the provider queries Azure AI Search with the user's message (or an embedding of it), retrieves relevant documents, and injects them as context messages.
- **Hypothesis on embeddings:** Text is embedded using Azure OpenAI's embedding models (e.g., `text-embedding-3-small`). These embeddings are stored in Azure AI Search vector fields. At query time, the user's message is embedded and a cosine similarity search retrieves the top-k documents.

#### Semantic Memory (Cross-session facts)
- **Technology:** Vector store + metadata storage
- **Hypothesis:** Individual facts about a user (preferences, past actions) are stored as embedding vectors with metadata in Azure AI Search or Cosmos DB with vector indexing (Cosmos DB now supports vector search). This is distinct from RAG in that it stores atomic facts rather than document chunks.

#### Episodic Memory (Conversation summaries)
- **Technology:** LLM summarization + document store
- **Hypothesis:** At the end of each conversation, a summarization prompt is sent to the LLM to extract key facts and decisions. These summaries are stored in Cosmos DB with user ID and timestamp indexes. On new conversations, recent summaries are retrieved by timestamp and injected as system messages.

---

## 6. Build, Test, Deploy, Invoke Lifecycle

### 6.1 Writing the Code

Every hosted agent sample follows a standard structure:

```
my-agent/
├── main.py              # Agent logic + server entry point
├── agent.yaml           # Deployment configuration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── .env.sample          # Template for local env vars
└── README.md            # Documentation
```

**Step 1: Choose your framework** — Agent Framework, LangGraph, or Custom.

**Step 2: Write `main.py`** — implement agent logic and wrap with the appropriate adapter:

```python
# Agent Framework
from azure.ai.agentserver.agentframework import from_agent_framework
from_agent_framework(create_agent()).run()

# LangGraph
from azure.ai.agentserver.langgraph import from_langgraph
from_langgraph(build_graph()).run()

# Custom
from azure.ai.agentserver.core import FoundryCBAgent
# Subclass and implement run_agent()
```

**Step 3: Write `agent.yaml`** — declare the agent's identity, environment, and resources:

```yaml
name: my-agent
description: My custom agent
template:
  name: my-agent
  kind: hosted                    # "hosted" = Foundry manages the container
  protocols:
    - protocol: responses         # OpenAI Responses API compatibility
      version: v1
  environment_variables:
    - name: AZURE_OPENAI_ENDPOINT
      value: ${AZURE_OPENAI_ENDPOINT}
    - name: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
      value: "{{chat}}"           # References the model resource below
resources:
  - kind: model
    id: gpt-4o-mini
    name: chat                    # Referenced by {{chat}} above
```

**Step 4: Write `Dockerfile`:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . user_agent/
WORKDIR /app/user_agent
RUN pip install -r requirements.txt
EXPOSE 8088
CMD ["python", "main.py"]
```

**Key port:** All agent servers listen on **port 8088** by default.

### 6.2 Testing Locally

**Step 1: Set up environment:**

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (copy .env.sample to .env and fill in values)
cp .env.sample .env
# Edit .env with your Azure credentials
```

**Step 2: Run the agent locally:**

```bash
python main.py
# Output: Server running on http://localhost:8088/
```

**Step 3: Test with curl:**

```bash
# Non-streaming request
curl -X POST http://localhost:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, what can you do?", "stream": false}' | jq .

# Streaming request
curl -X POST http://localhost:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Tell me about Seattle hotels", "stream": true}'
```

**Step 4: Test with Docker (validates the container build):**

```bash
docker build --platform=linux/amd64 -t my-agent .
docker run -p 8088:8088 --env-file .env my-agent
```

**Repo testing infrastructure:**

The repository uses `pytest` with a custom `conftest.py` that auto-discovers sample scripts:

📁 [`conftest.py`](../conftest.py)

```python
# Treats every .py file under doc-samples/agents/python/ as a test
# The test passes if the script runs without raising an exception
class SampleItem(pytest.Item):
    def runtest(self):
        runpy.run_path(str(self.fspath))
```

Linting and formatting tools (from [`tox.ini`](../tox.ini)):

```bash
# Format code
tox -e black -- --check .

# Lint code
tox -e ruff -- check .

# Run tests
tox -e pytest -- samples/python/hosted-agents/
```

### 6.3 Deploying to Azure AI Foundry

Deployment uses the **Azure Developer CLI (`azd`)** with the `ai agent` extension:

**Step 1: Provision infrastructure** (if not already done):

Use Bicep templates from [`infrastructure/infrastructure-setup-bicep/`](../infrastructure/infrastructure-setup-bicep/):

```bash
# Basic setup — quickest for prototyping
az deployment group create \
  --resource-group myResourceGroup \
  --template-file infrastructure/infrastructure-setup-bicep/40-basic-agent-setup/main.bicep
```

Available infrastructure tiers:

| Template | Description | Best For |
|----------|-------------|----------|
| `40-basic-agent-setup/` | Foundry project + gpt-4o | Rapid prototyping |
| `41-standard-agent-setup/` | + Customer-managed storage (Cosmos DB, Storage, AI Search) | Production with data control |
| `15-private-network-standard-agent-setup/` | + BYO Virtual Network | Enterprise security |

**Step 2: Deploy the agent:**

```bash
# Install the azd ai agent extension
azd extension install ai

# Deploy from your agent directory
cd samples/python/hosted-agents/agent-framework/echo-agent
azd ai agent deploy
```

*(Reference: https://aka.ms/azdaiagent/docs)*

**What happens behind the scenes:**
1. `azd` reads `agent.yaml` for configuration
2. Builds the Docker image from `Dockerfile`
3. Pushes the image to Azure Container Registry (ACR)
4. Creates a hosted agent version in the Foundry project
5. Creates a deployment that maps the agent to the container image
6. The Foundry platform provisions the necessary infrastructure to serve the agent

### 6.4 Invoking the Deployed Agent

Once deployed, your agent is accessible via the **OpenAI Responses API** endpoint:

```bash
# Get your project endpoint from Azure portal
PROJECT_ENDPOINT="https://<your-account>.services.ai.azure.com/api/projects/<your-project>"

# Get an Azure AD token
TOKEN=$(az account get-access-token --resource https://cognitiveservices.azure.com --query accessToken -o tsv)

# Invoke the agent
curl -X POST "${PROJECT_ENDPOINT}/responses" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, what can you do?",
    "agent": {"name": "echo-agent", "type": "agent_reference"},
    "stream": false
  }'
```

**Using the Python SDK:**

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

client = AIProjectClient(
    endpoint="https://<account>.services.ai.azure.com",
    credential=DefaultAzureCredential(),
)
openai_client = client.get_openai_client()

response = openai_client.responses.create(
    input="What is the weather in Seattle?",
    extra_body={"agent": {"name": "my-agent", "type": "agent_reference"}},
)
print(response.output_text)
```

**Conversation continuity:**

```python
# First request — starts a new conversation
response1 = openai_client.responses.create(
    input="Book me a hotel in Seattle for Feb 1-3",
    extra_body={"agent": {"name": "hotel-agent", "type": "agent_reference"}},
)

# Second request — continues the conversation
response2 = openai_client.responses.create(
    input="Actually, change that to Feb 5-7",
    extra_body={
        "agent": {"name": "hotel-agent", "type": "agent_reference"},
        "conversation": response1.conversation_id,   # Thread continuity
    },
)
```

---

## 7. Other Important Dimensions

### 7.1 Observability and Tracing

The custom system-utility-agent sample demonstrates **OpenTelemetry tracing**:

📁 [`samples/python/hosted-agents/custom/system-utility-agent/main.py`](../samples/python/hosted-agents/custom/system-utility-agent/main.py)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# In the agent initialization:
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Creating custom spans around tool calls:
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("agent_turn") as span:
    span.set_attribute("turn_number", turn)
    span.set_attribute("tool_name", tool_call.function.name)
    # ... execute tool ...
```

**Foundry integrates with Azure Monitor** via `azure-monitor-opentelemetry`. The C# samples show `.UseOpenTelemetry()` middleware for automatic tracing.

*(Hypothesis: when deployed to Foundry, the platform likely auto-injects an OpenTelemetry collector endpoint via environment variables. The `azure-monitor-opentelemetry` SDK sends traces to Application Insights, where you can see the full agent execution flow — each LLM call, tool execution, and response latency.)*

### 7.2 Infrastructure as Code

The repo includes extensive Bicep and Terraform templates:

📁 [`infrastructure/infrastructure-setup-bicep/`](../infrastructure/infrastructure-setup-bicep/)

```
infrastructure/
├── infrastructure-setup-bicep/
│   ├── 40-basic-agent-setup/           # Minimal Foundry project + model
│   ├── 41-standard-agent-setup/        # + Cosmos DB, Storage, AI Search
│   ├── 42-basic-agent-setup-with-customization/
│   ├── 15-private-network-standard-agent-setup/  # + VNet isolation
│   └── ... (40+ templates)
└── infrastructure-setup-terraform/     # Terraform equivalents
```

### 7.3 Security and Network Isolation

The infrastructure templates support three security tiers:

1. **Basic** — Public endpoints, platform-managed storage
2. **Standard** — Customer-managed Azure resources (you own the Cosmos DB, Storage Account, etc.)
3. **Private Network** — Bring Your Own VNet, all traffic stays within your network

```
┌─────────────────────────────────────────────────────┐
│ Your Virtual Network                                 │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Agent    │──│ Foundry  │──│ Cosmos DB         │  │
│  │ Container│  │ Project  │  │ (threads, state)  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│       │                              │               │
│  ┌──────────┐              ┌──────────────────┐     │
│  │ Azure    │              │ Azure AI Search  │     │
│  │ OpenAI   │              │ (vector store)   │     │
│  └──────────┘              └──────────────────┘     │
│                                                      │
│  Private endpoints — no public internet exposure     │
└─────────────────────────────────────────────────────┘
```

---

## Summary: Choosing Your Approach

| Dimension | Simplest | Most Powerful |
|-----------|----------|---------------|
| **Framework** | Prompt Agent (no code) | Custom `FoundryCBAgent` (full control) |
| **Sweet spot** | Agent Framework `ChatAgent` | LangGraph for complex flows |
| **Tools** | Local Python functions | MCP + Foundry hosted tools |
| **Memory** | Built-in `AgentThread` | Custom stores + RAG + checkpoints |
| **Deployment** | `azd ai agent deploy` | Bicep/Terraform + CI/CD |
| **Language** | Python | Python, C#, (TypeScript/Java for non-hosted) |

**Start here:**
1. Clone the repo: `git clone https://github.com/xue-cai/foundry-samples`
2. Try the echo agent: `cd samples/python/hosted-agents/agent-framework/echo-agent && pip install -r requirements.txt && python main.py`
3. Add tools: Look at `agent-with-local-tools` and `agent-with-foundry-tools`
4. Add memory: Look at `human-in-the-loop/agent-with-thread-and-hitl`
5. Deploy: Follow https://aka.ms/azdaiagent/docs
