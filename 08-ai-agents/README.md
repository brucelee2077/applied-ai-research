# AI Agents: LangGraph & Google ADK

> Comprehensive tutorials on building autonomous AI agents using both **LangGraph** (open-source) and **Google ADK** (enterprise) frameworks

## Overview

This section provides complete coverage of AI agent development using two major frameworks:

### **LangGraph** - Flexible Open-Source Framework
Build agents with explicit control flow, state management, and tool integration. Perfect for prototyping and platform-agnostic deployments.

### **Google ADK** - Enterprise Cloud Framework
Build production-ready agents with native Google Cloud integration, built-in Agent-to-Agent (A2A) communication, and managed deployment.

### What You'll Learn

**LangGraph Track:**
- Core Concepts: Understanding AI agents vs traditional LLMs
- LangGraph Fundamentals: Nodes, edges, state management, control flow
- Tool Integration: Giving agents external capabilities
- Advanced Patterns: ReAct, Plan-Execute, Reflection, Multi-Agent
- Production Systems: Persistence, error handling, monitoring

**Google ADK Track:**
- ADK Architecture: Skills, memory, agent runtime
- Agent-to-Agent (A2A): Multi-agent communication protocols
- Coordination Patterns: Pipeline, broadcast, hierarchical
- Google Cloud Deployment: Functions, Cloud Run, Vertex AI
- Enterprise Features: Monitoring, logging, scaling

---

## 📚 Learning Paths

### 🟢 LangGraph Track (Open Source)

#### Beginner

1. **[What Are AI Agents?](./langgraph/01_what_are_ai_agents.ipynb)**
   - Introduction to AI agents and their capabilities
   - Key differences from simple LLM interactions
   - Agent design patterns and use cases
   - Why LangGraph for agent orchestration

2. **[LangGraph Basics](./langgraph/02_langgraph_basics.ipynb)**
   - Core concepts: StateGraph, Nodes, Edges
   - Building your first graph
   - Conditional routing and loops
   - Simple LLM-powered agents

#### Intermediate

3. **[State Management & Tools](./langgraph/03_state_management_and_tools.ipynb)**
   - Advanced state schemas and reducers
   - Integrating external tools and APIs
   - Persistence and checkpointing
   - Human-in-the-loop workflows
   - Memory strategies (short-term vs long-term)

#### Advanced

4. **[Advanced Patterns](./langgraph/04_advanced_patterns.ipynb)**
   - **ReAct**: Reasoning and Acting pattern
   - **Plan-and-Execute**: Strategic planning agents
   - **Reflection**: Self-correcting agents
   - **Multi-Agent**: Collaborative agent systems
   - **Hierarchical**: Task decomposition
   - Production patterns (circuit breakers, fallbacks)

5. **[Practical Examples](./langgraph/05_practical_examples.ipynb)**
   - Research Assistant (search & synthesize)
   - Code Generator (write, test, debug)
   - Customer Support (categorize & respond)
   - Production tips and monitoring

6. **[Tool Calling Deep Dive](./langgraph/06_tool_calling_deep_dive.ipynb)** 🔧
   - Tool call mechanics and protocols
   - Parallel vs sequential execution
   - Error handling and retries
   - Argument validation
   - Caching and optimization

### 🔵 Google ADK Track (Enterprise)

7. **[ADK Fundamentals](./google-adk/01_adk_fundamentals.ipynb)** ☁️
   - Google ADK architecture
   - Skills, memory, and agents
   - ADK vs LangGraph comparison
   - Google Cloud setup
   - Deployment patterns (Functions, Cloud Run)

8. **[Agent-to-Agent (A2A) Communication](./google-adk/02_agent_to_agent_communication.ipynb)** 🤝
   - A2A protocol basics
   - Message passing between agents
   - Coordinator patterns
   - Multi-agent workflows (Pipeline, Broadcast, Hierarchical)
   - Production A2A patterns
   - Complete multi-agent system example

---

## 🎯 Quick Start

### Prerequisites

```bash
# Install required packages
pip install langgraph langchain langchain-openai langchain-community

# Set up your API key
export OPENAI_API_KEY="your-key-here"
```

### Your First Agent (5 minutes)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI

# 1. Define state
class State(TypedDict):
    messages: list

# 2. Create node
def call_model(state: State):
    llm = ChatOpenAI()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Build graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# 4. Run!
app = workflow.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

---

## 🏗️ Repository Structure

```
08-ai-agents/
├── langgraph/
│   ├── 01_what_are_ai_agents.ipynb       # Introduction & concepts
│   ├── 02_langgraph_basics.ipynb         # Core fundamentals
│   ├── 03_state_management_and_tools.ipynb  # Advanced state & tools
│   ├── 04_advanced_patterns.ipynb        # Production patterns
│   └── README.md                         # This file
├── examples/
│   ├── research_agent.py                 # Research assistant example
│   ├── coding_agent.py                   # Code generation agent
│   └── data_analysis_agent.py            # Data analyst agent
└── README.md                             # Section overview
```

---

## 💡 Key Concepts

### Agent Components

```
┌─────────────────────────────────────┐
│         AI Agent System             │
│                                     │
│  ┌──────────┐      ┌────────────┐  │
│  │  Brain   │◄────►│   Memory   │  │
│  │  (LLM)   │      │  (State)   │  │
│  └────┬─────┘      └────────────┘  │
│       │                             │
│       ▼                             │
│  ┌──────────┐      ┌────────────┐  │
│  │  Tools   │◄────►│  Planning  │  │
│  │ (Actions)│      │  (Strategy)│  │
│  └──────────┘      └────────────┘  │
└─────────────────────────────────────┘
```

### LangGraph Flow

```
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
    ┌────▼────┐
    │  Node A │ (Plan)
    └────┬────┘
         │
    ┌────▼────┐
    │  Node B │ (Execute)
    └────┬────┘
         │
      ┌──┴──┐
      │     │
      ▼     ▼
   Success  Retry?
      │      │
      │      └──► (Loop back)
      ▼
   ┌─────────┐
   │   END   │
   └─────────┘
```

---

## 🔧 Common Use Cases

### 1. Research Assistant
Searches web, reads papers, synthesizes findings

**Pattern**: ReAct
**Tools**: Web search, PDF reader, summarization
**Notebook**: [examples/research_agent.py](./examples/research_agent.py)

### 2. Code Generator
Writes, tests, and debugs code

**Pattern**: Plan-Execute + Reflection
**Tools**: Code execution, testing, linting
**Notebook**: [examples/coding_agent.py](./examples/coding_agent.py)

### 3. Data Analyst
Analyzes datasets, creates visualizations

**Pattern**: ReAct
**Tools**: Pandas, matplotlib, statistics
**Notebook**: [examples/data_analysis_agent.py](./examples/data_analysis_agent.py)

### 4. Customer Support
Answers questions, escalates when needed

**Pattern**: Multi-Agent (router + specialists)
**Tools**: Knowledge base, ticketing system
**Features**: Human-in-the-loop

---

## 📖 Detailed Topics

### State Management

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    # Messages accumulate (append)
    messages: Annotated[list, operator.add]

    # Current value (replace)
    current_step: str

    # Custom reducer
    metadata: Annotated[dict, custom_merge_function]
```

### Tool Integration

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

tools = [search_web]
llm_with_tools = llm.bind_tools(tools)
```

### Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Save agent state to database
checkpointer = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=checkpointer)

# Resume conversations
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(input, config)
```

---

## 🎓 Learning Resources

### Official Documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [Agent Design Patterns](https://python.langchain.com/docs/modules/agents/)

### Research Papers
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

---

## 🚀 Best Practices

### 1. Start Simple
- Begin with linear graphs
- Add complexity gradually
- Test each component independently

### 2. Design State Carefully
- Keep state minimal
- Use appropriate reducers
- Document state schema

### 3. Handle Errors
```python
def safe_node(state):
    try:
        # Node logic
        return updated_state
    except Exception as e:
        return {"error": str(e)}
```

### 4. Add Safeguards
- Maximum iteration limits
- Timeout handling
- Cost monitoring
- Human approval for critical actions

### 5. Monitor Performance
- Log all agent actions
- Track tool usage
- Measure success rates
- Optimize prompts

---

## 🔍 Debugging Tips

### 1. Stream Intermediate Steps
```python
for step in app.stream(input):
    print(f"Step: {step}")
```

### 2. Visualize Graph
```python
from IPython.display import Image
Image(app.get_graph().draw_mermaid_png())
```

### 3. Add Debug Nodes
```python
def debug_node(state):
    print(f"State: {state}")
    return state

workflow.add_node("debug", debug_node)
```

---

## ⚠️ Common Pitfalls

### 1. Infinite Loops
**Problem**: Agent loops endlessly
**Solution**: Add max iteration limits

```python
def should_continue(state):
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    return "continue"
```

### 2. Context Overflow
**Problem**: Too many messages in context
**Solution**: Summarize or prune old messages

### 3. Tool Failures
**Problem**: External APIs fail
**Solution**: Add retries and fallbacks

### 4. Cost Explosion
**Problem**: Too many LLM calls
**Solution**: Cache results, use smaller models for simple tasks

---

## 📊 Comparison: Agent Frameworks

| Framework | Control | Complexity | Best For |
|-----------|---------|------------|----------|
| **LangGraph** | High | Medium | Production systems |
| LangChain Agents | Medium | Low | Simple workflows |
| AutoGPT | Low | High | Experimentation |
| CrewAI | Medium | Low | Multi-agent tasks |

**Why LangGraph?**
- ✅ Explicit control flow
- ✅ State persistence built-in
- ✅ Cyclical graphs supported
- ✅ Easy debugging and testing
- ✅ Production-ready features

---

## 🛠️ Tools & Integrations

### Common Tool Categories

**Information Retrieval**
- Web search (Tavily, SerpAPI)
- Vector databases (Pinecone, Chroma)
- APIs (REST, GraphQL)

**Code Execution**
- Python REPL
- Shell commands
- Jupyter notebooks

**Data Processing**
- Pandas operations
- SQL queries
- File operations

**External Services**
- Email (SMTP)
- Calendar (Google Calendar API)
- Slack, Discord, etc.

---

## 🎯 Next Steps

After completing the tutorials:

1. **Build a Project**: Apply what you learned to a real problem
2. **Explore Multi-Agent**: Try collaborative agent systems
3. **Optimize**: Reduce latency and cost
4. **Deploy**: Put your agent in production
5. **Monitor**: Track performance and improve

---

## 📝 Additional Resources

### Example Projects
- Customer support chatbot
- Research paper summarizer
- Code review assistant
- Data analysis agent
- Task automation agent

### Advanced Topics
- Multi-modal agents (vision + text)
- Agent evaluation frameworks
- Prompt optimization
- Memory management strategies
- Security and safety considerations

---

## 🤝 Contributing

Found an issue or have suggestions?
- Open an issue in the main repository
- Submit a pull request with improvements
- Share your agent implementations

---

## 📄 License

This educational content is part of the Applied AI Research repository.
See the main [LICENSE](../../LICENSE) for details.

---

**Ready to build AI agents?** Start with [01_what_are_ai_agents.ipynb](./langgraph/01_what_are_ai_agents.ipynb)!
