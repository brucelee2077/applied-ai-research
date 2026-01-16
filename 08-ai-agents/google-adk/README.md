# Google Agent Development Kit (ADK) Tutorials

> Build enterprise-grade AI agents with Google's ADK framework and Agent-to-Agent (A2A) communication

## Overview

This section covers **Google's Agent Development Kit (ADK)**, Google Cloud's framework for building production-ready AI agents with built-in Agent-to-Agent (A2A) communication capabilities.

### What You'll Learn

- Google ADK fundamentals and architecture
- Building agents with skills (tools) and memory
- Agent-to-Agent (A2A) communication protocols
- Multi-agent coordination patterns
- Deploying ADK agents to Google Cloud
- ADK vs LangGraph comparison

---

## 📚 Tutorial Structure

### 1. **[ADK Fundamentals](./01_adk_fundamentals.ipynb)**
- What is Google ADK?
- Architecture and core concepts
- Setup and installation
- Skills, memory, and agents
- ADK vs LangGraph
- Deployment patterns

### 2. **[Agent-to-Agent (A2A) Communication](./02_agent_to_agent_communication.ipynb)**
- A2A protocol basics
- Message passing between agents
- Coordinator patterns
- Multi-agent workflows (Pipeline, Broadcast, Hierarchical)
- Production A2A patterns
- Complete multi-agent system example

---

## 🚀 Quick Start

### Installation

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Install ADK
pip install google-cloud-adk google-cloud-aiplatform

# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Your First ADK Agent (5 minutes)

```python
from google.cloud.adk import Agent, Skill, Memory

# Define a skill
class WeatherSkill(Skill):
    def __init__(self):
        super().__init__(name="get_weather", description="Get weather")

    def execute(self, location: str) -> dict:
        return {"location": location, "temp": "72°F"}

# Create agent
class WeatherAgent(Agent):
    def __init__(self):
        super().__init__(
            name="weather_assistant",
            description="Weather information agent",
            skills=[WeatherSkill()],
            memory=Memory(type="conversation")
        )

# Use agent
agent = WeatherAgent()
```

---

## 🎯 Key Concepts

### ADK Architecture

```
┌─────────────────────────────────────┐
│      Google ADK Architecture        │
├─────────────────────────────────────┤
│  ┌──────────────────────────────┐  │
│  │    Agent Runtime Engine      │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│   ┌─────────┼─────────┐            │
│   ▼         ▼         ▼            │
│ ┌────┐  ┌────┐  ┌──────┐          │
│ │LLM │  │Tools│  │Memory│          │
│ └────┘  └────┘  └──────┘          │
│  ┌──────────────────────────────┐  │
│  │   A2A Communication Layer    │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │   Google Cloud Services      │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### A2A Communication

```
┌──────────────┐      Message       ┌──────────────┐
│   Agent A    │ ─────────────────► │   Agent B    │
│ (Coordinator)│                     │ (Specialist) │
│              │ ◄───────────────── │              │
└──────────────┘     Response       └──────────────┘
```

---

## 💡 Common Use Cases

### 1. Multi-Agent Research System
**Pattern**: Pipeline (Research → Write → Review → Publish)

```python
coordinator = ContentCoordinator([
    ResearchAgent(),
    WriterAgent(),
    ReviewerAgent(),
    PublisherAgent()
])
```

### 2. Parallel Processing
**Pattern**: Broadcast (send to multiple specialists)

```python
results = await broadcast_coordinator.broadcast({
    "task": "analyze",
    "data": dataset
})
```

### 3. Hierarchical Organization
**Pattern**: Tree structure (CEO → Managers → Workers)

```python
ceo = HierarchicalCoordinator()
result = await ceo.delegate_project(project)
```

---

## 📊 ADK vs LangGraph

| Feature | Google ADK | LangGraph |
|---------|-----------|-----------|
| **Provider** | Google Cloud | Open Source |
| **Integration** | Native GCP | Platform agnostic |
| **A2A** | Built-in protocol | Custom implementation |
| **Deployment** | Cloud Functions/Run | Any platform |
| **Monitoring** | Cloud Logging | Custom |
| **Best For** | Enterprise GCP | General purpose |

### When to Use ADK

Choose Google ADK when:
- ✅ Already on Google Cloud Platform
- ✅ Need enterprise monitoring/logging
- ✅ Building multi-agent systems
- ✅ Want managed infrastructure
- ✅ Tight Vertex AI integration needed

### When to Use LangGraph

Choose LangGraph when:
- ✅ Need maximum flexibility
- ✅ Platform agnostic
- ✅ Open source preference
- ✅ Custom control flow
- ✅ Smaller deployments

---

## 🏗️ Multi-Agent Patterns

### Pipeline Pattern
Sequential processing through agent chain:
```
Agent A → Agent B → Agent C → Result
```

### Broadcast Pattern
Parallel execution across agents:
```
        ┌─→ Agent 1 ─┐
Input ─→├─→ Agent 2 ─┤→ Aggregate
        └─→ Agent 3 ─┘
```

### Hierarchical Pattern
Tree-based delegation:
```
       CEO Agent
          │
    ┌─────┼─────┐
    │     │     │
  Mgr1  Mgr2  Mgr3
```

---

## 🔧 Production Features

### Timeouts and Retries

```python
response = await agent.send_with_timeout(
    target_agent=specialist,
    message=request,
    timeout=30.0
)
```

### Error Handling

```python
response = await agent.send_with_retry(
    target_agent=specialist,
    message=request,
    max_retries=3
)
```

### Message Priority

```python
message = A2AMessage(
    from_agent=self.id,
    to_agent=target.id,
    payload=task,
    priority=10  # High priority
)
```

---

## 🚀 Deployment

### Cloud Functions

```bash
gcloud functions deploy my-agent \
  --runtime python39 \
  --trigger-http \
  --entry-point handle_request
```

### Cloud Run

```bash
gcloud run deploy my-agent \
  --source . \
  --platform managed
```

### Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Deploy agent
endpoint = agent.deploy_to_vertex_ai()
```

---

## 📖 Best Practices

1. **Define Clear Interfaces** between agents
2. **Use Correlation IDs** for tracking
3. **Implement Timeouts** on all A2A calls
4. **Log All Messages** for debugging
5. **Add Retry Logic** for resilience
6. **Monitor Agent Health** continuously
7. **Use Circuit Breakers** for failing agents

---

## 🔍 Example: Complete A2A Workflow

```python
class CompleteWorkflow:
    async def create_content(self, topic: str):
        # 1. Research phase
        research = await self.researcher.research(topic)

        # 2. Writing phase
        article = await self.writer.write(research)

        # 3. Review phase
        review = await self.reviewer.review(article)

        # 4. Publish if approved
        if review["approved"]:
            return await self.publisher.publish(article)

        return {"status": "rejected", "feedback": review}
```

---

## 📚 Additional Resources

### Official Documentation
- [Google Cloud ADK Docs](https://cloud.google.com/adk)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agent-builder)
- [A2A Protocol Specification](https://cloud.google.com/adk/a2a)

### Related Technologies
- [Vertex AI](https://cloud.google.com/vertex-ai)
- [Cloud Functions](https://cloud.google.com/functions)
- [Cloud Run](https://cloud.google.com/run)

---

## 🎓 Learning Path

1. **Start Here**: [ADK Fundamentals](./01_adk_fundamentals.ipynb)
2. **Then**: [A2A Communication](./02_agent_to_agent_communication.ipynb)
3. **Practice**: Build your own multi-agent system
4. **Deploy**: Push to Google Cloud
5. **Monitor**: Set up logging and metrics

---

## 🆚 Comparison with Other Frameworks

**ADK Strengths:**
- Native GCP integration
- Built-in A2A protocol
- Enterprise monitoring
- Managed deployment

**LangGraph Strengths:**
- Platform flexibility
- Open source
- Explicit control flow
- Broader community

**Best Approach:**
Use both! ADK for production GCP deployments, LangGraph for development and prototyping.

---

**Ready to build enterprise AI agents?** Start with [01_adk_fundamentals.ipynb](./01_adk_fundamentals.ipynb)!
