# AI Tariff Agent

## Week 1: Definition and Setup

## Description
AI agent for tariff classification (HS Code).  
This project is developed as part of the Hackathon AI Dev Days 2026.

## MVP Goal
Receive a product description and return a suggested HS Code.

## Project Plan
- Week 1: Definition and Setup
- Week 2: Agent Construction
- Week 3: Security and Scalability
- Week 4: Demo and UX
- Week 5: Final Delivery

## Getting Started
```bash
git clone https://github.com/fernando-pedernera/ai-tariff-agent.git
cd ai-tariff-agent
```

## Architecture (Conceptual) 

We have a conceptual agent that receives a description and provides a suggested HS Code. This is the first step toward a fully functional system.


flowchart TD
    A[User Entry point] --> B[Interface Chat / Form<br/>Simple UI for product description input]
    B --> C[Classification Agent Microsoft Agent Framework<br/>Orchestrates the reasoning flow]
    C --> D[Model Router Azure<br/>Shows scalability even if routing to a single agent]
    D --> E[Azure MCP Mock external regulations<br/>Simulates access to customs laws]
    D --> F[Azure AI Foundry GPT-4o<br/>Reasoning engine Brain]
    F <--> G[Azure AI Search RAG<br/>External normative memory, simulated in MVP]
    F --> H[Microsoft Foundry Security<br/>Compliance and enterprise readiness]
    H --> I[Output: Suggested HS Code<br/>Final result delivered to the user]
