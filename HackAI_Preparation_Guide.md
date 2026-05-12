# HackAI 4.0 - Preparation & API Quickstart Guide

Welcome to the HackAI 4.0 preparation guide! This document is designed to get you up and running with the APIs and tools required for the hackathon. It summarizes the setup instructions and provides the recommended learning paths.

## 1. Setting Up Your API Keys

You should store your API keys securely in a `.env` file to prevent accidental exposure. 

1. Create a file named `.env` in the root of your project directory.
2. Add your keys in the following format:

```env
HF_TOKEN=your_hugging_face_token_here
GEMINI_API_KEY=your_google_ai_studio_key_here
GROQ_API_KEY=your_groq_api_key_here
WANDB_API_KEY=your_weights_and_biases_key_here
```

⚠️ **Important Security Note:** Never commit your `.env` file to version control (e.g., GitHub). Make sure to add `.env` to your `.gitignore` file!

## 2. Recommended Packages

To interact with these APIs and build your AI models/agents, you'll need a set of standard Python packages. You can install them using pip:

```bash
pip install python-dotenv huggingface_hub google-generativeai groq wandb langchain langchain-community langgraph
```

- **python-dotenv**: To load your API keys from the `.env` file securely.
- **huggingface_hub**: To pull pre-trained models and datasets from Hugging Face.
- **google-generativeai**: The official Google SDK for Gemini models.
- **groq**: To interact with the ultra-fast Groq API for open-source LLMs.
- **wandb**: For experiment tracking and model evaluation.
- **langchain & langgraph**: Powerful frameworks for building agentic applications.

## 3. Recommended Learning Resources

Familiarity with these topics will significantly accelerate your project development during the hackathon:

### Transformer Architecture
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)

### LLM Agents
- [A Visual Guide to LLM Agents](https://newsletter.maartengrootendorst.com)

### Agent Frameworks (Hands-On)
- [LangChain Foundations (Course)](https://academy.langchain.com)
- [LangGraph Introduction (Course)](https://academy.langchain.com)

### Voice Agents
- [Deep Dive into Voice Agents (HuggingFace Blog)](https://huggingface.co/blog/voice-agents)
- [Build a Voice Agent with LiveKit (YouTube)](https://www.youtube.com/)
- [Build a Voice Agent with Pipecat (YouTube)](https://www.youtube.com/)

## 4. Next Steps

Open the accompanying Jupyter Notebook (`API_Tools_Quickstart.ipynb`) to see hands-on examples of how to initialize and use these APIs using Python.
