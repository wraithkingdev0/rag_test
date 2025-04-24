# LangSmith Setup Guide

LangSmith is LangChain's observability platform that helps you:
- 🔍 **Debug** your LLM applications
- 📊 **Monitor** performance and costs
- 📈 **Track** all LLM calls and traces
- 🧪 **Test** and evaluate your prompts

## Step 1: Install LangSmith

```bash
pip install langsmith
```

Or it's already in your `requirements.txt` - just run:
```bash
pip install -r requirements.txt
```

## Step 2: Get Your API Key

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in (it's free!)
3. Go to Settings → API Keys
4. Create a new API key
5. Copy the key

## Step 3: Add to Your .env File

Add these lines to your `.env` file:

```env
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=rag-langs  # Optional: name your project
```

**Important Notes:**
- Replace `your_api_key_here` with your actual API key
- `LANGCHAIN_PROJECT` is optional - it helps organize traces in LangSmith
- You can use different project names for different scripts (e.g., `rag-langchain`, `rag-langgraph`)

## Step 4: Run Your Code

That's it! Just run your scripts as normal:

```bash
python rag_langchain.py
# or
python rag_langgraph.py
```

All LLM calls will automatically be traced and sent to LangSmith.

## What You'll See in LangSmith

1. **Traces**: Every run of your chain/graph creates a trace
2. **Spans**: Each step (retrieve, context, generate) is a span
3. **Inputs/Outputs**: See exactly what went in and came out
4. **Timing**: See how long each step took
5. **Costs**: Track API costs per call
6. **Errors**: See any errors that occurred

## Viewing Your Traces

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Click on "Traces" in the sidebar
3. You'll see all your runs listed
4. Click on any trace to see the detailed execution flow

## Advanced: Custom Project Names

You can set different project names in your code:

```python
import os
os.environ["LANGCHAIN_PROJECT"] = "my-custom-project"
```

Or set it per script in your `.env` file.

## Troubleshooting

**No traces showing up?**
- Check that `LANGCHAIN_TRACING_V2=true` (must be lowercase "true" as a string)
- Verify your API key is correct
- Make sure you're connected to the internet

**Want to disable tracing temporarily?**
- Set `LANGCHAIN_TRACING_V2=false` or remove it from `.env`

## Benefits

✅ **Automatic**: No code changes needed - just set env vars
✅ **Free tier**: Generous free tier for personal use
✅ **Works with both**: LangChain chains and LangGraph graphs
✅ **Detailed**: See every LLM call, token usage, and timing

