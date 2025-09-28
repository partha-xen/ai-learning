import warnings
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Suppress urllib3 warnings
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")


sales = pd.DataFrame(
    [
        {
            "date": "2025-01-02",
            "region": "West",
            "product": "Widget",
            "units": 120,
            "price": 9.5,
        },
        {
            "date": "2025-01-03",
            "region": "West",
            "product": "Gadget",
            "units": 80,
            "price": 12.0,
        },
        {
            "date": "2025-01-03",
            "region": "East",
            "product": "Widget",
            "units": 200,
            "price": 9.0,
        },
        {
            "date": "2025-01-05",
            "region": "South",
            "product": "Gizmo",
            "units": 50,
            "price": 15.0,
        },
        {
            "date": "2025-01-05",
            "region": "East",
            "product": "Gadget",
            "units": 140,
            "price": 11.5,
        },
    ]
)
sales["date"] = pd.to_datetime(sales["date"])
sales["revenue"] = sales["units"] * sales["price"]
sales.head()


# 1) define tools over your DataFrame
@tool
def top_products_by_revenue(n: int = 3) -> str:
    """Return top-N products by total revenue across all regions."""
    n = int(n)
    totals = sales.groupby("product")["revenue"].sum().sort_values(ascending=False)
    return totals.head(n).to_string()


@tool
def revenue_for(region: str) -> str:
    """Return total revenue for a given region."""
    region = str(region).strip()
    total = sales.loc[sales["region"].str.lower() == region.lower(), "revenue"].sum()
    return f"{region}: {total:.2f}"


tools = [top_products_by_revenue, revenue_for]

# 2) choose an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Use the original memory but suppress the deprecation warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*migration guide.*")
    warnings.filterwarnings("ignore", message=".*ConversationBufferMemory.*")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 3) give the agent a prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data analyst. Use tools if needed; otherwise answer directly.",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# # 4) create an agent that uses OpenAI-style function/tool-calls
# agent = create_openai_functions_agent(llm, tools, prompt)
# ----------------------------
# 4. Create ReAct Agent with AgentExecutor to capture intermediate steps
# Note: LangChain agents are being deprecated in favor of LangGraph
# This code still works but consider migrating to LangGraph for new projects
# ----------------------------

# Suppress the deprecation warning for this specific call
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*LangChain agents.*")
    warnings.filterwarnings("ignore", message=".*LangGraph.*")

    # Create the base agent
    base_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # 👈 ReAct style
        memory=memory,
        verbose=True,  # 👈 Prints agent_scratchpad (Thought/Action/Observation)
    )

# Wrap in AgentExecutor to capture intermediate steps (agent scratchpad)
agent_exec = AgentExecutor(
    agent=base_agent.agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,  # 👈 This captures the scratchpad
)

print("\n=== First Query ===")
query1 = "What are the top 2 products by revenue?"
result1 = agent_exec.invoke({"input": query1})
print("Final Answer:", result1["output"])

# Print agent scratchpad for first query
print("\n===== Agent Scratchpad (First Query) =====")
for i, (action, observation) in enumerate(result1["intermediate_steps"]):
    print(f"\nStep {i+1}:")
    print(f"Thought: Agent decided to use tool '{action.tool}'")
    print(f"Action: {action.tool} with input {action.tool_input}")
    print(f"Observation: {observation}")

print("\n=== Second Query ===")
query2 = "What is the total revenue for West?"
result2 = agent_exec.invoke({"input": query2})
print("Final Answer:", result2["output"])

# Print agent scratchpad for second query
print("\n===== Agent Scratchpad (Second Query) =====")
for i, (action, observation) in enumerate(result2["intermediate_steps"]):
    print(f"\nStep {i+1}:")
    print(f"Thought: Agent decided to use tool '{action.tool}'")
    print(f"Action: {action.tool} with input {action.tool_input}")
    print(f"Observation: {observation}")

# # Inspect chat history
# print("\n===== chat_history =====")
# print(memory.load_memory_variables({})["chat_history"])

# ----------------------------
# 6. Inspect Chat History
# ----------------------------
print("\n===== chat_history =====")
for m in memory.load_memory_variables({})["chat_history"]:
    print(f"{m.type.upper()}: {m.content}")
