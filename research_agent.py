from typing import TypedDict, List
import operator
from tools import rag_search, rag_search_filter, web_search, final_answer, run_oracle, tools
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# Map tool names to their corresponding functions
tool_str_to_func = {
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "web_search": web_search,
    "final_answer": final_answer
}

# Define the AgentState TypedDict
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: List[AgentAction]

# Define the router function to determine the next tool to use
def router(state: AgentState):
    # Check if intermediate_steps is a non-empty list
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        last_action = state["intermediate_steps"][-1]
        return last_action.tool
    else:
        # If the format is invalid, default to 'final_answer'
        print("Router invalid format")
        return "final_answer"

# Define the function to run the selected tool
def run_tool(state: AgentState):
    # Retrieve the last action from intermediate_steps
    last_action = state["intermediate_steps"][-1]
    tool_name = last_action.tool
    tool_args = last_action.tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    # Run the tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    # Append the new action to intermediate_steps
    state["intermediate_steps"].append(action_out)
    return state

# Build the state graph
def build_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes to the graph
    graph.add_node("oracle", run_oracle)
    graph.add_node("rag_search_filter", run_tool)
    graph.add_node("rag_search", run_tool)
    graph.add_node("web_search", run_tool)
    graph.add_node("final_answer", run_tool)
    
    # Set the entry point of the graph
    graph.set_entry_point("oracle")
    
    # Add conditional edges based on the router function
    graph.add_conditional_edges(
        source="oracle",
        path=router
    )
    
    # Create edges from each tool back to the oracle
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")
    
    # If the final_answer tool is called, move to END
    graph.add_edge("final_answer", END)
    
    # Compile the graph into a runnable
    runnable = graph.compile()
    return graph, runnable

# Now, you can call build_graph() to get the graph and runnable
graph, runnable = build_graph()
