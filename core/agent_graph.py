from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from typing import TypedDict, Sequence, Annotated
from operator import add as add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    prev_context: str | None

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

def create_graph(llm, tools_dict, system_prompt: str, decider_llm):
    # --- CONTEXT DECIDER ---
    def context_decider(state: AgentState) -> AgentState:
        """Decides if context should be reset or kept."""
        query = state["messages"][-1].content

        decision = decider_llm.invoke([
                SystemMessage(content=(
                    """You are a context decider.

                    Your task is to classify the user's latest query into one of three categories:
                    - RESET → if the query is unrelated to the past conversation.
                    - KEEP → if the query depends on or continues the past conversation.
                    - CHITCHAT → if the query is casual small talk or unrelated to the task.

                    Respond with exactly one word: RESET, KEEP, or CHITCHAT.
                    Do not include anything else in your response."""
                )),
                HumanMessage(content=query)
            ])


        mode = decision.content.strip().upper()

        if mode in ["RESET"]:
            return {"prev_context": mode}
        if mode in ["CHITCHAT"]:
            return {"prev_context": mode}
        else:
            return {"prev_context": mode}
          
    # --- LLM CALL ---
    def call_llm(state: AgentState) -> AgentState:
        print(f"\nThe state of prev context is: {state['prev_context']}\n")
        messages = [SystemMessage(content=system_prompt.format(context_mode=state['prev_context']))] + list(state['messages'])
        message = llm.invoke(messages)
        return {'messages': [message]}

    # Tool execution    
    def take_action(state: AgentState) -> AgentState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provided')}")
            if t['name'] not in tools_dict:
                print(f"\nTool {t['name']} does not exist.")
                result = "Incorrect tool name. Please select a valid tool."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query',''))
                print(f"Result length: {len(str(result))}")
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        print("Tools execution complete.")
        return {'messages': results}

    # --- STATE GRAPH ---
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_node("context_decider", context_decider)

    graph.add_edge("context_decider", "llm")
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("context_decider")
    return graph.compile()
