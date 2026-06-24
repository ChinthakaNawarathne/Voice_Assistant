import sys
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from datetime import datetime

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition

import config
import search_tool
import voice_handler

# =====================================================================
# 1. DEFINE AGENT STATE
# =====================================================================
class AgentState(TypedDict):
    # The 'add_messages' annotation ensures new messages are appended to the list,
    # preserving the agent's short-term conversation memory automatically.
    messages: Annotated[Sequence[BaseMessage], add_messages]

# =====================================================================
# 2. DEFINE NATIVE TOOLS
# =====================================================================
@tool
def web_search_tool(query: str) -> str:
    """Queries the live web via Tavily search to discover real-time factual data.
    Use this tool whenever the user asks about live events, current cricket match results,
    weather conditions, or fresh news updates.
    """
    print(f"🕵️‍♂️ [LangGraph Tool]: Running web search for '{query}'...")
    return search_tool.search_web(query)

@tool
def system_clock_tool() -> str:
    """Retrieves the precise local time and calendar date from the host device.
    Use this tool whenever the user inquires about the current time, today's date, or scheduling.
    """
    now_time = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    print(f"🕵️‍♂️ [LangGraph Tool]: System clock extracted: {now_time}")
    return f"The current exact time and date is {now_time}."

tools_list = [web_search_tool, system_clock_tool]
tool_node = ToolNode(tools_list)

# =====================================================================
# 3. DEFINE NODES & COGNITIVE LOGIC
# =====================================================================
# Initialize Gemini using LangChain wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    google_api_key=config.GEMINI_API_KEY,
    temperature=0.1
).bind_tools(tools_list) # Bind tools directly to the model container

def call_agent(state: AgentState):
    """The brain node. Decides whether to reply or trigger an action node."""
    print("🧠 [LangGraph Node]: Thinking...")
    
    current_date = datetime.now().strftime("%B %d, %Y")
    system_prompt = SystemMessage(content=(
        "You are an elite, real-time voice assistant powered by LangGraph.\n"
        f"Context baseline: Today is {current_date}.\n"
        "Analyze the user's input. If answering requires a tool, call it instantly.\n"
        "CRITICAL: Keep your final response restricted to 1-2 highly conversational sentences max."
    ))
    
    # Prepend the system instructions to the ongoing message array stream
    messages_payload = [system_prompt] + state["messages"]
    response = llm.invoke(messages_payload)
    
    return {"messages": [response]}

# =====================================================================
# 4. BUILD THE GRAPH PIPELINE
# =====================================================================
workflow = StateGraph(AgentState)

# Define our functional runtime nodes
workflow.add_node("agent_brain", call_agent)
workflow.add_node("action_tools", tool_node)

# Map paths out using entrypoints and conditional routing
workflow.add_edge(START, "agent_brain")

# tools_condition is a built-in routing mechanism that checks if the model 
# wanted to run a tool, or if it's ready to terminate and respond.
workflow.add_conditional_edges(
    "agent_brain",
    tools_condition,
    {
        "tools": "action_tools", # If Gemini wants a tool, route to action_tools
        END: END                 # If Gemini is done, terminate graph execution
    }
)

# Loop back to brain after tool execution so it can read the results
workflow.add_edge("action_tools", "agent_brain")

# Compile the final application topology
app_agent = workflow.compile()

# =====================================================================
# 5. EXECUTION ENTRYPOINT
# =====================================================================
def main():
    print("==================================================")
    print("🦜 LangGraph Stateful Agent App Online")
    print("==================================================")
    
    voice = voice_handler.VoiceHandler()
    stop_words = {"stop", "exit", "enough", "quit", "terminate"}
    
    # Store persistent state messages list locally for conversation history continuity
    conversation_state = {"messages": []}

    voice.speak("System ready.")

    while True:
        try:
            user_input = voice.listen_to_user()
            if not user_input:
                continue
                
            if user_input.lower().strip(".,!?") in stop_words:
                voice.speak("Goodbye.")
                break
            
            # Append current turn to the message array payload
            conversation_state["messages"] = add_messages(
                conversation_state["messages"], 
                [HumanMessage(content=user_input)]
            )
            
            # Execute graph via unified invoke sequence
            final_output = app_agent.invoke(conversation_state)
            
            # Update history state tracking safely with compiled output
            conversation_state["messages"] = final_output["messages"]
            
            # Extract the raw response message object
            last_message = final_output["messages"][-1]
            
            # FIX: Safely parse LangChain/Gemini content layout down to a clean string
            if isinstance(last_message.content, list):
                # If the content arrives as a structured list containing text dicts
                assistant_reply = "".join(
                    [part["text"] for part in last_message.content if isinstance(part, dict) and "text" in part]
                )
            else:
                assistant_reply = str(last_message.content)
            
            # Fallback if text data extraction fails
            if not assistant_reply.strip():
                assistant_reply = "I process that successfully, but generated empty speech text."

            voice.speak(assistant_reply)
            
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"Runtime Exception: {e}")
            voice.speak("An edge case system error occurred.")

if __name__ == "__main__":
    main()