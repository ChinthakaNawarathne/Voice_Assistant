import re
import sys
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from datetime import datetime

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

import config
import search_tool
import voice_handler
import emotion_tracker

# =====================================================================
# 1. DEFINE AGENT STATE
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_emotion: str

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
# Initialize Groq with lower-token model (500K TPD, smaller = cheaper on tokens)
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=config.GROQ_API_KEY,
    temperature=0.1
).bind_tools(tools_list) # Bind tools directly to the model container

def call_agent(state: AgentState):
    """The brain node. Decides whether to reply or trigger an action node."""
    print("🧠 [LangGraph Node]: Thinking...")
    
    current_emotion = state.get("user_emotion", "neutral")
    current_date = datetime.now().strftime("%B %d, %Y")
    
    emotion_guidance = {
        "angry": "The user appears angry or frustrated. Be extra calm, patient, and de-escalating. Acknowledge their frustration softly.",
        "sad": "The user sounds sad or upset. Be gentle, empathetic, and warm in your response.",
        "fearful": "The user sounds anxious or fearful. Reassure them with a calm, confident tone.",
        "happy": "The user sounds happy. Match their positive, upbeat energy.",
        "surprised": "The user sounds surprised. Engage their curiosity with an enthusiastic tone.",
        "disgusted": "The user sounds displeased. Respond diplomatically and neutrally.",
        "neutral": "Respond in your normal helpful, friendly tone.",
    }.get(current_emotion, "Respond in your normal helpful, friendly tone.")
    
    system_prompt = SystemMessage(content=(
        "You are an elite, real-time voice assistant powered by LangGraph.\n"
        f"Context baseline: Today is {current_date}.\n"
        f"User's detected emotional state: {current_emotion.upper()}.\n"
        f"Tone guidance: {emotion_guidance}\n"
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
        "tools": "action_tools", # If Groq wants a tool, route to action_tools
        END: END                 # If Groq is done, terminate graph execution
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
    emotion = emotion_tracker.EmotionTracker()
    stop_words = {"stop", "exit", "enough", "quit", "terminate"}
    
    conversation_state = {"messages": [], "user_emotion": "neutral"}

    voice.speak("System ready.")

    while True:
        try:
            user_input, audio_data = voice.listen_to_user()
            if not user_input:
                continue
                
            if user_input.lower().strip(".,!?") in stop_words:
                voice.speak("Goodbye.")
                break
            
            # Detect emotion from raw voice audio
            user_emotion = "neutral"
            emotion_conf = 0.0
            if audio_data is not None:
                user_emotion, emotion_conf = emotion.analyze(
                    audio_data.frame_data, audio_data.sample_rate, user_input
                )
            print(f"🎭 User Emotion: [{user_emotion.upper()}] ({emotion_conf}% confidence)")
            
            # Store emotion in state for the graph
            conversation_state["user_emotion"] = user_emotion
            
            # Append current turn to the message array payload
            conversation_state["messages"] = add_messages(
                conversation_state["messages"], 
                [HumanMessage(content=user_input)]
            )
            
            # Execute graph via unified invoke sequence
            final_output = app_agent.invoke(conversation_state)
            
            # Update history state tracking safely with compiled output
            conversation_state["messages"] = final_output["messages"]
            
            # Walk backwards to find the last text-only assistant message
            last_message = None
            for msg in reversed(final_output["messages"]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_message = msg
                    break
            if last_message is None:
                continue
            
            # Derive the text content
            if isinstance(last_message.content, list):
                assistant_reply = "".join(
                    [part["text"] for part in last_message.content if isinstance(part, dict) and "text" in part]
                )
            else:
                assistant_reply = str(last_message.content or "")
            
            # Strip tool-call artifacts like <function=xxx>...</function>
            assistant_reply = re.sub(r"<function=[^>]*>.*?</function>", "", assistant_reply).strip()
            
            # Skip non-text responses (tool calls, empty content)
            if not assistant_reply:
                continue

            voice.speak(assistant_reply)
            
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"Runtime Exception: {e}")
            voice.speak("An edge case system error occurred.")

if __name__ == "__main__":
    main()