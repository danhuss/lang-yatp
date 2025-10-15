# from typing import TypedDict, Annotated
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages

# class YatpState(TypedDict):
#     messages: Annotated[list, add_messages]
import time
from dotenv import load_dotenv
from typing import Optional
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_ollama import ChatOllama

def activity_search(city: str, likes: Optional[str] = None, dislikes: Optional[str] = None) -> str:
    """Tool to search for activities in a given city. Also accepts arguments for likes and dislikes."""
    return f"{city} is known for it's sandy beaches and vibrant nightlife."

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def create_model_with_retry(model_name: str = "llama3.1:8b", max_retries: int = 3):
    """Create model with retry logic"""
    for attempt in range(max_retries):
        try:
            model = ChatOllama(
                model=model_name,
                temperature=0.7,
                request_timeout=240.0  # Increase timeout for reliability
            )
            # Test the model with a simple call
            model.invoke("Hi")
            print(f"✓ Model '{model_name}' loaded successfully")
            return model
        except Exception as e:
            print(f"⚠ Attempt {attempt + 1}/{max_retries} failed to load model: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise Exception(f"Failed to load model after {max_retries} attempts. Is Ollama running?")


def main():
    user_input = None
    load_dotenv()
    model = create_model_with_retry(model_name="gpt-oss")

    activity_agent = create_react_agent(
        model=model,
        tools=[activity_search],
        prompt="Suggest an activity in a city based on user preferences.",
        name="activity_agent"
    )

    hotel_agent = create_react_agent(
        model=model,
        tools=[book_hotel],
        prompt="You are a hotel booking assistant.",
        name="hotel_agent"
    )

    flight_agent = create_react_agent(
        model=model,
        tools=[book_flight],
        prompt="You are a flight booking assistant.",
        name="flight_agent"
    )

    supervisor_agent = create_supervisor(
        agents=[activity_agent, hotel_agent, flight_agent],
        model=model,
        output_mode="full_history",
        prompt="""You manage a small travel agency. Your goal is to help users decide on and plan their trips. 
You have three agents assistants: for general questions use activity_agent, for booking flights use flight_agent, 
and for booking hotels use hotel_agent. Start by answering any questions the user has about travel or their 
destination. Then, if the user wants to book a flight or hotel, delegate to the appropriate agent. Make sure 
you have all the information needed before booking flights and hotels like the dates, location, etc. Always 
confirm with the user before booking anything."""
    ).compile()

    # png_data = supervisor_agent.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)

    print("Welcome to the Travel Agency! Type 'quit' or 'exit' to end.")
    conversation_state = {"messages": []}

    while True:
        user_input = input("> ").strip()

        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'q', 'bye', 'goodbye']:
            print("Goodbye!")
            break
        print("Processing your request... (press Ctrl+C to cancel)")
        conversation_state["messages"].append({"role": "user", "content": user_input})

        try:
            for event in supervisor_agent.stream(
                conversation_state,
                {"recursion_limit": 50}
            ):
                # print(event)
                # print("-----")
                for value in event.values():
                    if value is None:
                        continue
                    if "messages" not in value:
                        continue
                    messages = value.get("messages") 
                    if not messages or len(messages) == 0:
                        continue
                    conversation_state["messages"] = messages
                    last_message = messages[-1]
                    agent_name = getattr(last_message, 'name', None) or 'Assistant'
                    content = None
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        content = last_message['content']
                    
                    if content and isinstance(content, str) and content.strip():
                        print(f"\n[{agent_name}]: {content}")
                    
                    # Debug: Show tool calls if present (optional)
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            print(f"  → Calling tool: {tool_call.get('name', 'unknown')}")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

#     result = supervisor_agent.invoke(
#         {"messages": [{"role": "user", "content": user_input}]}
# #        {"messages": [{"role": "user", "content": "I want to visit a Miami FL, what should I do? Then go ahead and book a flight from LA to Miami and a stay at Motel 6."}]}
#     )

    # print(result)
    # print(result.get('messages')[-1].content)

if __name__ == "__main__":
    main()
