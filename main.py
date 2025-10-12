# from typing import TypedDict, Annotated
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages

# class YatpState(TypedDict):
#     messages: Annotated[list, add_messages]
import sys
import os
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

def main():
    user_input = None
    load_dotenv()
    model = ChatOllama(model="llama3.1:8b", request_timeout=300)

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
        prompt="""You manage a small travel agency. Your goal is to help users decide on and plan their trips. 
You have three agents assistants: an activity search agent, flight booking agent, and hotel booking agent. Start by 
answering any questions the user has about travel or their destination. Then, if the user wants to book a flight or hotel,
delegate to the appropriate agent. Make sure you have allthe information needed before booking flights and hotels like the dates, location, etc. 
Always confirm with the user before booking anything."""
    ).compile()

    # png_data = supervisor_agent.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)

    if user_input is None:
        print("Please enter your travel request:")
        user_input = input("> ")

    if not user_input.strip():
        print("No input provided. Exiting.")
        sys.exit(0)

    for event in supervisor_agent.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if "messages" in value and value["messages"]:
                last_message = value["messages"][-1]
                if hasattr(last_message, 'content'):
                    print("Assistant:", last_message.content)

#     result = supervisor_agent.invoke(
#         {"messages": [{"role": "user", "content": user_input}]}
# #        {"messages": [{"role": "user", "content": "I want to visit a Miami FL, what should I do? Then go ahead and book a flight from LA to Miami and a stay at Motel 6."}]}
#     )

    # print(result)
    # print(result.get('messages')[-1].content)

if __name__ == "__main__":
    main()
