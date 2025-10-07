# from typing import TypedDict, Annotated
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages

# class YatpState(TypedDict):
#     messages: Annotated[list, add_messages]

from typing import Optional
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

def activity_search(city: str, likes: Optional[str] = None, dislikes: Optional[str] = None) -> str:
    """Tool to search for activities in a given city. Also accepts arguments for likes and dislikes."""
    return f"{city} is known for it's sandy beaches and vibrant nightlife."

model = ChatOllama(model="llama3.2")

agent = create_react_agent(
    model=model,
    tools=[activity_search],
    prompt="Suggest an activity in a city based on user preferences."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "I want to visit a Miami FL, what should I do?"}]}
)

print(result.get('messages')[-1].content)

# def main():
#     print("Hello from lang-yatp!")


# if __name__ == "__main__":
#     main()
