from dotenv import load_dotenv

load_dotenv()   
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
        
    


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
# llm = ChatOllama(model="gemma4:e4b")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in Israel on linkedin that are accepting new applications and list their details.")]})
    print(result)


if __name__ == "__main__":
    main()
