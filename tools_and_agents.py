from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import create_openai_functions_agent ,AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from pinecone import Pinecone
from langchain_core.messages import HumanMessage , AIMessage
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq

GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embeddings = OpenAIEmbeddings()
model = ChatGroq(temperature=0.5, model_name="llama3-70b-8192")
TAVILY_API_KEY = "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"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
# model = ChatOpenAI(model="gpt-3.5-turbo-1106" , temperature = 0.5)
PC_API_KEY = "529d6a46-a7cf-46a5-a7a1-398fd0a08284"
os.environ["PINECONE_API_KEY"] = PC_API_KEY
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


index_name = "mixeddata"

# connect to index
index = pc.Index(index_name)

# view index stats

pinecone = PineconeVectorStore(
     embedding=embeddings, index_name=index_name
)
retriever = pinecone.as_retriever()
prompt = ChatPromptTemplate.from_messages([
SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a friendly customer service bot and provide answer from given context')),
 MessagesPlaceholder(variable_name='chat_history', optional=True),
 HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
 MessagesPlaceholder(variable_name='agent_scratchpad')
])

search = TavilySearchResults()

retriever_agent = create_retriever_tool(
    retriever , 
    "Support Ticket - Data Sync Errors",  # Include "Data Sync Errors" in the title for clarity
  """
  Search for information about errors you encounter during data synchronization. 

  Use this tool for questions related to:
    * Data sync errors (e.g., phase 1 errors, data corruption)
    * Troubleshooting data sync issues
    * Solutions for common data sync problems

  For any questions about general support or errors outside of data sync, please refer to the appropriate channels.
  """,
)

tools = [search , retriever_agent]

agent = create_tool_calling_agent(
    llm=  model,
    prompt = prompt,
    tools = tools,

)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
) 

def process_chat(agentExecutor , user_input , chat_history):
    response= agentExecutor.invoke({
        "input":user_input,
        "chat_history":chat_history
    })
    return response["output"]

if __name__ == "__main__":
    chat_history = []
    running = True  # Initialize the flag to True

    while running:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'bye']:
            print("Exiting the chat. Goodbye!")
            running = False  # Set the flag to False to exit the loop
        else:
            response = process_chat(agentExecutor, user_input, chat_history)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            print(f"V-Assistant: {response}")

