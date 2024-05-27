import speech_recognition as sr
import requests
import json
import os
import pyaudio
import io
import wave

# Initialize PyAudio
audio_player = pyaudio.PyAudio()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import create_openai_functions_agent ,AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
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

from pydub import AudioSegment
import time
from pygame import mixer

# Recognize speech from mic
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    return response

GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
OPENAI_API_KEY = "sk-nzO2bpiEUHY5lwGWd43IT3BlbkFJ9u7o4AmleYcssP7p3VLx7"
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
from pyht import Client, TTSOptions, Format

# Initialize PlayHT API with your credentials
client = Client("FZwyFZ1d5pXBXE4VRklYmETIiyJ3", "4fd5b1fb5ef84afa89fcfe2efbe0ba0d")

# configure your stream
options = TTSOptions(
    voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
    sample_rate=44_100,
    format=Format.FORMAT_MP3,
    speed=0.2,
    temperature=0.7,  # Adjust temperature here (lower value for more predictable results)
    voice_guidance=3  # Adjust voice_guidance here (lower value for a more generic and friendly tone)
)

def process_chat(agentExecutor , user_input , chat_history):
    response= agentExecutor.invoke({
        "input":user_input,
        "chat_history":chat_history
    })
    return response["output"]
if __name__ == "__main__":
    recognizer = sr.Recognizer()


    # Choose an input device (example: device 1)
    mic = sr.Microphone()

    chat_history = []
    running = True

    while running:
        result = recognize_speech_from_mic(recognizer, mic)
            
        if result["transcription"]:
            user_input = result["transcription"].strip().lower()
            print(f"You: {user_input}")

            if user_input in ['exit', 'bye']:
                print("Exiting the chat. Goodbye!")
                break
            else:
                # Placeholder function for processing chat
                response = process_chat(agentExecutor, user_input, chat_history)
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=response))
                
                print(f"V-Assistant: {response}")

                # Generate speech from response using PlayHT API
                text_to_speech = response
                audio_stream = io.BytesIO()
                for chunk in client.tts(text=text_to_speech, voice_engine="PlayHT2.0-turbo", options=options):
                    audio_stream.write(chunk)

            # Reset stream position to the beginning
                audio_stream.seek(0)

                # Save audio to file
                output_filename = "output.mp3"
                with open(output_filename, 'wb') as audio_file:
                    audio_file.write(audio_stream.getvalue())
                print(f"Audio saved to {output_filename}")

                device_index = 2
                # mixer.pre_init(devicename="Headset Microphone (Hi Res USB-C AUDIO)")
                mixer.init()
                mixer.music.load("D:\\tickets_data_2024\\sql_bot\\output.mp3")
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)
        elif result["error"]:
            print(f"ERROR: {result['error']}")

# Terminate PyAudio
audio_player.terminate()