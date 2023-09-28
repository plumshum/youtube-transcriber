# use langchain to write a summary of the text
# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our chat model. We'll use the default which is gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Prompt templates for dynamic values
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# To create our chat messages
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import os
import openai
from dotenv import load_dotenv
_ = load_dotenv('config.env')

openai.api_key  = os.environ['OPENAI_API_KEY']

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# test video
video_id='TPcXVJ1VSRI'
transcript = YouTubeTranscriptApi.get_transcript(video_id)
content = TextFormatter.format_transcript(transcript,transcript)


llm = ChatOpenAI(temperature=0)

system_template='you are a wonderful youtube video summarizer'
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template='summarize this youtube video: {text}' # {text} is a placeholder for the document_variable_name

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])

chain = load_summarize_chain(llm, chain_type="refine", verbose=True, refine_prompt=chat_prompt)
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
texts = text_splitter.create_documents([content])
output = chain.run({"input_documents": texts})

print(output)