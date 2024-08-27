#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py

RAG application using LlamaIndex using indexing data, 
prompt engineering, and queries to OpenAI models.

Index is a data structure that allows quick retrieval of relevant context
for an user query -- it is the core foundation for RAG use-cases in LlamaIndex.
At a high-level, Indexes are built from documents -- pdf, api outputs, etc. --
and are used to build query engines and chat engines to enable Q/A and chat over the document data.
Behind the scenes, Indexes store data in Node objects -- chucks of original data and metadata.
Indexes then expose a retriever interface that supports myriad of additional configuration and automation.
The most common index by far is the VectorStoreIndex.

The following code indexes data from a document, builds a query string using the
context chunks retrieved, and the user query, and fetches a chat completion response from
an OpenAI model.

Langchain RAG prompt templates and custom prompt templates are used for better prompting.
"""

__author__ = "Vikram Singh"
__credits__ = []
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Vikram Singh"
__email__ = "vikramsingh.tech"
__status__ = "Production"

import os
from dotenv import load_dotenv

# Import Path for managing complex file paths.
from pathlib import Path

# For logging and stream handling from llamaIndex.
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Markdown for presenting results.
from IPython.display import Markdown, display

# llama_index comes with various libraries useful for ML and prompting.
# For Vector database for the text embeddings and Prompt templating engine.
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
# Import PyMuPDFReader from llama_index specifically to help read the pdfs. 
from llama_index.readers.file import PyMuPDFReader
# llama_index OpenAI to make it easier to connect and query to OpenAI models.
from llama_index.llms.openai import OpenAI


# Sample wget code to fetch online pdfs.
# wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

# Load the environment variables from .env.
load_dotenv()

# Get the Open AI api key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the pdf reader as loader.
loader = PyMuPDFReader()
# documents is being initialized with the data from the arxiv llama pdf in our data folder.
documents = loader.load(file_path="./data/llama2.pdf")

# Initialize the different OpenAI GPT models to use.
gpt35_llm = OpenAI(model="gpt-3.5-turbo")
gpt4_llm = OpenAI(model="gpt-4")

# Documents are split into chunks and parsed into Node objects.
# Nodes are lightweight abstractions over text strings that keep track of metadata and relationships.
# VectorStoreIndex stores vectors in memory by default and can be made persistent with further customization.
index = VectorStoreIndex.from_documents(documents)

# Sample query based on the sample llama2.pdf.
query_str = "What are the potential risks associated with the use of Llama 2 as mentioned in the context?"

# Setup the query engine from the VectorStoreIndex named index.
# Query engine is a generic interface that allows us to ask questions over the data.
# similarity_top_k is the number of top k results to return.
# Setting llm to the OpenAI model gpt35_llm defined earlier.
query_engine = index.as_query_engine(similarity_top_k=2, llm=gpt35_llm)
# We will use the vector_retriever for testing if needed.
vector_retriever = index.as_retriever(similarity_top_k=2)

# Query engine takes in a natural language query and returns a rich response.
# The query engine is initialized earlier on top of the VectorStoreIndex with similarity and model configurations. 
response = query_engine.query(query_str)
# Print the model response to the query for visualization. 
print(str(response))
