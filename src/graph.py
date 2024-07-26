"""
Script responsible for build a knowledge graph using
Neo4j from unstructured text
"""

import os
from typing import List
import yaml
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader, TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
#os.environ["NEO4J_AUTH"]=None
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "secret"

class GraphBuilder():
    """
    Encapsulates the core functionality requires to build a full knowledge graph 
    from multiple sources of unstructured text

    _extended_summary_
    """
    def __init__(self):
        self.config = self.load_config()
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(model=self.config['hyperparameters']['generator_model'], temperature=self.config['hyperparameters']['temperature'])

    def load_config(self):
        with open("src/config.yaml", 'r') as config_file:
            return yaml.safe_load(config_file)

    def chunk_document_text(self, raw_docs):
        """
        Accepts raw text context extracted from source and applies a chunking 
        algorithm to it. 

        Args:
            raw_docs (str): The raw content extracted from the source

        Returns:
            List: List of document chunks
        """
        text_splitter = TokenTextSplitter(chunk_size=self.config['hyperparameters']['chunk_size'], chunk_overlap=self.config['hyperparameters']['chunk_overlap'])
        docs = text_splitter.split_documents(raw_docs[:3])
        return docs

    def graph_document_text(self, text_chunks):
        """
        Uses experimental LLMGraphTransformer to convert unstructured text into a knowledge graph

        Args:
            text_chunks (List): List of document chunks
        """
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        graph_docs = llm_transformer.convert_to_graph_documents(text_chunks)
        self.graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True
        )

    def chunk_and_graph(self, raw_docs):
        """
        Breaks the raw text into chunks and converts into a knowledge graph

        Args:
            raw_docs (str): The raw content extracted from the source
        """
        text_chunks = self.chunk_document_text(raw_docs)
        if text_chunks is not None:
            self.graph_document_text(text_chunks)


    def extract_youtube_transcript(self, url) -> List:
        """
        Uses the Langchain interface to extract YouTube transcript from
        the specified URL. Under the hood this uses youtube-transcript-api

        Args:
            url (str): URL of the YouTube video to fetch transcript for

        Returns:
            List: Extracted transcript documents
        """
        return YoutubeLoader.from_youtube_url(url).load()

    def extract_youtube_transcripts(self, urls):
        """
        Extracts multiple URLs YouTube transcripts

        Args:
            urls (List): A list of YouTube urls
        """
        for url in urls:
            transcript = self.extract_youtube_transcript(url)
            self.chunk_and_graph(transcript)

    def extract_pdf_content(self, pdf_text):
        """
        Uses the pdf uploaded file to extract text

        Args:
            pdf_text (str): The provided pdf text
        """
        
        self.chunk_and_graph(pdf_text)

    def extract_wikipedia_content(self, search_query):
        """
        Uses the search query and LangChain interface to extract 
        content from the results of a Wikipedia search

        Args:
            search_query (str): The query to search for Wikipedia content on
        """
        raw_docs = WikipediaLoader(query=search_query).load()
        self.chunk_and_graph(raw_docs)

    def graph_text_content(self, path):
        """
        Provided with a text document, will extract and chunk the text
        before generating a graph

        Args:
            path (str): Text document path
        """
        text_docs = TextLoader(path).load()
        print(text_docs)
        self.chunk_and_graph(text_docs)

    def graph_text_documents(self, paths):
        """
        Provided with an array of text documents will extract and
        graph each of them

        Args:
            paths (List): Document paths to extract and graph
        """
        for path in paths:
            self.graph_text_content(path)

    def index_graph(self) -> Neo4jVector:
        """
        Uses the existing graph to create a vector index. This vector representation
        is based off the properties specified. 

        Returns:
            Neo4jVector: The vector representation of the graph nodes specified in the configuration
        """
        vector_index = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(),
            url=self.config['graph']["NEO4J_URI"],
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD'),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index

        

    def reset_graph(self):
        """
        WARNING: Will clear entire graph, use with caution
        """
        self.graph.query(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )
