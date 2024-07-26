"""
Primary entry point for the RAG app. Integrates other RAG functionality into a UI
"""
import os
import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from graph import GraphBuilder
from rag import RAG
from pdfreader import SimplePDFViewer, PageDoesNotExist
from loguru import logger
import pandas as pd
from io import StringIO
import sys


def load_config():
        with open("src/config.yaml", 'r') as config_file:
            return yaml.safe_load(config_file)
        
load_dotenv()
config = load_config()
# Neo4j Client Setup
os.environ["NEO4J_URI"] = config['graph']["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = os.getenv('NEO4J_USERNAME')
os.environ["NEO4J_PASSWORD"] = os.getenv('NEO4J_PASSWORD')

#openAI API 
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')



class RAGApp:
    """
    Class to encapsulate all methods required to query a graph for retrieval augmented generation
    """
    def __init__(self):
        self.config = config
        self.rag = RAG()
        logger.info("Waiting for user action to create knowledge Graph")
        self.init_ui()
        
        

        

        


    def content_to_graph(self, sources, format, status_text, progress_bar):
        """Populate the graph with content."""
        logger.info("Building graph from content")


        if sources is not None:
            try:
                if format == 'pdf':
                    logger.info("PDF Extracting")
                    status_text.text("PDF Extracting")
                    self.rag.graph_builder.extract_pdf_content(sources)
                    status_text.text("Finished PDF Extraction")
                    logger.info("Populating graph with knowledge from sources")
                elif format == 'links':
                    steps = len(sources)+1
                    for count, source in enumerate(sources):
                        try:
                            step = count+1
                            progress_bar.progress(step/steps)
                            if 'youtube' in source:
                                logger.info("YouTube Extracting")
                                status_text.text("YouTube Extracting")
                                self.rag.graph_builder.extract_youtube_transcripts(source)
                                status_text.text("Completed YouTube Extraction")
                                logger.info("Completed YouTube Extraction")
                                
                            else:
                                logger.info("Wikipedia Extracting")
                                status_text.text("Wikipedia Extracting")
                                self.graph = self.rag.graph_builder.extract_wikipedia_content(source)
                                status_text.text("Completed Wikipedia Extraction")
                                logger.info("Completed Wikipedia Extraction")
                                
                        except Exception as e:
                            print(f"Error processing source {source}: {e}")
                            logger.error(f"Error processing source {source}: {e}")

                    progress_bar.progress(step/steps)
                    logger.info("Populating graph with knowledge from sources")
                    
                    step += 1
                    progress_bar.progress(step/steps)    

                else:
                    raise ValueError("Unsupported format provided.")
                
                
            except Exception as e:
                logger.error(f"Error building graph: {e}")
                status_text.text(f"Error building graph: {e}")
        else:
            logger.warning("No sources provided to populate the graph.")
            status_text.text("No sources provided to populate the graph.")

    def read_pdf(self, uploaded_pdf_file):
            pdf_text = ""
            viewer = SimplePDFViewer(uploaded_pdf_file)
            try:
                while True:
                    viewer.render()
                    pdf_text += "".join(viewer.canvas.strings)
                    viewer.next()
            except PageDoesNotExist:
                pass
            return pdf_text
    
 

    def submit(self, key, value):
        st.session_state[key] = value

    def graph_management(self):
        with st.sidebar:
            st.header("Graph Management")
            st.write("Below are options to populate and reset your graph database")



            # Initialize session state variables
            if "links" not in st.session_state:
                st.session_state.links = []
            if "txt_file" not in st.session_state:
                st.session_state.txt_file = None
            if "pdf_text" not in st.session_state:
                st.session_state.pdf_text = None

            # Load link button and input
            with st.form(key='link_form'):
                text_input = st.text_input("Enter a YouTube-web-link or Wikipedia-topic-name:", key="link_input")
                submit_link = st.form_submit_button(label='Load link')
                if submit_link and text_input:
                    st.session_state.links.append(text_input)
                    st.write(f"Link added: {text_input}")

            # Upload txt file
            uploaded_txt_file = st.file_uploader("Upload a txt file with multiple Youtube-links or Wikipedia-topics-names", type="txt")
            if uploaded_txt_file:
                stringio = StringIO(uploaded_txt_file.getvalue().decode('utf-8'))
                txt_lines = stringio.readlines()
                for line in txt_lines:
                    st.write(line.strip())
                    st.session_state.links.append(line.strip())

            # Upload PDF text file
            uploaded_pdf_file = st.file_uploader("Upload language content from a PDF file", type="pdf")
            if uploaded_pdf_file:
                pdf_text = self.read_pdf(uploaded_pdf_file)
                st.session_state.pdf_text = pdf_text
                st.write("PDF file uploaded successfully.")

            # Populate graph button
            if st.button("populate graph"):
                if st.session_state.pdf_text:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    self.content_to_graph(st.session_state.pdf_text, 'pdf',st.empty(),progress_bar)
                if st.session_state.links:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    self.content_to_graph(st.session_state.links, 'links', st.empty(),progress_bar)
                st.write("Graph populated with provided data.")
                logger.info("Graph populated with provided data.")
            


            # Reset graph button
            if st.button("reset graph"):
                self.reset_graph()
                logger.info("Graph reseted")
                st.session_state.links = []
                st.session_state.txt_file = None
                st.session_state.pdf_text = None
                self.vector_index = []
                logger.info("Vector Index reseted")
                st.write("Graph reseted")

        #evaluator = RAGASEvaluator()
        #results = evaluator.evaluate_pipeline()
        #st.write("Performance Evaluation Results:")
        #st.write(results)

    def reset_graph(self):
        """
        Will reset the graph by deleting all relationships and nodes
        """
        self.rag.graph_builder = GraphBuilder()

    def get_response(self, question: str) -> str:
        """
        For the given user question will formulate a search query and use a custom RAG retriever 
        to fetch related content from the knowledge graph. 
        You can use the context for conjectures. However those conjectures must align with all possible statements from the context.
        But make sure not to halluscinate and stay as close to truth as possible. 

        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The results of the invoked graph based question
        """
        self.rag = RAG()
        logger.info(f"entity_extract_chain:  {self.rag.entity_extract_chain}")
        search_query = self.rag.create_search_query(st.session_state.chat_history, question)
        logger.info(f"search_query:  {search_query}")

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""

        logger.info(f"template:  {template}")

        prompt = ChatPromptTemplate.from_template(template)

        logger.info(f"prompt:  {prompt}")
        

        chain = (
            RunnableParallel(
            {
                "context": lambda x: self.rag.retriever(search_query, self.vector_index),
                "question": RunnablePassthrough(),
            }
            )
            | prompt
            | self.rag.llm
            | StrOutputParser()
        )

        logger.info(f"chat_history:  {st.session_state.chat_history}")
        log_context = self.rag.retriever(search_query, self.vector_index)
        logger.info(f"context:  {log_context}")

        # Using invoke method to get response
        response = chain.invoke({"chat_history": st.session_state.chat_history, "question": question})
        logger.info(f"response:  {response}")
        return response
    
    def user_utterance(self):

        user_query = st.chat_input("Ask a question about the text sources....")
        logger.info(f"user_query:  {user_query}")

        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
        
            with st.chat_message("Human"):
                st.write(user_query)

            # Get response from the assistant
            with st.chat_message("AI"):
                logger.info("Get response from the assistant")
                with st.spinner("Loading"):
                    ai_response = self.get_response(user_query)
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=ai_response))
                    new_ai_message = {"role":"assistant","content": ai_response}
                    st.session_state.chat_history.append(new_ai_message)
        


    def init_ui(self):
        """
        Primary entry point for the app. Creates the chat interface that interacts with the LLM. 
        """
        st.set_page_config(page_title="Langchain RAG Bot", layout="wide")
        st.title("Knowledge Retrieval Assistant")


        if "links" not in st.session_state:
            st.session_state.links = []
        if "txt_file" not in st.session_state:
            st.session_state.txt_file = None
        if "pdf_text" not in st.session_state:
            st.session_state.pdf_text = None

    
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
            AIMessage(content="What would you like to know from the content?")
            ]

        self.graph_management()
        self.vector_index = self.rag.graph_builder.index_graph()
        logger.info(f"vector_index:  {self.vector_index}")
        
        user_query = st.chat_input("Ask a question about the text sources....")
        logger.info(f"user_query:  {user_query}")

        if user_query is not None and user_query != "":
            with st.spinner("Loading"):
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                logger.info("query added to chat_history")

                for message in st.session_state.chat_history: 
                    if isinstance(message, HumanMessage):
                        with st.chat_message("Human"):
                            st.write(message.content)
                            logger.info(f"Human question return : {message.content}")

                ai_response = self.get_response(user_query)
                st.session_state.chat_history.append(AIMessage(content=ai_response))
                logger.info("response added to chat_history")


        for message in st.session_state.chat_history: 
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
                    logger.info(f"AI response return: {message.content}")

                    

        
    

if __name__ == "__main__":
    logger.info("Starting to initialize Retrieval architecture")
    app = RAGApp()
