import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Handle empty chunks and batch processing for Gemini embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Validate we have text chunks to process
    if not text_chunks or len(text_chunks) == 0:
        raise ValueError("No text chunks provided for embedding")

    # Process in batches of 100 with empty chunk validation
    batch_size = 100
    vectorstore = None
    valid_chunks = [chunk for chunk in text_chunks if chunk.strip()]

    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i:i+batch_size]
        if not batch:  # Skip empty final batches
            continue
            
        # Create or extend FAISS index
        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch, embedding=embeddings)
        else:
            batch_store = FAISS.from_texts(batch, embedding=embeddings)
            vectorstore.merge_from(batch_store)

    return vectorstore

def get_text_chunks(text):
    """Improved chunking with content validation"""
    if not text.strip():
        return []
        
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return [chunk for chunk in text_splitter.split_text(text) if chunk.strip()]

# Update the document processing section
if st.button("Process"):
    with st.spinner("Processing documents..."):
        # Reset states
        st.session_state.conversation = None
        st.session_state.chat_history = None
        
        # Get PDF text with validation
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("No text content found in PDFs. Scanned documents?")
            
            
        # Create and validate text chunks
        text_chunks = get_text_chunks(raw_text)
        if not text_chunks:
            st.error("Failed to create valid text chunks from document content")
            
            
        # Create vector store with batch processing
        try:
            vectorstore = get_vectorstore(text_chunks)
        except ValueError as e:
            st.error(str(e))
            
            
        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.success("Documents processed successfully!")

def get_conversation_chain(vectorstore):
    # Initialize LLM with proper parameters
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3
    )
    
    # Define custom prompt template
    qa_prompt = PromptTemplate(
        template="""Use the following context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Context: {context}
        Question: {question}
        Helpful Answer:""",
        input_variables=["context", "question"]
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True
    )

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process documents first using the sidebar!")
        
    
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(
        page_title='Chat with multiple PDFs',
        page_icon=':books:',
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    
    # Chat interface
    user_question = st.chat_input(
        "Ask a question about your documents",
        disabled=not st.session_state.conversation
    )
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Document processing sidebar
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            with st.spinner("Analyzing documents..."):
                # Reset states
                st.session_state.conversation = None
                st.session_state.chat_history = None
                
                # Process documents
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text extracted from PDFs")
                    return
                
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                st.success("Documents processed successfully!")
                st.balloons()

if __name__ == '__main__':
    main()
