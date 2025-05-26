import streamlit as st
from dotenv import load_dotenv
import os
import nest_asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
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
    if not text.strip():
        return []
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return [chunk for chunk in text_splitter.split_text(text) if chunk.strip()]

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No valid text chunks provided")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    batch_size = 100
    vectorstore = None

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        if not batch:
            continue
            
        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch, embedding=embeddings)
        else:
            batch_store = FAISS.from_texts(batch, embedding=embeddings)
            vectorstore.merge_from(batch_store)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    qa_prompt = PromptTemplate(
        template="""Use this context: {context}
        Question: {question}
        If unsure, say you don't know. Answer:""",
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
    if not st.session_state.conversation:
        st.warning("Process documents first!")
        return
    
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    nest_asyncio.apply()  # Critical async fix
    st.set_page_config(
        page_title='PDF Chat',
        page_icon=':books:',
        layout="wide"
    )
    
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :books:")
    
    user_question = st.chat_input("Ask about documents")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    st.session_state.conversation = None
                    st.session_state.chat_history = None
                    
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text found")
                        return
                    
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("No valid chunks")
                        return
                    
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Ready for questions!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
