import os 
os.environ["PYDANTIC_V2_ARBITRARY_TYPES_ALLOWED"] = "1"
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.messages import AIMessage,HumanMessage,SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings , GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
import PyPDF2 as pdf
import numpy
import tempfile
import time
import warnings
warnings.filterwarnings("ignore")



st.markdown("""
<style>
/* App background and font */
.stApp {
    background: linear-gradient(135deg, #0a0a23 0%, #1f1f3d 100%);
    color: #e0e0ff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
    padding: 1rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151540, #0a0a23);
    color: #d0d0ff;
    border-right: 2px solid #4b4b85;
    font-weight: 600;
}

/* Headings */
h1, h2, h3, h4 {
    color: #a8a8ff;
    font-weight: 700;
    text-shadow: 0 0 6px #7272ff;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #7f5fff, #a14fff);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 0.7rem 1.5rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(143, 75, 255, 0.6);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #a14fff, #7f5fff);
    box-shadow: 0 8px 20px rgba(143, 75, 255, 0.9);
    transform: scale(1.1);
}

/* Input fields */
input, textarea {
    background-color: #222244 !important;
    color: #ccc !important;
    border: 2px solid #4b4b85 !important;
    border-radius: 12px;
    padding: 12px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}
input:focus, textarea:focus {
    border-color: #7f5fff !important;
    outline: none !important;
    box-shadow: 0 0 10px #a14fff;
}

/* File uploader styling */
div[data-testid="stFileUploader"] > label {
    display: block;
    padding: 10px 15px;
    border: 2px dashed #7f5fff;
    border-radius: 15px;
    color: #a8a8ff;
    cursor: pointer;
    font-weight: 600;
    text-align: center;
    transition: background-color 0.3s ease, border-color 0.3s ease;
    user-select: none;
}
div[data-testid="stFileUploader"]:hover > label {
    background-color: rgba(127, 95, 255, 0.15);
    border-color: #a14fff;
    box-shadow: 0 0 12px #a14fff;
}

/* Chat container */
.chat-container {
    max-width: 800px;
    margin: 2rem auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 1rem;
}

/* User message bubble aligned right */
.user-message {
    align-self: flex-end;
    background: linear-gradient(135deg, #7f5fff, #a14fff);
    color: white;
    padding: 14px 22px;
    border-radius: 20px 20px 0 20px;
    max-width: 75%;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 6px 12px rgba(161, 79, 255, 0.6);
    word-wrap: break-word;
    text-align: right;
}

/* AI message bubble aligned left */
.ai-message {
    align-self: flex-start;
    background: #2f2f57;
    color: #dcdcff;
    padding: 14px 22px;
    border-radius: 20px 20px 20px 0;
    max-width: 75%;
    font-size: 1.1rem;
    font-weight: 500;
    box-shadow: 0 6px 12px rgba(47, 47, 87, 0.6);
    word-wrap: break-word;
    text-align: left;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 12px;
}
::-webkit-scrollbar-track {
    background: #1a1a40;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb { 
    background: #7f5fff;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #a14fff;
}
</style>
""", unsafe_allow_html=True)






# st.title("Chatbot Summaries with :blue[RAG] Magic :sparkles:")
# st.title("📚 :blue[RAG Chatbot Summaries]  ✨")

# st.title("RAG-based Chatbot Summary Engine :blue[🚀]")

st.title("✨ Summarize Smarter with :blue[RAG Technology]")






class Config:
    arbitrary_types_allowed = True


# api_key1=os.getenv("google_api")


# api_key = st.sidebar.text_input("Enter your API Key:", type="password")
api_key1=st.sidebar.text_input("Enter the Groq API key",type="password")


doc=st.sidebar.file_uploader(" Summarization Sectore", type=["pdf"],key=1)
doc1=st.sidebar.file_uploader("Chat your own Data", type=["pdf"],key=2)

if api_key1:
     LLM=ChatGroq(
          api_key=api_key1,
          model="llama3-8b-8192"
     )

 
else:
    pass

def get_text(file):
    if file is not None:
        reader = pdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
     


prompt = f"""
Summarize the following text in two versions:
1. A concise summary in approximately 100 words.
2. A detailed summary in approximately 1000 words.

Text:
{{text}}
"""

if doc is not None:
    # Process the file
    try: 
        text = get_text(doc)
        text1=text[:5000]
        promt=prompt.format(text=text1) 
        with st.spinner("Processing..."):
            response= LLM.invoke(promt)
            result=response.content
            st.markdown(f"<div class='chat-message ai-message'>🤖  AI: {result}</div>", unsafe_allow_html=True)
        st.success("Processing complete!")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        text=""
else:
    pass
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a conversational AI that uses retrieved context and prior user interactions to provide consistent and helpful answers.")
    ]
 
store_data="datastore"



if doc1:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc1.read())
            temp_file_path = temp_file.name

            loader = PyMuPDFLoader(temp_file_path)  
            data = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=5040, chunk_overlap=1000)
            split_docs = splitter.split_documents(data)
            st.write(f' No fo chunk {len(split_docs)}')
            # embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key1,model="models/embedding-001")
            with st.spinner("Embedding........💽"):
                embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
                vector_store.save_local(store_data)
            with st.spinner("Chatting selection is opening........."):
                 pass

else:
        
        st.stop()

if os.path.exists(store_data):
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.load_local(store_data, embeddings=embeddings,allow_dangerous_deserialization=True)
        if doc:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(doc.getvalue())
                temp_file_path = temp_file.name
                loader = PyMuPDFLoader(temp_file_path)
                data = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                new_chunks = splitter.split_documents(data)
                embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vector_store = FAISS.load_local(store_data, embeddings=embeddings,allow_dangerous_deserialization=True)
                batch_size = 10
                for i in range(0, len(new_chunks), batch_size):
                    chunk_batch = new_chunks[i:i+batch_size]
                    vector_store.add_documents(chunk_batch)
                    time.sleep(20)
                # vector_store.add_documents(documents=new_chunks)
                # vector_store = FAISS.add_documents(documents=new_chunks,self) # ✅ Add new chunks to the existing store
                vector_store.save_local(store_data)     # ✅ Save updated stor

    

with st.status("retriever data...", expanded=True):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

 


prompt=ChatPromptTemplate.from_template("""



Use only the information from the context to answer the user's question. If the information is missing, say "I don't know."
                          
context:{context}
                          
question:{question}


""")

def format(doc):
    return " ".join(docs.page_content for docs in doc)




chain=(
        {"context":retriever | format , "question":RunnablePassthrough()}
        |prompt
        |LLM
        |StrOutputParser()
    )
st.header("Chatting Selection is Open let start the Coversation")
question = st.chat_input("Ask your question:")

# st.markdown("""
#     <style>
#         .chat-container {
#             display: flex;
#             flex-direction: column;
#         }
#         .chat-message {
#             max-width: 70%;
#             padding: 5px 10px;
#             border-radius: 12px;
#             margin: 8px;
#         }
#         .user-message {
#             align-self: flex-end;
#             background: linear-gradient(135deg, #E3FDF5, #FFE6FA);
#             color: #333;
#             border: 1px solid #A6DCEF;  
#             text-align: right;
#         }
#         .ai-message {
#             align-self: flex-start;
#             background: linear-gradient(135deg, #FFFAFA, #F5DEB3);
#             color: #000;
#             border: 1px solid #FFC26F; 
#         }
#         div.stButton > button:first-child {
#         background-color: #ff4d4d ; 
#         color: white;
#         }
#     </style>
# """, unsafe_allow_html=True)


if "chat_history" not  in st.session_state:

        st.session_state.chat_history = [
        SystemMessage(content="""
You are a helpful, knowledgeable AI assistant designed to answer user questions using both your internal knowledge and external documents retrieved via a Retrieval-Augmented Generation (RAG) system.

Your responsibilities include:
- Providing accurate, grounded answers based on the retrieved context.
- Citing relevant sources when applicable.
- Admitting when the answer is not known or outside the retrieved context.
- Maintaining a professional, concise, and user-friendly tone.

Guidelines:
- Use only the retrieved documents to answer questions when context is provided.
- If no context is provided or the context is insufficient, fall back on your general knowledge but clearly indicate this.
- Do not fabricate citations or facts.
- If a document contains conflicting information, indicate uncertainty and prefer the more reliable or recent source.
- Do not answer questions unrelated to the domain unless specifically instructed.

Assume the user may upload documents such as PDFs or ask questions about specific content. Always aim to be accurate, transparent, and grounded in the data.

""")
    ]

#process the input 
if question:
    # try:
        human_message = HumanMessage(content=question)
        st.session_state.chat_history.append(human_message)
        with st.spinner("🧠 Generating........"):
            result = chain.invoke(question)


        if result:
            with st.spinner("🧠 Generating........"):
                ai_message = AIMessage(content=result)
                st.session_state.chat_history.append(ai_message)

        # Display all chat messages
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"<div class='chat-message user-message'>👤 You: {message.content}</div>", unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.markdown(f"<div class='chat-message ai-message'>🤖  AI: {message.content}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Limit message history
        st.session_state.chat_history = st.session_state.chat_history[-100:]

    # except Exception as e:
    #     st.error(f"Error: {str(e)}")
    #     st.markdown("<div class='chat-message ai-message'> Oops! Something went wrong. Try again 💬</div>", unsafe_allow_html=True)

# Clear button
if st.button("Clear Chat",type="primary"):
    st.session_state.chat_history = [
        SystemMessage(content="""
You are a helpful, knowledgeable AI assistant designed to answer user questions using both your internal knowledge and external documents retrieved via a Retrieval-Augmented Generation (RAG) system.

Your responsibilities include:
- Providing accurate, grounded answers based on the retrieved context.
- Citing relevant sources when applicable.
- Admitting when the answer is not known or outside the retrieved context.
- Maintaining a professional, concise, and user-friendly tone.

Guidelines:
- Use only the retrieved documents to answer questions when context is provided.
- If no context is provided or the context is insufficient, fall back on your general knowledge but clearly indicate this.
- Do not fabricate citations or facts.
- If a document contains conflicting information, indicate uncertainty and prefer the more reliable or recent source.
- Do not answer questions unrelated to the domain unless specifically instructed.

Assume the user may upload documents such as PDFs or ask questions about specific content. Always aim to be accurate, transparent, and grounded in the data.

""")
    ]
    for key in list(st.session_state.keys()):
        if key != "chat_history":  # keep only chat_history
            del st.session_state[key]

    st.rerun()
    st.success("Chat history cleared!")
