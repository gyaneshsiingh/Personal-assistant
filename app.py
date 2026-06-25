from fastapi import status
from glob import glob
import os 
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="Personal Knowledge Assistant API")


VECTOR_STORE = None
RAG_CHAIN = None

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = "llama-3.1-8b-instant"

os.environ["GROQ_API_KEY"] = ""
def build_index(data_dir: str = "./knowledge_base"):
    global VECTOR_STORE,RAG_CHAIN
    print("1. Collecting documents")


    pdf_loader = DirectoryLoader(data_dir, glob = "**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(data_dir, glob = "**/*.txt", loader_cls=TextLoader)

    documents = pdf_loader.load() + txt_loader.load()

    if not documents:
        print("No Documents Found In Directory.")
        return

    
    print("2. Chunking Documents...")

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
    # chunk_overlap = 200,
    # length_function = len)

    chunker_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    text_splitter = SemanticChunker(chunker_embeddings,breakpoint_threshold_type='percentile')


    chunks  = text_splitter.split_documents(documents)

    print("3. Generating Embeddings & Building Vector Store...")

    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

    VECTOR_STORE = FAISS.from_documents(chunks,embeddings)

    VECTOR_STORE.save_local("faiss_index")

    print("4. Index built successfully")



def setup_rag():
    global VECTOR_STORE, RAG_CHAIN

    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    
    if os.path.exists("faiss_index"):
        VECTOR_STORE = FAISS.load_local("faiss_index",embeddings,
        allow_dangerous_deserialization=True)
    else:
        build_index()

    if not VECTOR_STORE:
        return

    
    retriever = VECTOR_STORE.as_retriever(search_kwargs = {"k": 15})

    llm = ChatGroq(model = LLM_MODEL,temperature=0)

    system_prompt = (
        "You are an intelligent knowledge assistant with access to multiple resumes and documents. "
        "Use the retrieved context below to answer questions as specifically and completely as possible. "
        "When answering, always mention which person or document the information comes from. "
        "If a question asks about a specific person, focus only on that person's information. "
        "Only say 'I don't know' if the topic is completely absent from ALL retrieved context. "
        "If partial information is available, provide it rather than refusing to answer."
        "\n\nContext:\n{context}"
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    document_prompt = PromptTemplate (
        input_variables=["page_content", "source"],
        template="Source Document: {source} \n\nContent: {page_content}"
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    RAG_CHAIN = create_retrieval_chain(retriever, qa_chain)
    
    

@app.on_event("startup")
async def startup_event():
    os.makedirs("./knowledge_base", exist_ok=True)
    setup_rag()




@app.get("/")
async def root():
    return {"message": "API is running! Go to http://127.0.0.1:8000/docs to test it."}

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model = QueryResponse)
async def ask_question(request: QueryRequest):
    if not RAG_CHAIN:
        raise HTTPException(status_code = 500, detail = "Knowledge base is empty.Please add files to ./knowledge_base and restart")

    result = RAG_CHAIN.invoke({"input": request.query})

    answer = result["answer"]

    sources = []

    for doc in result["context"]:
        source = doc.metadata.get("source", "Unknown")

        if source not in sources:
            sources.append(source)



    return {"answer": answer, "sources": sources}




