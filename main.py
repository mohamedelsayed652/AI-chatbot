from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("docs/resume.pdf")
docs = loader.load()

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# Create LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask question
while True:
    query = input("\nAsk something about your resume: ")
    if query.lower() == "exit":
        break

    answer = qa.run(query)
    print("\nAnswer:", answer)

