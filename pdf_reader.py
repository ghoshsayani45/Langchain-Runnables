from langchain_community.document_loaders import TextLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print("Docs: ",docs)

# Convert the text into embeddings and store them in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Create a retriever (Fetches the relevant document)
retriever = vectorstore.as_retriever()

# Manually retrieve relevant documents
query = "Which classifier variant does Ridge Regressor have?"
retrieved_docs = retriever.invoke(query)
print("Retrievd docs: ",retrieved_docs)

# Combine retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
print("Retrievd text: ",retrieved_text)

# Initialize the LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.7)

# Manually pass the retrieved text to LLM
prompt = f"Based on the following text, answer the following question: {query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

# Print the answer
print("Answer: ",answer)

