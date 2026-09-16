from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

# Load the LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.7)

# Create a PromptTemplate
prompt = PromptTemplate(
    input_variables = ["topic"],
    template = "Suggest a catchy blog title about {topic}."
)

# Create a LLM Chain
chain = prompt | llm

# Run the chain with a specific topic
topic = input('Enter a topic: ')
output = chain.invoke(topic)

print("Generated blog title: ",output)

