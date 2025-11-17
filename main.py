from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from rag import retriever

# Creating the model object
model = OllamaLLM(model='llama3.2')

# Template to create a prompt
TEMPLATE = """
Context: {context}
Question: {question}
"""

# Initialising a prompt using TEMPLATE
prompt = ChatPromptTemplate.from_template(TEMPLATE)

# Prompt to Model Pipeline
chain = prompt | model

print('\nWelcome to Local LLM + RAG\n')

while True:

       
    question = input("Enter your Question (x to exit): ")

    if question.lower().strip() == 'x':
        break


    # Using the Retriever to get the context for the question
    context = retriever.invoke(question)

    # Passing values for the placeholders in the template
    result = chain.invoke({'context': context, 'question': question})
    print('------------------------------------------------------------')
    print(f'Answer: {result}')
    print('------------------------------------------------------------\n\n')



