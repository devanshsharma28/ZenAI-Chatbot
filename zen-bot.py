#import necessary libraries
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Setup llama3.2:latest model with ollama
# This is used for generating responses in the Zen Bot application
llm = ChatOllama(model ="llama3.2:latest")

#Create a chat prompt template 
prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant. Answer the user's questions to the best of your ability."),
        ("human", "{question}"),
    ]
)

# Combine the prompt and the model into a chain
# This chain will take user input and generate a response using the LLM
chain = prompt | llm

# Example usage of the chain to generate a response
def chat_bot(question):
    # Invoke the chain with the user's question
    response= chain.invoke({"question": question})
    return response.content

# Set up the Gradio interface

with gr.Blocks() as demo:
    gr.Markdown("## Zensar ChatBot!")
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...", lines=1)
    output_box = gr.Textbox(label="Answer",interactive=False)
    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=chat_bot,  # Function to call when the button is clicked
        inputs=input_box,  # Input from the textbox     
        outputs=output_box  # Output to the textbox
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True to allow public access to the interface