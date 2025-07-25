from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr
import uuid

llm = ChatOllama(model ="llama3.2:latest")
 
prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are the friendly and helpful Zen Bot. Start with a greeting and then answer the user's questions."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Initialize the chat history
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = RunnableWithMessageHistory(
    runnable = prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def clear_history(session_id):
    if session_id in store:
        del store[session_id]  # Remove the chat history for this session
    return "All previous chat history has been cleared.", None, session_id  # Clear output, history_state, keep session_id

#chain = prompt | llm

#def chat_bot(question):
#    response= chain.invoke({"question": question})
#    return response.content

def chat_bot(user_input, history_state, session_id=str(uuid.uuid4())):
    if user_input is None or user_input.strip() == "":
        friendly_message = (
            "Hello! 😊 Please enter a question so I can assist you. "
            "It looks like you submitted an empty message or just spaces."
        )
        return friendly_message, history_state, session_id

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ).content

    if history_state is None:
        history_state = []
    history_state.append((user_input, response))
    
    return response, history_state, session_id


with gr.Blocks() as demo:
    gr.Markdown("## Zensar ChatBot!")
    history_state = gr.State(value=None) 
    session_id = gr.State(value=str(uuid.uuid4()))
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...", lines=1)
    output_box = gr.Textbox(label="Answer",interactive=False)
#    creativity = gr.Slider(0, 2, value=0, label="Count", info="Choose between 0 and 2")
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear conversation history")


    submit_button.click(
        fn=chat_bot,  
        inputs=[input_box, history_state, session_id],       
        outputs=[output_box, history_state, session_id]
    )

    clear_button.click(
        fn=clear_history,
        inputs=session_id,
        outputs=[output_box, history_state, session_id]
    )

if __name__ == "__main__":
    demo.launch()  