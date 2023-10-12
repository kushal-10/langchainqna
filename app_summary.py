# Application file for Gradio App for smaller model
#Less than a minute for an answer


import gradio as gr
import time
from lc.chain import DefineChain

title = """<h1 align="center">Chat</h1>"""
description = """<br><br><h3 align="center">This is a literature chat model, which can currently answer questions to New Data provided.</h3>"""

def user(user_message, history):
    return "", history + [[user_message, None]]

def respond(message, chat_history):
    question = str(message)
    t5chain = DefineChain("new")
    # output = t5chain.llm_chain(query=question, k=3)
    output = t5chain.get_summary(question)
    answer = output[5:]
    print(output)
    bot_message = answer
    chat_history.append((message, bot_message))
    time.sleep(2)
    return " ", chat_history

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")) as chat:
    gr.HTML(title)
    chatbot = gr.Chatbot().style(height=750)
    msg = gr.Textbox(label="Send a message", placeholder="Send a message",
                             show_label=False).style(container=False)

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    gr.Examples([
        ["What are the main topics of these papers?"]
    ], inputs=msg, label= "Click on any example to copy in the chatbox"
    )

    gr.HTML(description)


chat.queue()
chat.launch()