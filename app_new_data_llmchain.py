# Application file for Gradio App for smaller model
#Less than a minute for an answer

# options = "new", "last_mile", "two_sided", "transportation/transportation_n" n=1..6
dir = "transportation/transportation_6"

demo_query = "What is the motivation, the methodology and the results of this research? What are the key points of this research?"

import gradio as gr
import time
from lc.chain import DefineChain

import pandas as pd
import os

title = """<h1 align="center">Chat</h1>"""
description = """<br><br><h3 align="center">This is a literature chat model, which can currently answer questions to New Data provided.</h3>"""

def user(user_message, history):
    return "", history + [[user_message, None]]

def respond(message, chat_history):
    question = str(message)
    t5chain = DefineChain(dir)
    output = t5chain.llm_chain(query=question, k=3)
    answer = output[5:]
    splits = dir.split("/")
    pdf_dir = "data_" + splits[0] + "/data_" + splits[1]
    name = os.listdir(pdf_dir)

    store_data = {
        "query" : question,
        "paper" : name,
        "summary" : answer
    }

    data_df = pd.DataFrame(store_data)
    # os.mkdir(splits[0])
    data_df.to_csv("csvs/" + splits[1] + ".csv")
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
        ["The keyword here is #transportation#. What are the key points of this research based on this keyword? Please also describe how do the authors claim these points."],
        ["What are the key points of this research? How do the authors arrive at these points and what is the impact of these points?"],
        ["The keyword here is #transportation#. What are the key points of this research based on this keyword? Please explain these points in detail."],
        ["The keyword here is #transportation#. What are the key points of this research based on this keyword? How do the authors arrive at these points?"]

    ], inputs=msg, label= "Click on any example to copy in the chatbox"
    )

    gr.HTML(description)


chat.queue()
chat.launch()