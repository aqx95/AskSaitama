import gradio as gr
from main import ask

custom_path = '/tmp/test/'

def qna_interface(file, query):
    # Generate the answer using your question answering module
    ans = ask(file.name, query)
    return ans


demo = gr.Interface(
    fn=qna_interface,
    inputs=[
        gr.File(label="Upload a document"), 
        gr.Textbox(label='Enter your question')
        ],
    outputs="text",
    capture_session=custom_path)

demo.launch()
