import gradio as gr

def load():
    return [
        {"role": "user", "content": "Can you show me some media?"},
        {"role": "assistant", "content": "Here's an audio clip:"},
        {"role": "assistant", "content": gr.Audio("https://github.com/gradio-app/gradio/raw/main/gradio/media_assets/audio/audio_sample.wav")},
        {"role": "assistant", "content": "And here's a video:"},
        {"role": "assistant", "content": gr.Video("https://github.com/gradio-app/gradio/raw/main/gradio/media_assets/videos/world.mp4")}
    ]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    button = gr.Button("Load audio and video")
    button.click(load, None, chatbot)

demo.launch()