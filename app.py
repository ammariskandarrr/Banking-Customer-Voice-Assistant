import os
import time
import base64
from pathlib import Path
from io import BytesIO
import gradio as gr
from groq import Groq
from openai import OpenAI

# Initialize API clients
groq_client = Groq()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def stream_gpt_response(transcribed_text, history):
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Act as a friendly Bank customer service, answer briefly customer question."},
                {"role": "user", "content": transcribed_text}
            ],
            stream=True
        )

        gpt_response = ""
        history.append([transcribed_text, ""])

        for chunk in completion:
            if chunk.choices[0].delta.content:
                gpt_response += chunk.choices[0].delta.content
                history[-1][1] = gpt_response
                yield history, None

        return history, gpt_response

    except Exception as e:
        history.append([transcribed_text, f"⚠️ GPT-4 error: {str(e)}"])
        return history, None

def process_audio(audio_file, history=[]):
    try:
        # Stream transcription
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                language="en"
            )
        transcribed_text = transcription.text
        history.append([f"🎙️: {os.path.basename(audio_file)}", transcribed_text])
        yield history, None

        # Stream GPT response
        for history, _ in stream_gpt_response(transcribed_text, history):
            yield history, None

        # Generate TTS
        response = groq_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            response_format="wav",
            input=history[-1][1],
        )

        # Create audio buffer
        audio_buffer = BytesIO()
        for chunk in response.iter_bytes():
            audio_buffer.write(chunk)

        # Convert to base64 for HTML audio
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('ascii')
        audio_html = f"""
        <audio controls autoplay style="width: 100%">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """

        yield history, audio_html

    except Exception as e:
        history.append([None, f"⚠️ Error: {str(e)}"])
        yield history, None

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Bank Voice Assistant (Streaming)")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎤 Speak Now",
            show_download_button=False
        )

    chatbot = gr.Chatbot(
        label="💬 Conversation History",
        bubble_full_width=False,
        height=400
    )

    html_output = gr.HTML(
        label="🗣️ AI Voice Response"
    )

    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, html_output],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(share=True)
