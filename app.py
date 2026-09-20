print("starting server")
from flask import Flask, request, send_file
from groq import Groq
import edge_tts
import asyncio
import os
import tempfile
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/process', methods=['POST'])
def process_audio():
    import wave
    audio_data = request.data
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    
    with wave.open(temp_input.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        
        if len(audio_data) < 1000:
           return "too short", 204
        
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_data = audio_array.tobytes()
        wf.writeframes(audio_data)
        print(f"audio bytes: {len(audio_data)}, peak: {np.max(np.abs(audio_array))}")
    
    with open(temp_input.name, "rb") as f:
       transcription = groq.audio.transcriptions.create(
           file=(temp_input.name, f.read()),
           model="whisper-large-v3"
       )
    text = transcription.text
    print(f"Heard: {text}")
    if not text.strip():
        return "nothing heard", 204
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a witty, playful and occasionally sarcastic robot desk assistant. Tease the user slightly when appropriate but be helpful. never be genuinely hostile, insulting or unnecessarily aggressive. Keep responses natural and conversational. If the user is frustrated, be helpful rather than mocking them. Never use emojis"},
            {"role": "user", "content": text}
        ]
    )
    reply = response.choices[0].message.content
    print(f"Reply: {reply}")
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')

    async def generate_speech():
        communicate = edge_tts.Communicate(reply, "en-US-JennyNeural")
        await communicate.save(temp_mp3.name)
        
    asyncio.run(generate_speech())

    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_mp3.name,
        '-ar', '24000', '-ac', '1', '-f', 'wav',
        temp_output.name
    ])
            
    return send_file(temp_output.name, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
