import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_ID = "sleepy-panda21/Llama_fine_tuned"  

try:
    client = InferenceClient(model=BASE_MODEL, token=HF_TOKEN)
except Exception:
    client = None

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", 
    "self harm", "hurt myself", "cut myself", "overdose", 
    "better off dead", "end it all"
]

CRISIS_MESSAGE = """**Important Notice:** It sounds like you are going through a very difficult time. Please know that this is an AI, not a clinical professional, and it cannot provide the medical help you might need right now. 

**If you are in immediate danger or experiencing a crisis, please reach out for help immediately:**
- **Emergency Services:** Call 911 (US) or your local emergency number.
- **National Suicide Prevention Lifeline (US):** Call or text 988 (Available 24/7)
- **Crisis Text Line (US/Canada):** Text HOME to 741741

You are not alone, and there is support available. Please connect with a human professional."""

def predict(message, history):
    if not client:
        yield "Configuration Error: Hugging Face Inference client failed to initialize."
        return

    # Check for crisis keywords
    lower_message = message.lower()
    if any(keyword in lower_message for keyword in CRISIS_KEYWORDS):
        yield CRISIS_MESSAGE
        return

    messages = [
        {
            "role": "system",
            "content": "You are an AI counseling assistant. Your role is to provide a warm, empathetic, and non-judgmental space."
        }
    ]
    
    for turn in history:
        content = turn.get("content", "")
        if isinstance(content, str):
            messages.append({"role": turn.get("role", "user"), "content": content})
        elif isinstance(content, (list, tuple)):
            text_part = "".join([b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"])
            messages.append({"role": turn.get("role", "user"), "content": text_part})
            
    messages.append({"role": "user", "content": message})

    response_text = ""
    try:
        
        response_stream = client.chat_completion(
            messages=messages,
            max_tokens=250,
            temperature=0.7,
            stream=True,
            extra_body={"adapter_id": ADAPTER_ID}
        )
        
        for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                token = chunk.choices[0].delta.content
                if token:
                    response_text += token
                    yield response_text 
                
    except Exception as e:
        yield f"Connection Notice: {str(e)}\n\n(Note: The serverless endpoint may be loading your adapter matrices into memory.)"

counselor_theme = gr.themes.Soft(
    primary_hue="emerald",  
    secondary_hue="stone"
).set(
    body_background_fill="*secondary_50", 
    block_background_fill="white",         
    block_border_width="1px"
)

app_title = """
<div style="text-align: center; max-width: 650px; margin: 0 auto; padding-top: 10px;">
  <span style='font-size: 2.5rem;'>🌱</span>
  <h1 style="font-weight: 900; margin-bottom: 0.5rem; margin-top: 0.5rem; color: #1e293b;">
    Supportive AI Counselor
  </h1>
  <p style="font-size: 1.1rem; color: #475569; margin-bottom: 0.5rem;">
    A private, supportive, and non-judgmental space to talk about overwhelm or stress.
  </p>
  <p style="font-size: 0.85rem; color: #ef4444; background-color: #fee2e2; padding: 8px; border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid #f87171;">
    <strong>Disclaimer:</strong> This is an AI assistant, not a replacement for a clinical psychiatrist or medical professional. If you are in a crisis, please contact emergency services.
  </p>
</div>
"""

with gr.Blocks() as demo:
    gr.HTML(app_title)
    
    gr.ChatInterface(
        fn=predict,
        textbox=gr.Textbox(
            placeholder="Share what's on your mind today...", 
            container=False, 
            scale=7
        ),
        examples=[ 
            "I've been feeling extremely stressed about classes lately.",
            "It feels like I'm overwhelmed and might fail my exams.",
            "I need to talk to someone about feeling burnt out."
        ]
    )

if __name__ == "__main__":
    demo.launch(theme=counselor_theme)