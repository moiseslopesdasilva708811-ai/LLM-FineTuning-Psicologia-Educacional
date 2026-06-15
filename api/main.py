from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import gc
import re
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import PeftModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Registro de modelos
MODEL_REGISTRY = {
    # modelos base
    "base_1": ["google/flan-t5-small", None, 1],
    "base_2": ["sshleifer/bart-tiny-random", None, 1],
    "base_3": ["Qwen/Qwen2-0.5B", None, 0],
    "base_4": ["EleutherAI/gpt-neo-125M", None, 0]
    
    # modelos fine-tunados
    "modelo_1": ["google/flan-t5-small", "./train/lora_seq2seq_model_1", 1],
    "modelo_2": ["sshleifer/bart-tiny-random", "./train/lora_tiny_bart_final", 1],
    "modelo_3": ["Qwen/Qwen2-0.5B", "./train/modelo_final_lora", 0],
    "modelo_4": ["EleutherAI/gpt-neo-125M", "./train/modelo_final", 0]
}

loaded_models = {}

class ChatRequest(BaseModel):
    query: str
    modelo_id: str

# função para carregar todos os modelos 
def get_model(modelo_id: str):
    if modelo_id not in loaded_models:
        loaded_models.clear()
        gc.collect()
        torch.cuda.empty_cache()
        
        base_id, adapter_path, m_type = MODEL_REGISTRY[modelo_id]
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        
        if m_type == 1:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_id, low_cpu_mem_usage=True)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_id, low_cpu_mem_usage=True)
            
        # AQUI ESTÁ A CORREÇÃO:
        if adapter_path:
            model = PeftModel.from_pretrained(base_model, adapter_path).to("cpu")
        else:
            model = base_model.to("cpu") # Carrega apenas o base puro
            
        loaded_models[modelo_id] = (tokenizer, model, m_type)
    return loaded_models[modelo_id]

def clean_ai_response(text: str) -> str:
    # 1. Remove citações e ruídos numéricos
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'\d+/\d+', '', text)
    
    # 2. Dicionário de Correção Acadêmica
    correcoes = {
        "Vegitocsy": "Vygotsky", "Vegotsky": "Vygotsky", 
        "Vigotsky": "Vygotsky", "Piaget": "Piaget"
    }
    for erro, acerto in correcoes.items():
        text = text.replace(erro, acerto)
        
    # 3. Limpeza de frases órfãs ou alucinadas no início
    if "The answer is:" in text:
        text = text.split("The answer is:")[-1]
        
    return " ".join(text.split()).strip()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        tokenizer, model, m_type = get_model(request.modelo_id)
        
        full_prompt = f"System: Educational Psychology Expert. Answer accurately: {request.query}\nAnswer:"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,
            do_sample=False, 
            temperature=0.01,
            repetition_penalty=3.0,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
            
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrai apenas a resposta após o "Answer:"
        final_raw = raw_output.split("Answer:")[-1] if "Answer:" in raw_output else raw_output
        
        # Passa pelo filtro de limpeza
        clean_text = clean_ai_response(final_raw)
        
        return {"resposta": clean_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
