from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("kreimben/CodeMind-gemma")
model = AutoModelForCausalLM.from_pretrained("kreimben/CodeMind-gemma")

# GPU 사용 가능 여부에 따라 device 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    generated_text: str

def get_completion(query: str, model, tokenizer, device='cpu', max_new_tokens=512) -> str:
    prompt_template = '<start_of_turn>user ' \
                      'Below is an instruction that describes a task. Write a response that appropriately completes the request. ' \
                      '{query}<end_of_turn>\n' \
                      '<start_of_turn>model '

    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate-text/", response_model=TextResponse)
def generate_text(request: TextRequest):
    try:
        result = get_completion(request.text, model, tokenizer, device)
        return TextResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
