import os
import torch
import gc
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
from torch.cuda.amp import autocast

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_model():
    model_name = "OrionStarAI/Orion-14B-Chat"
    model_dir = f"./models/orion"
    try:
        if not os.path.exists(model_dir) or not all(os.path.exists(os.path.join(model_dir, file)) for file in ['config.json']):
            logging.info(f"Downloading {model_name} model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                         torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True,
                                                         device_map="auto")
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
        else:
            logging.info(f"Loading existing {model_name} model...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True,
                                                         torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True,
                                                         device_map="auto")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

gc.collect()
torch.cuda.empty_cache()

try:
    model, tokenizer = load_model()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

def generate_response(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logging.info(f"Input tokenized. Shape: {inputs.input_ids.shape}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            with autocast():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=1000,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        logging.info(f"Generation complete. Output shape: {outputs.shape}")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Response decoded. Length: {len(response)}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return response
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        raise

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        logging.info(f"Received prompt. Length: {len(prompt)}")
        response = generate_response(prompt)
        logging.info("Response generated successfully")
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
