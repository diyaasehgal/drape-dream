from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, io, os, requests, json
from PIL import Image

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = BASE_DIR
HF_TOKEN = "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"

GROQ_KEY = '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'

ROOM_STYLES = {
    'modern':   'modern contemporary living room, large windows, neutral beige walls, wooden floor, minimalist furniture',
    'classic':  'classic elegant living room, high ceilings, ornate moldings, warm lighting, traditional furniture',
    'bedroom':  'luxury bedroom, soft lighting, king size bed, elegant decor, large window',
    'dining':   'elegant dining room, chandelier, wooden dining table, chairs, large window',
    'minimal':  'minimalist Scandinavian room, white walls, simple furniture, natural light, clean lines',
    'boho':     'bohemian cozy living room, warm earthy tones, plants, rattan furniture, layered textiles',
}

def analyze_fabric(image_b64, mime):
    """Use Groq to analyze fabric color and pattern"""
    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'max_tokens': 200,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{image_b64}'}},
                    {'type': 'text', 'text': 'Describe this curtain fabric in detail for an AI image generator. Include: exact colors, pattern type (floral/geometric/solid/striped etc), texture (sheer/heavy/velvet/cotton/silk etc), and any distinctive features. Be specific and concise in 2-3 sentences. Start directly with the description.'}
                ]}]
            },
            timeout=15
        )
        data = resp.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "elegant curtain fabric with subtle pattern and rich texture"

def generate_room_image(fabric_desc, room_style, style_key):
    """Generate room image using Hugging Face FLUX model"""
    room_desc = ROOM_STYLES.get(style_key, ROOM_STYLES['modern'])
    
    prompt = (
        f"Interior design photograph, {room_desc}, "
        f"with beautiful floor-to-ceiling curtains made of {fabric_desc}, "
        f"curtains hanging naturally with elegant folds and drapes on both sides of the window, "
        f"professional interior photography, soft natural lighting, 4K quality, "
        f"photorealistic, high detail, beautiful composition"
    )
    
    negative = (
        "cartoon, illustration, drawing, anime, ugly, blurry, low quality, "
        "distorted, watermark, text, signature, bad anatomy, wrong colors"
    )

    # Try FLUX schnell first (fastest free model)
    models = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]
    
    for model in models:
        try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            payload = {"inputs": prompt}
            if "stable-diffusion" in model:
                payload["parameters"] = {
                    "negative_prompt": negative,
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "width": 768,
                    "height": 768,
                }
            
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('image'):
                return base64.b64encode(resp.content).decode(), None
            elif resp.status_code == 503:
                # Model loading, try next
                continue
            else:
                error_text = resp.text[:200]
                print(f"Model {model} failed: {resp.status_code} - {error_text}")
                continue
                
        except requests.Timeout:
            continue
        except Exception as e:
            print(f"Model {model} error: {e}")
            continue
    
    return None, "All models failed or are loading. Please try again in 30 seconds."

@app.route('/')
def index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(PUBLIC_DIR, filename)

@app.route('/visualize', methods=['POST', 'OPTIONS'])
def visualize():
    if request.method == 'OPTIONS':
        return '', 200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST'
        }
    
    try:
        data = request.get_json()
        fabric_b64 = data.get('fabric_main', '')
        fabric_mime = data.get('fabric_mime', 'image/jpeg')
        style = data.get('style', 'modern')
        mode = data.get('mode', 'single')
        
        if not fabric_b64:
            return jsonify({'error': 'No fabric image provided'}), 400
        
        # Step 1: Analyze fabric
        print(f"Analyzing fabric for style: {style}")
        fabric_desc = analyze_fabric(fabric_b64, fabric_mime)
        print(f"Fabric desc: {fabric_desc}")
        
        # Step 2: If combo mode, analyze sheer too
        if mode == 'combo':
            sheer_b64 = data.get('fabric_sheer', '')
            sheer_mime = data.get('sheer_mime', 'image/jpeg')
            if sheer_b64:
                sheer_desc = analyze_fabric(sheer_b64, sheer_mime)
                fabric_desc = f"layered curtains with sheer {sheer_desc} behind and {fabric_desc} as main panels"
        
        # Step 3: Generate room image
        print("Generating room image...")
        img_b64, error = generate_room_image(fabric_desc, ROOM_STYLES.get(style), style)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'image': img_b64,
            'fabric_description': fabric_desc
        }), 200, {'Access-Control-Allow-Origin': '*'}
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500, {'Access-Control-Allow-Origin': '*'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
