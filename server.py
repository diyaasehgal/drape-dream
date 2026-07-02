from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, requests, base64, json

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROQ_KEY = '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'

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Only fabric analysis via Groq - image generation happens in browser"""
    if request.method == 'OPTIONS':
        return '', 200, {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type'}
    
    try:
        data = request.get_json()
        fabric_b64 = data.get('fabric_main', '')
        fabric_mime = data.get('fabric_mime', 'image/jpeg')
        style = data.get('style', 'modern')
        mode = data.get('mode', 'single')

        ROOM_STYLES = {
            'modern':  'modern contemporary living room, large windows, neutral beige walls, wooden floor',
            'classic': 'classic elegant living room, high ceilings, warm lighting, traditional furniture',
            'bedroom': 'luxury bedroom, soft lighting, king size bed, large window with natural light',
            'dining':  'elegant dining room, chandelier lighting, wooden dining table, large windows',
            'minimal': 'minimalist Scandinavian room, white walls, simple furniture, clean natural light',
            'boho':    'bohemian cozy living room, warm earthy tones, plants, rattan furniture',
        }
        room = ROOM_STYLES.get(style, ROOM_STYLES['modern'])

        # Analyze fabric with Groq
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'max_tokens': 150,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:{fabric_mime};base64,{fabric_b64}'}},
                    {'type': 'text', 'text': 'Describe this curtain fabric for an AI image generator. Include exact colors, pattern (floral/geometric/solid/striped), and texture (sheer/velvet/cotton/silk/linen). Be specific, 2 sentences max.'}
                ]}]
            },
            timeout=15
        )
        fabric_desc = resp.json()['choices'][0]['message']['content'].strip()

        # Build image generation prompt
        if mode == 'combo':
            sheer_b64 = data.get('fabric_sheer', '')
            if sheer_b64:
                sheer_resp = requests.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
                    json={
                        'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                        'max_tokens': 80,
                        'messages': [{'role': 'user', 'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{sheer_b64}'}},
                            {'type': 'text', 'text': 'Describe this sheer fabric: color and texture only. 1 sentence.'}
                        ]}]
                    },
                    timeout=15
                )
                sheer_desc = sheer_resp.json()['choices'][0]['message']['content'].strip()
                fabric_desc = f"double layer curtains: sheer {sheer_desc} behind, {fabric_desc} as main panels"

        prompt = (
            f"Professional interior design photograph of a {room}, "
            f"featuring floor-to-ceiling curtains made of {fabric_desc}, "
            f"curtains hanging with natural elegant folds on both sides of large windows, "
            f"beautiful soft lighting, photorealistic, 4K quality, high detail, "
            f"professional real estate photography style"
        )

        return jsonify({'prompt': prompt, 'fabric_description': fabric_desc}), 200, {
            'Access-Control-Allow-Origin': '*'
        }

    except Exception as e:
        return jsonify({'error': str(e)}), 500, {'Access-Control-Allow-Origin': '*'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
