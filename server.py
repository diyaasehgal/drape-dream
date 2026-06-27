from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import base64
import io
import os

app = Flask(__name__, static_folder='public')
CORS(app)

ROOM_PATH = os.path.join(os.path.dirname(__file__), 'room_template.jpg')

# Exact curtain polygon coords in 900x1350 room photo
LEFT_POLY  = [(0,0),(255,0),(248,337),(232,675),(214,1012),(196,1350),(0,1350)]
RIGHT_POLY = [(644,0),(900,0),(900,1350),(704,1350),(686,1012),(668,675),(652,337)]

GRADES = {
    'modern':  dict(bright=1.00, sat=1.00, contrast=1.00, warmth=0),
    'classic': dict(bright=0.88, sat=0.75, contrast=1.08, warmth=18),
    'bedroom': dict(bright=1.05, sat=0.85, contrast=0.92, warmth=8),
    'dining':  dict(bright=0.92, sat=1.10, contrast=1.05, warmth=-5),
    'minimal': dict(bright=1.12, sat=0.45, contrast=0.88, warmth=4),
    'boho':    dict(bright=0.95, sat=1.20, contrast=1.02, warmth=24),
}

def get_luminance(arr):
    r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    lum = (0.299*r + 0.587*g + 0.114*b) / 255.0
    mn, mx = lum.min(), lum.max()
    return (lum - mn) / (mx - mn + 1e-6)

def make_poly_mask(H, W, poly):
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(3))
    return np.array(mask).astype(float) / 255.0

def tile_fabric(fab_img, panel_w, W, H):
    fw, fh = fab_img.size
    scale = (panel_w / fw) * 1.15
    tw = max(int(fw * scale), 60)
    th = max(int(fh * scale), 60)
    fab_s = fab_img.resize((tw, th), Image.LANCZOS)
    tiled = Image.new('RGB', (W, H))
    for y in range(0, H, th):
        for x in range(0, W, tw):
            tiled.paste(fab_s, (x, y))
    return np.array(tiled).astype(float)

def apply_grade(arr, style):
    g = GRADES.get(style, GRADES['modern'])
    a = arr.astype(float)
    r, gv, b = a[:,:,0], a[:,:,1], a[:,:,2]
    r  = r  * g['bright'] + g['warmth'] * 0.6
    gv = gv * g['bright'] + g['warmth'] * 0.2
    b  = b  * g['bright'] - g['warmth'] * 0.4
    avg = (r + gv + b) / 3.0
    r  = avg + (r  - avg) * g['sat']
    gv = avg + (gv - avg) * g['sat']
    b  = avg + (b  - avg) * g['sat']
    r  = ((r /255 - 0.5) * g['contrast'] + 0.5) * 255
    gv = ((gv/255 - 0.5) * g['contrast'] + 0.5) * 255
    b  = ((b /255 - 0.5) * g['contrast'] + 0.5) * 255
    return np.clip(np.stack([r, gv, b], axis=2), 0, 255)

def decode_fabric(b64str):
    return Image.open(io.BytesIO(base64.b64decode(b64str))).convert('RGB')

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('public', path)

@app.route('/visualize', methods=['POST', 'OPTIONS'])
def visualize():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    fabric_main_b64  = data.get('fabric_main', '')
    fabric_sheer_b64 = data.get('fabric_sheer', '')
    style = data.get('style', 'modern')
    mode  = data.get('mode', 'single')

    if not fabric_main_b64:
        return jsonify({'error': 'fabric_main is required'}), 400

    # Load room
    room = Image.open(ROOM_PATH).convert('RGB')
    W, H = room.size  # 900 x 1350
    room_arr = np.array(room).astype(float)

    # Get luminance for fold shading
    lum = get_luminance(room_arr)

    # Apply color grade
    result = apply_grade(room_arr, style)

    # Decode fabrics
    main_fab = decode_fabric(fabric_main_b64)

    # Combo: sheer layer across window
    if mode == 'combo' and fabric_sheer_b64:
        sheer_fab = decode_fabric(fabric_sheer_b64)
        win_x = LEFT_POLY[1][0]   # 255
        win_w = RIGHT_POLY[0][0] - win_x  # 644-255 = 389
        sheer_tiled = tile_fabric(sheer_fab, win_w, W, H)
        lm = (lum * 0.84 + 0.28)
        sheer_shaded = np.clip(sheer_tiled * lm[:,:,np.newaxis], 0, 255)
        sm = np.zeros((H, W), float)
        sm[:, win_x:RIGHT_POLY[0][0]] = 0.38
        sm3 = sm[:,:,np.newaxis]
        result = result * (1 - sm3) + sheer_shaded * sm3

    # Left curtain panel
    left_w = LEFT_POLY[1][0]
    left_tiled = tile_fabric(main_fab, left_w, W, H)
    lm = lum * 0.84 + 0.28
    left_shaded = np.clip(left_tiled * lm[:,:,np.newaxis], 0, 255)
    left_mask = make_poly_mask(H, W, LEFT_POLY)[:,:,np.newaxis]
    result = result * (1 - left_mask) + left_shaded * left_mask

    # Right curtain panel
    right_w = W - RIGHT_POLY[0][0]
    right_tiled = tile_fabric(main_fab, right_w, W, H)
    right_shaded = np.clip(right_tiled * lm[:,:,np.newaxis], 0, 255)
    right_mask = make_poly_mask(H, W, RIGHT_POLY)[:,:,np.newaxis]
    result = result * (1 - right_mask) + right_shaded * right_mask

    # Final image
    out_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out_img.save(buf, format='JPEG', quality=92)
    result_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({'image': result_b64})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
