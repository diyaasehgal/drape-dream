from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import base64
import io
import os

# Flask app - serve static files from current directory
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOM_PATH = os.path.join(BASE_DIR, 'room_template.jpg')
PUBLIC_DIR = BASE_DIR

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

def make_mask(H, W, poly):
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(3))
    return np.array(mask).astype(float) / 255.0

def tile_fabric(fab, panel_w, W, H):
    fw, fh = fab.size
    scale = (panel_w / fw) * 1.15
    tw = max(int(fw*scale), 60)
    th = max(int(fh*scale), 60)
    fs = fab.resize((tw, th), Image.LANCZOS)
    t = Image.new('RGB', (W, H))
    for y in range(0, H, th):
        for x in range(0, W, tw):
            t.paste(fs, (x, y))
    return np.array(t).astype(float)

def apply_grade(arr, style):
    g = GRADES.get(style, GRADES['modern'])
    r,gv,b = arr[:,:,0].copy(), arr[:,:,1].copy(), arr[:,:,2].copy()
    r  = r*g['bright']  + g['warmth']*0.6
    gv = gv*g['bright'] + g['warmth']*0.2
    b  = b*g['bright']  - g['warmth']*0.4
    avg = (r+gv+b)/3
    r  = avg+(r-avg)*g['sat']
    gv = avg+(gv-avg)*g['sat']
    b  = avg+(b-avg)*g['sat']
    r  = ((r/255-0.5)*g['contrast']+0.5)*255
    gv = ((gv/255-0.5)*g['contrast']+0.5)*255
    b  = ((b/255-0.5)*g['contrast']+0.5)*255
    return np.clip(np.stack([r,gv,b],axis=2),0,255)

@app.route('/')
def index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(PUBLIC_DIR, filename)

@app.route('/visualize', methods=['POST','OPTIONS'])
def visualize():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200

    try:
        data = request.get_json()
        main_b64  = data.get('fabric_main','')
        sheer_b64 = data.get('fabric_sheer','')
        style = data.get('style','modern')
        mode  = data.get('mode','single')

        if not main_b64:
            return jsonify({'error':'fabric_main required'}), 400

        room = Image.open(ROOM_PATH).convert('RGB')
        W, H = room.size
        room_arr = np.array(room).astype(float)
        lum = get_luminance(room_arr)
        result = apply_grade(room_arr, style)

        main_fab = Image.open(io.BytesIO(base64.b64decode(main_b64))).convert('RGB')

        if mode == 'combo' and sheer_b64:
            sheer_fab = Image.open(io.BytesIO(base64.b64decode(sheer_b64))).convert('RGB')
            wx = LEFT_POLY[1][0]
            ww = RIGHT_POLY[0][0] - wx
            st = tile_fabric(sheer_fab, ww, W, H)
            lm = (lum*0.84+0.28)[:,:,np.newaxis]
            ss = np.clip(st*lm, 0, 255)
            sm = np.zeros((H,W,1))
            sm[:, wx:RIGHT_POLY[0][0], 0] = 0.38
            result = result*(1-sm) + ss*sm

        lm = (lum*0.84+0.28)[:,:,np.newaxis]

        lt = tile_fabric(main_fab, LEFT_POLY[1][0], W, H)
        ls = np.clip(lt*lm, 0, 255)
        lmask = make_mask(H,W,LEFT_POLY)[:,:,np.newaxis]
        result = result*(1-lmask) + ls*lmask

        rt = tile_fabric(main_fab, W-RIGHT_POLY[0][0], W, H)
        rs = np.clip(rt*lm, 0, 255)
        rmask = make_mask(H,W,RIGHT_POLY)[:,:,np.newaxis]
        result = result*(1-rmask) + rs*rmask

        out = Image.fromarray(np.clip(result,0,255).astype(np.uint8))
        buf = io.BytesIO()
        out.save(buf, format='JPEG', quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = jsonify({'image': b64})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    except Exception as e:
        resp = jsonify({'error': str(e)})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
