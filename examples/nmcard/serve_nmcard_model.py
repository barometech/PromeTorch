"""
Minimal web UI for NM Card Mini trained model.
Run: python serve_nmcard_model.py
Open: http://localhost:8080
"""
import http.server, json, urllib.parse, numpy as np, os, sys

# Load model
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'nanogpt_nmcard_weights')
W = {}
for f in os.listdir(WEIGHTS_DIR):
    if f.endswith('.npy'):
        W[f[:-4]] = np.load(os.path.join(WEIGHTS_DIR, f))

with open(os.path.join(WEIGHTS_DIR, 'config.json')) as f:
    cfg = json.load(f)

chars = cfg['chars']; V = cfg['V']; D = cfg['D']; T = cfg['T']
ch2idx = {c: i for i, c in enumerate(chars)}
idx2ch = {i: c for i, c in enumerate(chars)}

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def rms_norm(x, g, eps=1e-5):
    return x / np.sqrt(np.mean(x**2, -1, keepdims=True) + eps) * g

def generate(prompt, max_len=300, temp=0.7):
    tokens = [ch2idx.get(c, 0) for c in prompt]
    for _ in range(max_len):
        inp = tokens[-T:]
        x0 = W['embed'][inp] + W['pos'][:len(inp)]
        n1 = rms_norm(x0, W['g1'])
        x1 = x0 + np.maximum(n1 @ W['W1'] + W['b1'], 0) @ W['W2'] + W['b2']
        n2 = rms_norm(x1, W['g2'])
        x2 = x1 + np.maximum(n2 @ W['W3'] + W['b3'], 0) @ W['W4'] + W['b4']
        logits = (x2 @ W['Wh'] + W['bh'])[-1]
        p = softmax(logits) ** (1 / temp)
        p /= p.sum()
        tokens.append(np.random.choice(V, p=p))
    return ''.join(idx2ch.get(t, '?') for t in tokens)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PromeTorch on NM Card Mini</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
.header { text-align: center; padding: 40px 20px 20px; }
.header h1 { font-size: 28px; color: #ff6b35; margin-bottom: 8px; }
.header .sub { font-size: 14px; color: #888; }
.header .badge { display: inline-block; background: #1a3a1a; color: #4caf50; padding: 4px 12px; border-radius: 12px; font-size: 12px; margin-top: 10px; }
.container { width: 100%; max-width: 700px; padding: 20px; }
.input-area { display: flex; gap: 10px; margin-bottom: 20px; }
.input-area input { flex: 1; background: #1a1a1a; border: 1px solid #333; color: #fff; padding: 12px 16px; font-size: 16px; font-family: inherit; border-radius: 8px; outline: none; }
.input-area input:focus { border-color: #ff6b35; }
.input-area button { background: #ff6b35; color: #fff; border: none; padding: 12px 24px; font-size: 16px; font-family: inherit; border-radius: 8px; cursor: pointer; white-space: nowrap; }
.input-area button:hover { background: #ff8555; }
.input-area button:disabled { background: #555; cursor: wait; }
.output { background: #111; border: 1px solid #222; border-radius: 8px; padding: 20px; min-height: 200px; white-space: pre-wrap; line-height: 1.6; font-size: 15px; }
.output .generated { color: #4fc3f7; }
.stats { display: flex; gap: 20px; margin-top: 20px; font-size: 12px; color: #666; }
.stats span { background: #111; padding: 4px 10px; border-radius: 4px; }
.examples { margin-top: 20px; }
.examples .title { font-size: 12px; color: #666; margin-bottom: 8px; }
.examples button { background: #1a1a1a; border: 1px solid #333; color: #aaa; padding: 6px 12px; margin: 2px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 13px; }
.examples button:hover { border-color: #ff6b35; color: #fff; }
.footer { text-align: center; padding: 30px; font-size: 11px; color: #444; }
</style>
</head>
<body>
<div class="header">
    <h1>PromeTorch NM Card Mini</h1>
    <div class="sub">Neural network trained on NeuroMatrix NMC4 hardware</div>
    <div class="badge">16 NMC4 cores @ 1GHz &bull; Q16.16 fixed-point &bull; 13,185 params</div>
</div>
<div class="container">
    <div class="input-area">
        <input type="text" id="prompt" placeholder="Enter prompt..." value="ROMEO:&#10;" />
        <button id="btn" onclick="gen()">Generate</button>
    </div>
    <div class="output" id="output">Type a prompt and click Generate.</div>
    <div class="examples">
        <div class="title">Try these:</div>
        <button onclick="setPrompt('ROMEO:\\n')">ROMEO:</button>
        <button onclick="setPrompt('JULIET:\\n')">JULIET:</button>
        <button onclick="setPrompt('KING:\\n')">KING:</button>
        <button onclick="setPrompt('First Citizen:\\n')">First Citizen:</button>
        <button onclick="setPrompt('To be or not to be')">To be or not...</button>
        <button onclick="setPrompt('What is ')">What is</button>
        <button onclick="setPrompt('My lord, ')">My lord,</button>
    </div>
    <div class="stats">
        <span>Loss: LOSS_VAL</span>
        <span>Training: 55s on NMC4</span>
        <span>1001 hardware matmuls</span>
        <span>Shakespeare corpus</span>
    </div>
</div>
<div class="footer">
    PromeTorch &mdash; First ML framework with native Russian hardware support<br>
    Trained on NM Card Mini (NTC Module K1879VM8YA) &bull; 16 NMC4 cores &bull; 512 GFLOPS
</div>
<script>
function setPrompt(p) {
    document.getElementById('prompt').value = p.replace(/\\n/g, '\\n');
    gen();
}
async function gen() {
    const btn = document.getElementById('btn');
    const out = document.getElementById('output');
    btn.disabled = true; btn.textContent = 'Generating...';
    out.textContent = 'Thinking...';
    const prompt = document.getElementById('prompt').value;
    try {
        const r = await fetch('/generate?prompt=' + encodeURIComponent(prompt));
        const data = await r.json();
        out.innerHTML = '<span style="color:#ff6b35">' + escHtml(prompt) + '</span><span class="generated">' + escHtml(data.generated) + '</span>';
    } catch(e) { out.textContent = 'Error: ' + e; }
    btn.disabled = false; btn.textContent = 'Generate';
}
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>'); }
</script>
</body>
</html>""".replace('LOSS_VAL', f"{cfg['loss']:.3f}")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/generate'):
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            prompt = params.get('prompt', ['ROMEO:\n'])[0]
            result = generate(prompt, max_len=300, temp=0.7)
            generated_part = result[len(prompt):]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'prompt': prompt, 'generated': generated_part}).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())

    def log_message(self, fmt, *args):
        if '/generate' in str(args):
            print(f"  Generate: {args}")

PORT = 8080
print(f"PromeTorch NM Card Mini Model Server")
print(f"Model: {cfg['params']:,} params, loss={cfg['loss']:.3f}")
print(f"Trained with {cfg['card_matmuls']} hardware matmuls on NMC4")
print(f"")
print(f"Open: http://localhost:{PORT}")
print(f"Press Ctrl+C to stop")

server = http.server.HTTPServer(('', PORT), Handler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    server.server_close()
