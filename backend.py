from pathlib import Path
from flask import Flask, send_from_directory, abort, jsonify

ROOT = Path(__file__).resolve().parent
WEBSITE_DIR = ROOT / "website"
SCALE_DIR = ROOT / "scale"
LOGS_DIR = ROOT / "evaluation" / "logs"

app = Flask(__name__, static_folder=str(WEBSITE_DIR), static_url_path="")

@app.route("/")
def index():
    return app.send_static_file('index.html')

@app.route('/api/scale/<path:filename>')
def scale_file(filename):
    file_path = SCALE_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(SCALE_DIR), filename)
    abort(404)

@app.route('/api/logs/<path:filename>')
def logs_file(filename):
    file_path = LOGS_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(LOGS_DIR), filename)
    abort(404)

@app.route('/api/status')
def status():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)
