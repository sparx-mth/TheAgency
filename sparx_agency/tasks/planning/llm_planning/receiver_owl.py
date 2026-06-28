#!/usr/bin/env python3
# receiver_owl.py
from flask import Flask, request, jsonify
import os, json, time
from werkzeug.utils import secure_filename
from house_config import get_config

cfg = get_config()

app = Flask(__name__)
OUT_DIR = cfg.ingest_out_dir
os.makedirs(OUT_DIR, exist_ok=True)

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        # 1) Parse multipart/form-data
        meta_str = request.form.get("meta", "")
        if not meta_str:
            return jsonify({"error": "missing form field 'meta'"}), 400
        try:
            meta = json.loads(meta_str)
        except Exception as e:
            return jsonify({"error": f"bad meta json: {e}"}), 400

        # 2) Derive a stem for saving outputs
        stem = None
        try:
            stem = os.path.splitext(os.path.basename(meta["image"]["path"]))[0]
        except Exception:
            stem = f"frame_{int(time.time())}"

        # 3) In single-image mode, clear old files then wait so watcher sees the change
        if not cfg.accumulate_mode:
            for old_file in os.listdir(OUT_DIR):
                old_path = os.path.join(OUT_DIR, old_file)
                try:
                    os.remove(old_path)
                except Exception:
                    pass
            time.sleep(1)

        # 4) Save meta JSON (pretty)
        json_name = secure_filename(f"{stem}_dets.json")
        json_path = os.path.join(OUT_DIR, json_name)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[ingest] {stem}: {len(meta.get('detections', []))} det(s)  -> saved")
        return jsonify({"ok": True, "stem": stem}), 200

    except Exception as e:
        app.logger.exception("ingest error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    mode_str = "ACCUMULATE (keep all)" if cfg.accumulate_mode else "SINGLE-IMAGE (clear on new)"
    print(f"Receiver OWL | Mode: {mode_str}")
    app.run(host=cfg.receiver_host, port=cfg.receiver_port, debug=False)