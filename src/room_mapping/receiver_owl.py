# receiver_owl.py
from flask import Flask, request, jsonify
import os, json, time
from werkzeug.utils import secure_filename

app = Flask(__name__)
OUT_DIR = "./ingest_out"
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

        # file = request.files.get("image")
        # if file is None:
        #     return jsonify({"error": "missing file part 'image'"}), 400

        # 2) Derive a stem for saving outputs
        # Prefer the original image’s stem if present; else timestamp.
        stem = None
        try:
            stem = os.path.splitext(os.path.basename(meta["image"]["path"]))[0]
        except Exception:
            stem = f"frame_{int(time.time())}"

        # # 3) Save annotated image
        # jpg_name = secure_filename(f"{stem}_ann.jpg")
        # jpg_path = os.path.join(OUT_DIR, jpg_name)
        # file.save(jpg_path)

        # 4) Save meta JSON (pretty)
        json_name = secure_filename(f"{stem}_dets.json")
        json_path = os.path.join(OUT_DIR, json_name)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[ingest] {stem}: {len(meta.get('detections', []))} det(s)  -> saved")
        return jsonify({"ok": True, "stem": stem}), 200

    except Exception as e:
        # Never crash—log and return error
        app.logger.exception("ingest error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=False)