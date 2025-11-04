import os
import json
from pathlib import Path
from typing import List, Dict

from flask import Flask, request, jsonify, render_template, send_from_directory
import importlib.util


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "web_app" / "uploads"
CORE_FILE = BASE_DIR / "Z-VisionPro-pcb-detection.py"


def load_core_class():
    spec = importlib.util.spec_from_file_location("zvisionpro_core", str(CORE_FILE))
    if spec is None or spec.loader is None:
        raise ImportError("无法加载核心检测模块")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core.ZVisionPro


ZVisionPro = load_core_class()


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Ensure directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Single shared detector instance (lazy-init on first use)
    detector_holder: Dict[str, object] = {"platform": None}

    def get_detector():
        if detector_holder["platform"] is None:
            model_path = os.environ.get("MODEL_PATH") or None
            detector_holder["platform"] = ZVisionPro(model_path=model_path)
        return detector_holder["platform"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/upload")
    def upload():
        if "files" not in request.files:
            return jsonify({"error": "未找到文件字段 'files'"}), 400
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "未选择文件"}), 400

        saved_paths: List[Path] = []
        for f in files:
            if not f.filename:
                continue
            suffix = Path(f.filename).suffix.lower()
            if suffix not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            dst = UPLOADS_DIR / f.filename
            # avoid overwrite
            i = 1
            while dst.exists():
                dst = UPLOADS_DIR / f"{Path(f.filename).stem}_{i}{suffix}"
                i += 1
            f.save(str(dst))
            saved_paths.append(dst)

        if not saved_paths:
            return jsonify({"error": "没有可用的图像文件"}), 400

        platform = get_detector()

        results = []
        for p in saved_paths:
            try:
                r = platform.detect(str(p), save_result=True, output_dir=str(RESULTS_DIR))
                # normalize result image path to web route
                rendered = r.get("rendered_image_path")
                web_image = None
                if rendered:
                    rp = Path(rendered)
                    if rp.exists():
                        web_image = f"/results/{rp.name}"
                # also build json path
                json_path = (RESULTS_DIR / f"{p.stem}_result.json")
                web_json = f"/api/result-json/{json_path.name}" if json_path.exists() else None
                results.append({
                    "image": f"/uploads/{p.name}",
                    "rendered": web_image,
                    "json": web_json,
                    "num_detections": r.get("num_detections", 0),
                    "reliability": r.get("reliability", {}),
                })
            except Exception as e:
                results.append({
                    "image": f"/uploads/{p.name}",
                    "error": str(e),
                })

        return jsonify({"count": len(results), "results": results})

    @app.get("/api/results")
    def list_results():
        items = []
        for img in RESULTS_DIR.glob("*_result.jpg"):
            name = img.name
            j = RESULTS_DIR / name.replace("_result.jpg", "_result.json")
            meta = {}
            if j.exists():
                try:
                    meta = json.loads(j.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            items.append({
                "rendered": f"/results/{name}",
                "json": f"/api/result-json/{j.name}" if j.exists() else None,
                "timestamp": meta.get("timestamp"),
                "num_detections": len(meta.get("detections", [])) if meta else None,
            })
        items.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        return jsonify({"items": items})

    @app.post("/api/clear-results")
    def clear_results():
        removed = 0
        errors: List[str] = []
        try:
            for p in RESULTS_DIR.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                except Exception as e:
                    errors.append(f"{p.name}: {e}")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"removed": removed, "errors": errors})

    @app.get("/api/result-json/<path:filename>")
    def get_result_json(filename: str):
        return send_from_directory(str(RESULTS_DIR), filename, mimetype="application/json")

    @app.get("/results/<path:filename>")
    def get_result_image(filename: str):
        return send_from_directory(str(RESULTS_DIR), filename)

    @app.get("/uploads/<path:filename>")
    def get_upload(filename: str):
        return send_from_directory(str(UPLOADS_DIR), filename)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


