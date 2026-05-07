#!/usr/bin/env python3
"""
comicml training dashboard — FastAPI server (port 8001).

Start with:
    python3 webapp/train/server.py
or:
    cd webapp/train && uvicorn server:app --port 8001 --reload
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PROJECT = _HERE.parent.parent  # <root>/webapp/train/server.py → <root>
_MODELS_DIR = _PROJECT / "models"
_DATA_DIR   = _PROJECT / "data"
_GT_FILE    = _DATA_DIR / "ground_truth.json"

sys.path.insert(0, str(_PROJECT))

def comicscan_dir() -> Path:
    return _MODELS_DIR

def ground_truth_file() -> Path:
    return _GT_FILE

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="comicml training dashboard")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse(str(_HERE / "static" / "index.html"))

# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
_active_runs: Dict[str, Any] = {}  # run_id → {proc, log_path, status}

class TrainConfig(BaseModel):
    train: str
    holdout: str
    epochs: int = 120
    batch_size: int = 8
    lr: float = 1e-4
    input_size: int = 768
    warm_restarts: int = 40
    seed: int = 137
    output: Optional[str] = None

@app.get("/api/train/defaults")
async def train_defaults():
    import re
    src = (_PROJECT / "comicml" / "train.py").read_text()
    train_m   = re.search(r'p_train\.add_argument\("--train",\s*default="([^"]+)"', src)
    holdout_m = re.search(r'p_train\.add_argument\("--holdout",\s*default="([^"]+)"', src)
    return {
        "train":   train_m.group(1)   if train_m   else "",
        "holdout": holdout_m.group(1) if holdout_m else "",
    }

@app.post("/api/train/start")
async def train_start(cfg: TrainConfig):
    run_id = str(uuid.uuid4())[:8]

    output_name = cfg.output or f"comicml_model_run_{run_id}.pt"
    if not Path(output_name).is_absolute():
        output_path = _MODELS_DIR / output_name
    else:
        output_path = Path(output_name)

    log_path = output_path.with_name(output_path.stem + "_log.jsonl")

    cmd = [
        sys.executable, "-m", "comicml.train", "train",
        "--train", cfg.train,
        "--holdout", cfg.holdout,
        "--epochs", str(cfg.epochs),
        "--batch-size", str(cfg.batch_size),
        "--lr", str(cfg.lr),
        "--input-size", str(cfg.input_size),
        "--warm-restarts", str(cfg.warm_restarts),
        "--seed", str(cfg.seed),
        "--output", str(output_path),
    ]
    proc = subprocess.Popen(cmd, cwd=str(_PROJECT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    _active_runs[run_id] = {
        "proc": proc,
        "log_path": log_path,
        "status": "running",
        "output_path": output_path,
    }
    return {"run_id": run_id, "log_path": str(log_path)}

@app.get("/api/train/{run_id}/stream")
async def train_stream(run_id: str):
    """SSE stream that tails the JSONL log file and emits epoch events."""
    run = _active_runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    async def event_generator():
        log_path: Path = run["log_path"]
        seen_lines = 0
        while True:
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                for line in lines[seen_lines:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield f"event: epoch\ndata: {json.dumps(data)}\n\n"
                        seen_lines += 1
                    except json.JSONDecodeError:
                        pass

            proc = run["proc"]
            if proc.poll() is not None:
                # Process finished — flush any remaining lines then close
                await asyncio.sleep(0.5)
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    for line in lines[seen_lines:]:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            yield f"event: epoch\ndata: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError:
                            pass
                if proc.returncode == 0:
                    run["status"] = "done"
                    yield 'event: done\ndata: {"status":"done"}\n\n'
                else:
                    run["status"] = "error"
                    yield f'event: error_msg\ndata: {{"message":"exit code {proc.returncode}"}}\n\n'
                return

            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/api/train/{run_id}/stop")
async def train_stop(run_id: str):
    run = _active_runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    run["proc"].terminate()
    run["status"] = "stopped"
    return {"status": "stopped"}

@app.get("/api/train/status")
async def train_status():
    for run_id, run in _active_runs.items():
        if run["status"] == "running":
            return {"status": "running", "run_id": run_id}
    return {"status": "idle"}

@app.get("/api/train/logs")
async def train_logs():
    """List all JSONL log files in comicscan_dir."""
    cs_dir = comicscan_dir()
    if not cs_dir.exists():
        return []
    logs = []
    for f in sorted(cs_dir.glob("*_log.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            lines = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            last = lines[-1]
            best = min((l["val_px"] for l in lines), default=None)
            logs.append({
                "filename":     f.name,
                "total_epochs": last.get("total_epochs", len(lines)),
                "best_val_px":  round(best, 3) if best is not None else None,
            })
        except Exception:
            pass
    return logs

@app.get("/api/train/logs/{filename}")
async def train_log_detail(filename: str):
    cs_dir = comicscan_dir()
    path = cs_dir / filename
    if not path.exists() or not path.name.endswith("_log.jsonl"):
        raise HTTPException(404, "Log not found")
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return lines

@app.get("/api/train/watch/{filename}")
async def train_watch(filename: str):
    """SSE endpoint that tails an existing log file live — for externally-launched runs."""
    cs_dir = comicscan_dir()
    log_path = cs_dir / filename
    if not filename.endswith("_log.jsonl"):
        raise HTTPException(400, "Not a log file")

    async def event_generator():
        seen_lines  = 0
        stale_ticks = 0
        last_data   = None
        # Epochs take ~60–120 s each; only declare "done" after 5 min of silence
        STALE_LIMIT = 300
        while True:
            if log_path.exists():
                raw_lines = log_path.read_text().splitlines()
                new = raw_lines[seen_lines:]
                for line in new:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        last_data = data
                        yield f"event: epoch\ndata: {json.dumps(data)}\n\n"
                        seen_lines  += 1
                        stale_ticks  = 0
                    except json.JSONDecodeError:
                        seen_lines += 1  # skip unparseable lines so we don't re-read them
                # If the last epoch we've emitted is the final epoch, we're done
                if last_data and last_data.get("epoch") == last_data.get("total_epochs"):
                    yield 'event: done\ndata: {"status":"done"}\n\n'
                    return
                if not new:
                    stale_ticks += 1
                    if stale_ticks >= STALE_LIMIT:
                        yield 'event: done\ndata: {"status":"done"}\n\n'
                        return
                else:
                    stale_ticks = 0
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
PAGE_SIZE = 60

@app.get("/api/dataset/entries")
async def dataset_entries(scan_dir: str = "", has_correction: str = "", search: str = "",
                          offset: int = 0, limit: int = PAGE_SIZE):
    gt_path = ground_truth_file()
    if not gt_path.exists():
        raise HTTPException(404, "ground_truth.json not found")
    entries = json.loads(gt_path.read_text())

    if scan_dir:
        entries = [e for e in entries if e["scan_dir"].endswith("/" + scan_dir) or e["scan_dir"] == scan_dir]
    if has_correction == "true":
        entries = [e for e in entries if e.get("has_correction")]
    elif has_correction == "false":
        entries = [e for e in entries if not e.get("has_correction")]
    if search:
        s = search.lower()
        entries = [e for e in entries if s in e["filepath"].lower() or s in e["scan_dir"].lower()]

    total = len(entries)
    page = entries[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": [
            {k: e[k] for k in ("scan_dir","page_index","filepath","dpi","image_width","image_height",
                                "gt_corners","det_corners","has_correction")
             if k in e}
            for e in page
        ],
    }

@app.get("/api/dataset/image/{b64path}")
async def dataset_image(b64path: str, max_size: int = 400):
    try:
        filepath = base64.b64decode(b64path.encode()).decode()
    except Exception:
        raise HTTPException(400, "Invalid path encoding")
    path = Path(filepath)
    if not path.exists():
        raise HTTPException(404, "Image not found")
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(404, "Could not read image")
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "Encode failed")
    return Response(content=bytes(buf), media_type="image/jpeg")

@app.get("/api/dataset/stats")
async def dataset_stats():
    gt_path = ground_truth_file()
    if not gt_path.exists():
        return {"total": 0, "corrected": 0, "dirs": []}
    entries = json.loads(gt_path.read_text())
    dirs = sorted(set(e["scan_dir"].rsplit("/", 1)[-1] for e in entries))
    corrected = sum(1 for e in entries if e.get("has_correction"))
    return {"total": len(entries), "corrected": corrected, "dirs": dirs}

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
_eval_tasks: Dict[str, Any] = {}

@app.post("/api/eval/start")
async def eval_start(model: str):
    task_id = str(uuid.uuid4())[:8]
    cs_dir = comicscan_dir()
    model_path = cs_dir / model if not Path(model).is_absolute() else Path(model)
    if not model_path.exists():
        raise HTTPException(404, f"Model not found: {model_path}")

    _eval_tasks[task_id] = {"status": "running", "n_pages": 0, "error": None, "results": None}

    async def run_eval():
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _run_eval_sync, model_path)
            _eval_tasks[task_id].update({"status": "done", "n_pages": len(results["per_page"]), "results": results})
        except Exception as e:
            _eval_tasks[task_id].update({"status": "error", "error": str(e)})

    asyncio.create_task(run_eval())
    return {"task_id": task_id}

def _run_eval_sync(model_path: Path):
    """Run evaluation in a thread executor."""
    from comicml.train import _load_entries, _split_entries, _predict_single, INPUT_SIZE
    from comicml.models import CornerRegressor, CornerHeatmapRegressor

    gt_path = ground_truth_file()
    entries = json.loads(gt_path.read_text())

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "regression")
    if model_type == "heatmap":
        model = CornerHeatmapRegressor(pretrained=False)
    else:
        model = CornerRegressor(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._input_size = ckpt.get("input_size", INPUT_SIZE)
    model._model_type = model_type

    device = torch.device("cpu")
    holdout_dirs = set(ckpt.get("holdout_dirs", []))
    train_dirs   = set(ckpt.get("train_dirs", []))

    eval_entries = [e for e in entries if e.get("has_correction")]
    per_page = []
    per_dir: Dict[str, list] = {}

    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None:
            continue
        if entry.get("gt_rotate180"):
            img = cv2.rotate(img, cv2.ROTATE_180)

        size = getattr(model, "_input_size", INPUT_SIZE)
        pred_a = _predict_single(model, device, img, size)
        W = img.shape[1]
        flipped = cv2.flip(img, 1)
        pred_b = _predict_single(model, device, flipped, size)
        pred_b = [[W - x, y] for x, y in pred_b]
        pred_b = [pred_b[1], pred_b[0], pred_b[3], pred_b[2]]
        pred = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in zip(pred_a, pred_b)]

        gt = entry["gt_corners"]
        d = float(np.mean([np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred, gt)]))

        name = entry["scan_dir"].rsplit("/", 1)[-1]
        split = "holdout" if name in holdout_dirs else ("train" if name in train_dirs else "other")

        per_page.append({
            "filepath":   entry["filepath"],
            "scan_dir":   entry["scan_dir"],
            "page_index": entry["page_index"],
            "error_px":   round(d, 2),
            "pred":       [[round(x, 1), round(y, 1)] for x, y in pred],
            "gt":         gt,
            "split":      split,
        })
        per_dir.setdefault(name, {"errors": [], "split": split})["errors"].append(d)

    errs = np.array([p["error_px"] for p in per_page])
    summary = {
        "mean":   round(float(errs.mean()), 3) if len(errs) else None,
        "median": round(float(np.median(errs)), 3) if len(errs) else None,
        "p95":    round(float(np.percentile(errs, 95)), 3) if len(errs) else None,
        "max":    round(float(errs.max()), 3) if len(errs) else None,
    }
    per_dir_out = {}
    for name, d in per_dir.items():
        v = np.array(d["errors"])
        per_dir_out[name] = {
            "n":      len(v),
            "mean":   round(float(v.mean()), 3),
            "median": round(float(np.median(v)), 3),
            "max":    round(float(v.max()), 3),
            "split":  d["split"],
        }
    return {"summary": summary, "per_dir": per_dir_out, "per_page": per_page}

@app.get("/api/eval/{task_id}/status")
async def eval_status(task_id: str):
    task = _eval_tasks.get(task_id)
    if not task:
        raise HTTPException(404)
    return {k: v for k, v in task.items() if k != "results"}

@app.get("/api/eval/{task_id}/results")
async def eval_results(task_id: str):
    task = _eval_tasks.get(task_id)
    if not task or task["status"] != "done":
        raise HTTPException(404)
    return task["results"]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _read_ensemble_config() -> list:
    ec = _MODELS_DIR / "ensemble_config.json"
    if ec.exists():
        try:
            return json.loads(ec.read_text()).get("models", [])
        except Exception:
            pass
    # Fall back to ENSEMBLE_MODELS from comicml package
    try:
        from comicml import ENSEMBLE_MODELS
        return list(ENSEMBLE_MODELS)
    except Exception:
        pass
    return []

def _write_ensemble_config(models: list):
    ec = _MODELS_DIR / "ensemble_config.json"
    ec.write_text(json.dumps({"models": models}, indent=2) + "\n")

@app.get("/api/models/list")
async def models_list():
    cs_dir = comicscan_dir()
    if not cs_dir.exists():
        return []
    ensemble = _read_ensemble_config()
    results = []
    for pt in sorted(cs_dir.glob("*.pt")):
        entry: Dict[str, Any] = {
            "filename":    pt.name,
            "size_mb":     round(pt.stat().st_size / 1e6, 1),
            "in_ensemble": pt.name in ensemble,
            "model_type":  None, "input_size": None, "val_px": None,
            "epoch":       None, "seed": None, "train_dirs": None, "holdout_dirs": None,
        }
        try:
            ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
            entry.update({
                "model_type":   ckpt.get("model_type"),
                "input_size":   ckpt.get("input_size"),
                "val_px":       round(float(ckpt["val_px"]), 3) if "val_px" in ckpt else None,
                "epoch":        ckpt.get("epoch"),
                "seed":         ckpt.get("seed"),
                "train_dirs":   ckpt.get("train_dirs"),
                "holdout_dirs": ckpt.get("holdout_dirs"),
            })
        except Exception:
            pass
        results.append(entry)
    return results

class EnsembleModify(BaseModel):
    filename: str

@app.post("/api/ensemble/add")
async def ensemble_add(body: EnsembleModify):
    models = _read_ensemble_config()
    if body.filename not in models:
        models.append(body.filename)
    _write_ensemble_config(models)
    return {"models": models}

@app.post("/api/ensemble/remove")
async def ensemble_remove(body: EnsembleModify):
    models = [m for m in _read_ensemble_config() if m != body.filename]
    _write_ensemble_config(models)
    return {"models": models}

# ---------------------------------------------------------------------------
# Config (read-only — paths are derived from project layout)
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def config_get():
    return {
        "project_root":       str(_PROJECT),
        "models_dir":         str(_MODELS_DIR),
        "ground_truth_file":  str(_GT_FILE),
    }

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
