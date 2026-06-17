from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import modal

APP_DIR = "/app"
HF_CACHE_DIR = "/cache/huggingface"
OUTPUT_DIR = "/outputs"
REPO_ROOT = Path(__file__).parent

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_DATA_PATH = "data/training_data.jsonl"
DEFAULT_OUT_DIR = "probe_out_public_safety"
DEFAULT_PROMPT = "Explain SQL injection at a high level"
DEFAULT_MAX_EXAMPLES = 20_000

FALLBACK_PROJECT_DEPENDENCIES = [
    "torch>=2.2.0",
    "transformers>=4.55.0",
    "accelerate>=0.33.0",
    "datasets>=2.20.0",
    "safetensors>=0.4.3",
    "scikit-learn>=1.4.0",
    "tqdm>=4.66.0",
    "numpy>=1.26.0",
    "pillow>=12.2.0",
    "torchvision>=0.27.0",
]


def _project_dependencies() -> list[str]:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return FALLBACK_PROJECT_DEPENDENCIES

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


app = modal.App("open-constitution")

hf_cache = modal.Volume.from_name("open-constitution-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("open-constitution-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_project_dependencies())
    .add_local_dir(
        str(REPO_ROOT / "activation_probe_mvp"),
        remote_path=f"{APP_DIR}/activation_probe_mvp",
        copy=True,
    )
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=f"{APP_DIR}/scripts", copy=True)
    .add_local_dir(str(REPO_ROOT / "data"), remote_path=f"{APP_DIR}/data", copy=True)
)


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def train_probe(
    model_id: str = DEFAULT_MODEL_ID,
    data_path: str = DEFAULT_DATA_PATH,
    layer: int = -4,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = 100,
    lr: float = 1e-3,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    sample_seed: int = 0,
    no_chat_template: bool = False,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR

    remote_data_path = data_path if data_path.startswith("/") else f"{APP_DIR}/{data_path}"
    remote_out_dir = f"{OUTPUT_DIR}/{out_dir}"

    command = [
        "python",
        "-m",
        "scripts.train_probe",
        "--model_id",
        model_id,
        "--data_path",
        remote_data_path,
        "--layer",
        str(layer),
        "--out_dir",
        remote_out_dir,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--max_examples",
        str(max_examples),
        "--sample_seed",
        str(sample_seed),
    ]

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        subprocess.run(command, cwd=APP_DIR, check=True)
    finally:
        hf_cache.commit()
        outputs.commit()

    return remote_out_dir


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_guarded_generate(
    prompt: str = DEFAULT_PROMPT,
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    max_new_tokens: int = 80,
    no_chat_template: bool = False,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    outputs.reload()

    remote_probe_dir = f"{OUTPUT_DIR}/{probe_dir}"
    command = [
        "python",
        "-m",
        "scripts.guarded_generate",
        "--probe_path",
        f"{remote_probe_dir}/probe.pt",
        "--config_path",
        f"{remote_probe_dir}/config.json",
        "--prompt",
        prompt,
        "--max_new_tokens",
        str(max_new_tokens),
    ]

    if model_id is not None:
        command.extend(["--model_id", model_id])

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        subprocess.run(command, cwd=APP_DIR, check=True)
    finally:
        hf_cache.commit()

    return f"Guarded generation completed with probe output directory: {remote_probe_dir}"


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    data_path: str = DEFAULT_DATA_PATH,
    layer: int = -4,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = 100,
    lr: float = 1e-3,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    sample_seed: int = 0,
    no_chat_template: bool = False,
):
    remote_out_dir = train_probe.remote(
        model_id=model_id,
        data_path=data_path,
        layer=layer,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        max_examples=max_examples,
        sample_seed=sample_seed,
        no_chat_template=no_chat_template,
    )
    print(f"Saved probe outputs to Modal Volume path: {remote_out_dir}")


@app.local_entrypoint()
def generate(
    prompt: str = DEFAULT_PROMPT,
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    max_new_tokens: int = 80,
    no_chat_template: bool = False,
):
    result = run_guarded_generate.remote(
        prompt=prompt,
        model_id=model_id,
        probe_dir=probe_dir,
        max_new_tokens=max_new_tokens,
        no_chat_template=no_chat_template,
    )
    print(result)
