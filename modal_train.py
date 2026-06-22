from __future__ import annotations

import os
import subprocess
import tomllib
from collections import deque
from pathlib import Path

import modal

APP_DIR = "/app"
HF_CACHE_DIR = "/cache/huggingface"
OUTPUT_DIR = "/outputs"
REPO_ROOT = Path(__file__).parent

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_CLASSIFIER_MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_DATA_PATH = "data/examples.jsonl"
DEFAULT_CLASSIFIER_DATA_PATH = "data/training_data.jsonl"
DEFAULT_OUT_DIR = "probe_out_gemma4"
DEFAULT_CLASSIFIER_OUT_DIR = "classifier_out_modernbert"
DEFAULT_PROMPT = "Explain SQL injection at a high level"

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


def _read_last_lines(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""

    with path.open("r", encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=max_lines))


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
    timeout=24 * 60 * 60,
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
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_train_exchange_classifier(
    model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    data_path: str = DEFAULT_CLASSIFIER_DATA_PATH,
    out_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    prefix_augment: bool = True,
    logging_steps: int = 500,
    save_steps: int = 5000,
    resume_from_checkpoint: str | None = None,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    remote_data_path = data_path if data_path.startswith("/") else f"{APP_DIR}/{data_path}"
    remote_out_dir = f"{OUTPUT_DIR}/{out_dir}"
    remote_out_path = Path(remote_out_dir)
    remote_out_path.mkdir(parents=True, exist_ok=True)
    log_path = remote_out_path / "train.log"

    command = [
        "python",
        "-u",
        "-m",
        "scripts.train_exchange_classifier",
        "--model_id",
        model_id,
        "--data_path",
        remote_data_path,
        "--output-dir",
        remote_out_dir,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--max_length",
        str(max_length),
        "--logging_steps",
        str(logging_steps),
        "--save_steps",
        str(save_steps),
        "--disable_tqdm",
    ]

    if prefix_augment:
        command.append("--prefix_augment")

    if resume_from_checkpoint is not None:
        checkpoint_path = (
            resume_from_checkpoint
            if resume_from_checkpoint.startswith("/")
            else f"{OUTPUT_DIR}/{resume_from_checkpoint}"
        )
        command.extend(["--resume_from_checkpoint", checkpoint_path])

    try:
        print(f"Training classifier. Logs are being written to: {log_path}")
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                cwd=APP_DIR,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        print(f"Classifier training completed. Logs: {log_path}")
    except subprocess.CalledProcessError:
        print(f"Classifier training failed. Last lines from {log_path}:")
        print(_read_last_lines(log_path))
        raise
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
    classifier_dir: str | None = None,
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

    if classifier_dir is not None:
        remote_classifier_dir = f"{OUTPUT_DIR}/{classifier_dir}"
        command.extend(["--classifier_dir", remote_classifier_dir])

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        subprocess.run(command, cwd=APP_DIR, check=True)
    finally:
        hf_cache.commit()

    return f"Guarded generation completed with probe output directory: {remote_probe_dir}"


@app.function(
    image=image,
    gpu="A100",
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_benchmark_latency(
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    max_new_tokens: int = 64,
    prompts: list[str] | None = None,
    no_chat_template: bool = False,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    outputs.reload()

    remote_probe_dir = f"{OUTPUT_DIR}/{probe_dir}"
    remote_classifier_dir = f"{OUTPUT_DIR}/{classifier_dir}"
    out_path = f"{OUTPUT_DIR}/latency_benchmark.json"
    command = [
        "python",
        "-m",
        "scripts.benchmark_latency",
        "--probe_path",
        f"{remote_probe_dir}/probe.pt",
        "--config_path",
        f"{remote_probe_dir}/config.json",
        "--classifier_dir",
        remote_classifier_dir,
        "--max_new_tokens",
        str(max_new_tokens),
        "--out_path",
        out_path,
    ]

    if model_id is not None:
        command.extend(["--model_id", model_id])

    if prompts:
        for prompt in prompts:
            command.extend(["--prompt", prompt])

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    finally:
        hf_cache.commit()
        outputs.commit()

    return out_path


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    data_path: str = DEFAULT_DATA_PATH,
    layer: int = -4,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = 100,
    lr: float = 1e-3,
    no_chat_template: bool = False,
):
    remote_out_dir = train_probe.remote(
        model_id=model_id,
        data_path=data_path,
        layer=layer,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        no_chat_template=no_chat_template,
    )
    print(f"Saved probe outputs to Modal Volume path: {remote_out_dir}")


@app.local_entrypoint()
def train_classifier(
    model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    data_path: str = DEFAULT_CLASSIFIER_DATA_PATH,
    output_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    prefix_augment: bool = True,
    logging_steps: int = 500,
    save_steps: int = 5000,
    resume_from_checkpoint: str | None = None,
):
    remote_out_dir = run_train_exchange_classifier.remote(
        model_id=model_id,
        data_path=data_path,
        out_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        prefix_augment=prefix_augment,
        logging_steps=logging_steps,
        save_steps=save_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print(f"Saved classifier outputs to Modal Volume path: {remote_out_dir}")


@app.local_entrypoint()
def generate(
    prompt: str = DEFAULT_PROMPT,
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str | None = None,
    max_new_tokens: int = 80,
    no_chat_template: bool = False,
):
    result = run_guarded_generate.remote(
        prompt=prompt,
        model_id=model_id,
        probe_dir=probe_dir,
        classifier_dir=classifier_dir,
        max_new_tokens=max_new_tokens,
        no_chat_template=no_chat_template,
    )
    print(result)


@app.local_entrypoint()
def benchmark_latency(
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    max_new_tokens: int = 64,
    no_chat_template: bool = False,
):
    out_path = run_benchmark_latency.remote(
        model_id=model_id,
        probe_dir=probe_dir,
        classifier_dir=classifier_dir,
        max_new_tokens=max_new_tokens,
        no_chat_template=no_chat_template,
    )
    print(f"Latency benchmark written to Modal Volume path: {out_path}")
