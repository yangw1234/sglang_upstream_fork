from dataclasses import dataclass, field
import time
import os
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"

@dataclass
class Instance:
    model_name: str
    port: int
    trace_file: str
    model_path: Optional[str] = None

instances = [
    Instance(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        port=30001,
        trace_file="/novita_trace/multi_llm/meta_llama_Llama_3.1_8B_Instruct_2025_04_07_30d.csv",
        model_path="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/"
    ),
    Instance(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        port=30002,
        trace_file="/novita_trace/multi_llm/meta_llama_Llama_3.1_8B_Instruct_2025_04_07_30d.csv",
        model_path="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/"
    ),
]

def launch_servers():
    import os
    os.makedirs("./logs/servers/", exist_ok=True)
    serve_processes = []
    
    for instance in instances:
        # use popen to launch server in background
        import subprocess
        cmd = ["python", "-m", "sglang.launch_server",
               "--model", instance.model_name if instance.model_path is None else instance.model_path,
               "--disable-radix-cache",
               "--port", str(instance.port)]
        serve_process = subprocess.Popen(cmd, stdout=open(f"./logs/servers/server_{instance.port}.log", "w"),
                         stderr=subprocess.STDOUT)
        serve_processes.append(serve_process)
    
    for instance in instances:
        # check sglang server is up
        import time
        import requests
        url = f"http://localhost:{instance.port}/get_model_info"
        for _ in range(30):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server for {instance.model_name} is up at {url}")
                    break
            except requests.ConnectionError:
                pass
            time.sleep(2)
    return serve_processes

def launch_benchmarks():
    processes = []
    os.makedirs("./logs/benchmarks/", exist_ok=True)
    for rank, instance in enumerate(instances):
        import subprocess
        url = f"http://localhost:{instance.port}"
        cmd = [
            "python3", "python/sglang/bench_serving.py",
            "--backend", "sglang",
            "--base-url", url,
            "--model", instance.model_name if instance.model_path is None else instance.model_path,
            "--dataset-name", "novita",
            "--dataset-path", instance.trace_file,
            "--start-time-stamp", "2025-04-06 00:00:00.000000",
            "--end-time-stamp", "2025-04-06 00:00:03.000000",
        ]
        env = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": str(len(instances)),
            "RANK": str(rank)
        }
        processes.append(subprocess.Popen(cmd, env=env, stdout=open(f"./logs/benchmarks/benchmark_{instance.port}.log", "w"),
                         stderr=subprocess.STDOUT))
    
    for p in processes:
        p.wait()


if __name__ == "__main__":
    serve_processes = launch_servers()
    launch_benchmarks()
    for p in serve_processes:
        p.terminate()
    for p in serve_processes:
        p.wait()