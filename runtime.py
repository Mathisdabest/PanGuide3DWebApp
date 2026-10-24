"""Load the inspected checkpoint once; allocate GPU only for prediction."""
import os, time
import spaces  # must precede CUDA initialization on ZeroGPU
import torch
from panguide3d.model import load_model
from panguide3d.predict import predict

DEVICE = os.environ.get('PANGUIDE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
weights = os.environ.get('PANGUIDE_WEIGHTS')
if os.environ.get('PANGUIDE_MODEL_REPO'):
    from huggingface_hub import hf_hub_download
    revision=os.environ.get('PANGUIDE_MODEL_REVISION')
    if not revision: raise RuntimeError('A pinned PANGUIDE_MODEL_REVISION is required for remote weights')
    weights=hf_hub_download(os.environ['PANGUIDE_MODEL_REPO'],'panguide3d.pt',revision=revision,token=os.environ.get('HF_TOKEN'))
MODEL, DEVICE = load_model(weights, DEVICE)
if DEVICE == 'cpu': torch.set_num_threads(min(4, os.cpu_count() or 1))

@spaces.GPU(duration=60)
def infer(volume):
    start = time.perf_counter()
    if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
    prediction = predict(MODEL, volume, device=DEVICE)
    if DEVICE == 'mps': torch.mps.synchronize()
    elapsed = round(time.perf_counter()-start,3)
    peak = torch.cuda.max_memory_allocated() if DEVICE == 'cuda' else None
    return prediction, elapsed, peak
