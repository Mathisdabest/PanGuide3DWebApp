"""Run actual raw-CT inference and save masks with explicit provenance."""
import sys, time, json, resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from webcore import ROOT,load_upload
from runtime import infer, DEVICE
from webcore import data_from

case=sys.argv[1] if len(sys.argv)>1 else 'PanTS_00009473'
path=(ROOT/'public-samples'/case/'ct.nii.gz') if case.startswith('MSD_') else ROOT/'samples'/case/'input'/'ct.nii.gz'
original,volume=load_upload(path)
pred,elapsed,peak=infer(volume)
assert pred.pancreas_mask.shape == pred.tumor_mask.shape == volume.array.shape
assert np.isfinite(pred.pancreas_prob).all() and np.isfinite(pred.tumor_prob).all()
assert pred.pancreas_mask.any(), 'Real smoke test: no pancreas predicted'
data=data_from(original,volume,pred,elapsed)
cache=ROOT/'demo-cache'; cache.mkdir(exist_ok=True)
np.savez_compressed(cache/f'{case}.npz',**data)
metrics=dict(case=case,device=DEVICE,inference_seconds=elapsed,gpu_peak_bytes=peak,
    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    input_shape=list(original.GetSize()),model_shape=list(volume.shape),
    output_shape=list(data['tumor'].shape),tumor_voxels=int(data['tumor'].sum()),
    pancreas_voxels=int(data['pan'].sum()),
    tumor_confidence_means=data['tumor_confidence_means'].tolist(),
    tumor_confidence_voxels=data['tumor_confidence_voxels'].tolist(),checkpoint_sha256='917acfa357740361e8d25bbe4287c4e861e67d685ad95ff745b82a41f154a1b0')
(cache/f'{case}.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2),flush=True)
