"""Create a clean deployment folder. Never publishes credentials or PanTS data."""
from pathlib import Path
import shutil, json
root=Path(__file__).resolve().parent.parent
out=root/'space-package'
out.mkdir(exist_ok=True)
for name in ['app.py','runtime.py','webcore.py','requirements.txt','LICENSE']:
    shutil.copy2(root/name,out/name)
for name in ['panguide3d','reference','public-samples']:
    shutil.copytree(root/name,out/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
(out/'checkpoints').mkdir(exist_ok=True)
shutil.copy2(root/'checkpoints/panguide3d.json',out/'checkpoints/panguide3d.json')
# The user supplied these weights for the application; preserve attribution alongside them.
shutil.copy2(root/'checkpoints/panguide3d.pt',out/'checkpoints/panguide3d.pt')
(out/'demo-cache').mkdir(exist_ok=True)
for p in (root/'demo-cache').glob('MSD_*'):
    shutil.copy2(p,out/'demo-cache'/p.name)
shutil.copy2(root/'docs/SPACE_README.md',out/'README.md')
shutil.copy2(root/'docs/MODEL_AUDIT.md',out/'MODEL_AUDIT.md')
assert not (out/'samples').exists()
assert not (out/'.venv').exists()
print(out)
