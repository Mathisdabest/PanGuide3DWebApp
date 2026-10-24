"""Deploy only to the explicitly selected free ZeroGPU hardware; never billable."""
import argparse, sys
from pathlib import Path
from huggingface_hub import HfApi, SpaceHardware, get_token
root=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--repo',default='ilmuuu81/panguide3d')
args=parser.parse_args()
if not get_token():
    sys.exit('Local login required. Run .venv/bin/hf auth login in Terminal. Do not paste tokens in chat.')
api=HfApi()
username=api.whoami()['name']
if args.repo.split('/')[0]!=username:
    sys.exit('Repository owner differs from authenticated user; no changes made.')
folder=root/'space-package'
if not (folder/'demo-cache/MSD_pancreas_045.npz').exists():
    sys.exit('Build and package the real MSD demo cache first.')
if api.repo_exists(args.repo,repo_type='space'):
    # Prevent replacing an unrelated existing Space. Update only our own code marker.
    from huggingface_hub import hf_hub_download
    try:
        existing=Path(hf_hub_download(args.repo,'app.py',repo_type='space')).read_text()
    except Exception:
        sys.exit('Existing Space could not be verified as this app; no changes made.')
    if 'PanGuide 3D: research CT segmentation with genuine live and cached results' not in existing:
        sys.exit('Existing Space is not this application; select a new repository name.')
else:
    api.create_repo(args.repo,repo_type='space',space_sdk='gradio',private=False,space_hardware=SpaceHardware.ZERO_A10G)
# This enum is the API identifier for ZeroGPU, not a paid A10G allocation.
api.request_space_hardware(args.repo,hardware=SpaceHardware.ZERO_A10G)
api.add_space_variable(args.repo,key='PANGUIDE_HOST',value='0.0.0.0')
api.upload_folder(repo_id=args.repo,repo_type='space',folder_path=str(folder),commit_message='Deploy real-checkpoint PanGuide 3D research app')
print('Uploaded: https://huggingface.co/spaces/'+args.repo)
print('Build and hosted inference still require verification.')
