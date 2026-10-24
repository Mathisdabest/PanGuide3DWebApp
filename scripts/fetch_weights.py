"""Download the supplied checkpoint from a pinned public Space and verify its digest."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent.parent
REVISION='ad26164086bb1c6510418c03b46b81c798408a94'

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def main():
    target=ROOT/'checkpoints'/'panguide3d.pt'
    expected=json.loads(target.with_suffix('.json').read_text())['sha256']
    if target.exists() and digest(target)==expected:
        print('Checkpoint already present; SHA-256 verified.');return
    from huggingface_hub import hf_hub_download
    downloaded=Path(hf_hub_download(repo_id='ilmuuu81/panguide3d',repo_type='space',
        filename='checkpoints/panguide3d.pt',revision=REVISION,token=False,cache_dir=ROOT/'.cache'/'huggingface'))
    if digest(downloaded)!=expected:raise RuntimeError('Downloaded checkpoint digest does not match; refusing to install.')
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(downloaded,target)
    print('Checkpoint installed; SHA-256 verified: '+expected)

if __name__=='__main__':main()
