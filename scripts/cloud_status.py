from huggingface_hub import HfApi
import json
api=HfApi();runtime=api.get_space_runtime('ilmuuu81/panguide3d')
print(json.dumps(dict(stage=str(runtime.stage),hardware=str(runtime.hardware),requested_hardware=str(runtime.requested_hardware),raw=runtime.raw),default=str,indent=2))
