import requests,json,time
from huggingface_hub import get_token
from collections import deque
import sys
kind=sys.argv[1] if len(sys.argv)>1 else 'build'
lines=deque(maxlen=25)
try:
    with requests.get('https://huggingface.co/api/spaces/ilmuuu81/panguide3d/logs/'+kind,headers={'Authorization':'Bearer '+get_token()},stream=True,timeout=(10,8)) as response:
        response.raise_for_status(); start=time.monotonic()
        for line in response.iter_lines():
            if line:
                text=line.decode(errors='replace')
                if text.startswith('data:'):
                    try: text=json.loads(text[5:]).get('data',text)
                    except Exception: pass
                lines.append(text)
            if time.monotonic()-start>12: break
except requests.RequestException as error:
    print('Log stream ended:',type(error).__name__)
print('\n'.join(lines))
