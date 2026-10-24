"""Combine actual browser capture with synthesized narration, without altering results."""
from pathlib import Path
import subprocess, imageio_ffmpeg
root=Path(__file__).resolve().parent.parent
video=root/'recordings/full-demo/PanGuide3D-browser-180s.webm'
audio=root/'recordings/narration.aiff'
output=root/'deliverables/PanGuide3D-demo-3min.mp4'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-y','-i',str(video),'-i',str(audio),
    '-map','0:v:0','-map','1:a:0','-vf','fps=30','-c:v','libx264','-preset','fast','-crf','21','-threads','2',
    '-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-af','apad','-t','180','-movflags','+faststart',str(output)],check=True)
print(output)
