import sys,json,time,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
name=sys.argv[1]; cmd=json.loads(sys.argv[2]); duration=float(sys.argv[3]) if len(sys.argv)>3 else 12.
cmd['run_id']=time.time()
f=(P/(name+'_capture.log')).open('w')
proc=subprocess.Popen(['ffmpeg','-y','-f','x11grab','-framerate','30','-video_size','1600x900','-i',':88.0+0,0','-t',str(duration),'-an','-c:v','libx264','-preset','ultrafast','-crf','18','-pix_fmt','yuv420p',str(P/(name+'_raw.mp4'))],stdout=f,stderr=f)
start=time.time(); time.sleep(3.)
(P/'command.tmp').write_text(json.dumps(cmd));(P/'command.tmp').replace(P/'command.json')
(P/(name+'_capture.json')).write_text(json.dumps({'capture_wall_start':start,'command_wall_time':time.time(),'command':cmd,'duration':duration},indent=2))
result=proc.wait();f.close()
if result:raise RuntimeError('Capture failed')
print(P/(name+'_raw.mp4'))
