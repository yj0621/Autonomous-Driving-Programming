"""Edit actual X11 RViz2 captures: real-time cuts, captions, and magnified crops only."""
from pathlib import Path
from functools import lru_cache
import subprocess,json
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent
FONT=P.parent/'tuning_video/NotoSansKR.ttf'
@lru_cache(None)
def font(size):
    f=ImageFont.truetype(str(FONT),size)
    f.set_variation_by_axes([500])
    return f

SCENES=[
 dict(file='pp_verified_raw.mp4',start=3.3,
      title='PurePursuit · 긴 lookahead 설정에서 벽 충돌',
      params='t_clip_min=1.5  /  t_clip_max=6.0  /  m_l1=1.0  /  q_l1=0.8  /  speed scale=0.65',
      note='코너 안쪽으로 경로 이탈 → 시뮬레이터 wall 충돌 신호 확인',
      detail='청록색: 실제 주행 궤적   |   녹색: local waypoints   |   원: 차량 위치 강조',
      roi=(1210,465,1430,635)),
 dict(file='local_verified_raw.mp4',start=6.0,
      title='Local planner · 회피 경로 생성 실패와 정체',
      params='실제 spline_planner  /  evasion_dist=1.40 m  /  장애물 0.30 m  /  speed scale=0.25',
      note='TRACKBOUNDS 제약으로 회피 취소 · 출력 0점 · TRAILING 상태로 감속',
      detail='실행 로그: Evasion trajectory too close to TRACKBOUNDS, aborting evasion',
      roi=(1250,330,1470,500)),
 dict(file='tln_verified_raw.mp4',start=3.3,
      title='TinyLidarNet · 경로 이탈과 벽 충돌',
      params='실제 TLN_aug.keras 추론  /  기본 v_min=3.0 m/s  /  정적 장애물 3개 배치',
      note='벽 충돌 이후에도 양의 속도 명령 유지 · 정지 상태에서 v_cmd ≈ 4.0 m/s',
      detail='TLN이 직접 구동하며 planner/state 출력은 TLN 제어에 사용하지 않음',
      roi=(740,300,960,470)),
]

def annotate(raw,case,i):
    scene=SCENES[case];im=raw.copy();d=ImageDraw.Draw(im)
    def text(pos,value,size,color='#edf4ff'):d.text(pos,value,font=font(size),fill=color)
    d.rounded_rectangle((478,98,1574,217),12,fill='#101b2c',outline='#344660',width=2)
    text((496,106),f'FORZAETH / hyhall12   ·   실제 RViz2 화면 녹화   ·   {case+1:02d} / 03',21,'#55e4cd')
    text((496,137),scene['title'],30)
    text((496,181),scene['params'],18,'#b6c5d8')
    d.rounded_rectangle((478,758,1574,875),12,fill='#101b2c',outline='#344660',width=2)
    text((496,776),scene['note'],23,'#ffda85')
    text((496,822),scene['detail'],18,'#b6c5d8')
    # Magnify the same source frame; no simulated replacement imagery.
    zoom=raw.crop(scene['roi']).resize((406,314),Image.LANCZOS)
    d.rectangle((8,398,430,768),fill='#101b2c',outline='#55e4cd',width=2)
    text((20,406),'동일 녹화 화면 확대',21,'#55e4cd')
    im.paste(zoom,(16+(406-zoom.width)//2,447))
    d=ImageDraw.Draw(im)
    d.rectangle((0,892,1599,899),fill='#26334a')
    d.rectangle((0,892,int(1600*(case*180+i+1)/540),899),fill='#55e4cd')
    return im

if __name__=='__main__':
    output=P/'forzaeth_rviz2_18s.mp4'
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1600x900','-framerate','30','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    manifest=[]
    for case,scene in enumerate(SCENES):
        decoder=subprocess.Popen(['ffmpeg','-loglevel','error','-threads','2','-ss',str(scene['start']),'-i',str(P/scene['file']),'-t','6','-vf','fps=30','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
        for i in range(180):
            data=decoder.stdout.read(1600*900*3)
            if len(data)!=1600*900*3:raise RuntimeError(f'Truncated source: {scene["file"]} frame {i}')
            im=annotate(Image.frombytes('RGB',(1600,900),data),case,i)
            encoder.stdin.write(im.tobytes())
            if i==115:im.save(P/f'preview_{case+1}.jpg',quality=94)
        decoder.stdout.close()
        if decoder.wait():raise RuntimeError('Decode failed')
        manifest.append({**scene,'duration':6,'playback_speed':1,'output_start':case*6})
        print(f'Edited scene {case+1}/3',flush=True)
    encoder.stdin.close()
    if encoder.wait():raise RuntimeError('Encode failed')
    (P/'edit_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(output)
