#!/usr/bin/env python3
"""Render an explicitly labelled illustrative simulation, not a ROS recording."""
from pathlib import Path
import math, json, subprocess
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

OUT = Path(__file__).resolve().parent
FPS = 30
FONT = str(OUT / 'NotoSansKR.ttf')
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
@lru_cache(None)
def font(n, mono=False): return ImageFont.truetype(MONO if mono else FONT, n)
BG='#0b1220'; PANEL='#121e30'; MUTED='#8fa4bd'; WHITE='#edf4ff'
CYAN='#49dec9'; RED='#ff6e79'; GOLD='#ffd16e'
def pt(x,y): return (int(72+x*42), int(376-y*42))
def simulate(case):
    dt=1/FPS; x=0.8; y=0.8 if case==0 else 0.; yaw=0.; steer=0.; rows=[]; hit=False
    for i in range(180):
        t=i*dt; side=1 if int(t/.65)%2==0 else -1
        pathx=np.linspace(0,21,421)
        pathy=np.zeros_like(pathx) if case!=1 else side*1.05*np.exp(-((pathx-10)/2.5)**2)
        ld=.35 if case==0 else 1.3
        v=3.0
        k=int(np.argmin((pathx-x)**2+(pathy-y)**2))
        j=min(420,k+int(ld/.05))
        alpha=math.atan2(pathy[j]-y,pathx[j]-x)-yaw
        cmd=float(np.clip(math.atan2(2*.33*math.sin(alpha),ld),-.36,.36)) if case!=2 else 0.
        steer+=(cmd-steer)*min(1.,dt/(.28 if case==0 else .18))
        if not hit:
            x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt
            yaw+=v/.33*math.tan(steer)*dt
        gap=math.hypot(10.8-x,y)-.65 if case==2 else math.hypot(10-x,y)-.65
        if case>0 and gap<=0: hit=True
        if case==0 and abs(y)>1.42: hit=True
        rows.append(dict(t=t,x=x,y=y,yaw=yaw,steer=steer,cmd=cmd,v_cmd=v,
                         collision=hit,clearance=gap,side=side,target=[float(pathx[j]),float(pathy[j])]))
    return rows

titles=['PurePursuit','Local planner','TinyLidarNet']
subs=['짧은 lookahead + 조향 지연','회피 후보 경로 전환 예시','최저속도 제한 재현 예시']
issues=['조향 진동 → 경로 추종 불안정','좌·우 회피 경로 전환 → 회피 여유 감소','장애물 접근 중에도 속도 명령 3 m/s']
notes=['운동학 자전거 모델 / Ld 0.35 m / 지연 0.28 s',
       '회피 경로를 0.65초마다 전환한 대리 모델',
       'v_min=3.0 후처리 적용 / 신경망 추론 미사용']
allrows=[simulate(c) for c in range(3)]
def render(case,i):
    rows=allrows[case]; r=rows[i]; im=Image.new('RGB',(1280,720),BG); d=ImageDraw.Draw(im)
    def txt(x,y,s,n=22,c=WHITE,mono=False): d.text((x,y),s,font=font(n,mono),fill=c)
    d.rounded_rectangle((32,26,1248,100),14,fill=PANEL)
    txt(54,39,'ForzaETH  |  튜닝 문제 재현',30)
    txt(858,47,'18s  /  2D SIMULATION',21,CYAN,True)
    txt(44,116,f'0{case+1}   {titles[case]}',34)
    txt(44,164,subs[case],22,MUTED)
    txt(850,125,f'{(case*180+i)/FPS:05.2f} / 18.00 s',20,MUTED,True)
    d.rounded_rectangle((38,210,936,555),12,fill=PANEL)
    for x in range(22):
        d.line([pt(x,-3.8),pt(x,3.8)],fill='#1d2b40')
    for y in range(-3,4): d.line([pt(0,y),pt(20,y)],fill='#1d2b40')
    d.rectangle((*pt(0,1.7),*pt(20,-1.7)),fill='#273447')
    for yy in [-1.7,1.7]: d.line([pt(0,yy),pt(20,yy)],fill='#c5d0dd',width=3)
    for x in np.arange(0,20,.9): d.line([pt(x,0),pt(x+.4,0)],fill='#6f849b',width=2)
    if case>0:
        ox=10 if case==1 else 10.8
        a,b=pt(ox-.4,.4),pt(ox+.4,-.4)
        d.rounded_rectangle((*a,*b),5,fill=RED)
        txt(a[0]-26,a[1]-35,'장애물',18,RED)
    if case==1:
        for side in [-1,1]:
            coords=[pt(float(x),side*1.05*math.exp(-((x-10)/2.5)**2)) for x in np.linspace(1,19,150)]
            d.line(coords,fill=GOLD if side==r['side'] else '#556078',width=3 if side==r['side'] else 1)
    hist=rows[:i+1]
    if len(hist)>1: d.line([pt(v['x'],v['y']) for v in hist],fill=CYAN,width=4)
    if case==0:
        p=pt(*r['target']); d.line([pt(r['x'],r['y']),p],fill=GOLD,width=2)
        d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=GOLD)
    if case==2:
        for angle in np.linspace(-.65,.65,17):
            distance=min(3.5,max(.2,r['clearance']+.4)) if abs(angle)<.18 else 3.5
            d.line([pt(r['x'],r['y']),pt(r['x']+distance*math.cos(angle),r['y']+distance*math.sin(angle))],fill='#386579',width=1)
    def carpoint(a,b): return pt(r['x']+a*math.cos(r['yaw'])-b*math.sin(r['yaw']),r['y']+a*math.sin(r['yaw'])+b*math.cos(r['yaw']))
    d.polygon([carpoint(.38,.18),carpoint(.38,-.18),carpoint(-.38,-.18),carpoint(-.38,.18)],fill=RED if r['collision'] else CYAN)
    d.line([carpoint(0,0),carpoint(.6,0)],fill=WHITE,width=3)
    txt(56,520,'실행 궤적',16,CYAN); txt(170,520,'기준 경로 / 회피 후보',16,MUTED)
    if r['collision']:
        txt(650,236,'충돌 · 시뮬레이션 정지',19,RED)
    d.rounded_rectangle((956,210,1244,555),12,fill=PANEL)
    txt(976,228,'SIM TELEMETRY',18,MUTED,True)
    txt(976,269,'속도 명령',18,MUTED); txt(976,296,'3.00 m/s',28,GOLD,True)
    txt(976,340,'조향각',18,MUTED); txt(976,365,f'{math.degrees(r["steer"]):+05.1f} deg',26,CYAN,True)
    txt(976,411,'조향 기록  (±21°)',17,MUTED)
    d.line((976,488,1220,488),fill='#40506a')
    if i>0: d.line([(976+j/179*244,488-v['steer']/.36*35) for j,v in enumerate(hist)],fill=CYAN,width=2)
    d.rounded_rectangle((38,575,1244,660),12,fill='#223044')
    txt(58,582,issues[case],27,GOLD)
    txt(58,621,notes[case],18,MUTED)
    txt(40,676,'설명용 재현 · 실제 ROS/ForzaETH 실행 녹화 아님 · 수치는 예시 조건',18,MUTED)
    for j in range(3):
        xx=1008+j*79
        d.rounded_rectangle((xx,684,xx+68,690),3,fill=CYAN if j==case else '#33445a')
    return im

if __name__=='__main__':
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    target=OUT/'forzaeth_tuning_18s.mp4'
    proc=subprocess.Popen([ff,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','-', '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for c in range(3):
        for i in range(180): proc.stdin.write(render(c,i).tobytes())
        print(f'Scene {c+1}/3 rendered',flush=True)
    proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
    preview=Image.new('RGB',(1280,720))
    for c in range(3): preview.paste(render(c,110).resize((640,360)),((c%2)*640,(c//2)*360))
    preview.save(OUT/'preview.jpg',quality=92)
    (OUT/'telemetry.json').write_text(json.dumps(allrows,indent=2))
    print(target)
