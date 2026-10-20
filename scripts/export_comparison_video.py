import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from swarm_sar.config import EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.training.rollout import HeuristicActor, run_episode
from swarm_sar.evaluation.metrics import episode_metrics
from swarm_sar.visualization.plots import CELL_CMAP

out = root/'results/comparison_demo'
out.mkdir(parents=True,exist_ok=True)
logs = []
for loss in [0.0,1.0]:
 cfg = EnvConfig(num_drones=3,max_steps=180)
 cfg.comm.packet_loss=loss
 if loss==1: cfg.comm.range_m=0
 env=SARSwarmEnv(cfg,seed=200)
 log=run_episode(env,HeuristicActor(env,'auction',200),seed=200)
 logs.append(log)
(out/'video-metrics.json').write_text(json.dumps({
 'controller':'scripted heuristic auction; not a learned policy',
 'backend':'self-contained simulation', 'map_seed':200,
 'intervention':'0% packet loss versus 100% loss plus zero radio range; peer visibility also changes',
 'episodes':[episode_metrics(log,grid_size=64) for log in logs]},indent=2))
plt.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
fig,axes=plt.subplots(1,2,figsize=(12,6),dpi=120)
fig.suptitle('Swarm-SAR | Heuristic coordination in simulation',fontsize=18,weight='bold')
fig.text(.5,.03,'Same map seed 200 | 3 drones | 180 steps | Radio range also changes in disabled condition',ha='center',fontsize=10)
for ax,log in zip(axes,logs):
 ax.imshow(log.world_grid,cmap=CELL_CMAP,vmin=0,vmax=8,origin='lower')
 ax.set_xticks([]); ax.set_yticks([])
trails=[[ax.plot([],[],lw=2,color=c)[0] for c in ['#0077bb','#ee7733','#009988']] for ax in axes]
dots=[ax.scatter([0,0,0],[0,0,0],s=65,c=['#0077bb','#ee7733','#009988'],edgecolors='black') for ax in axes]
victims=[ax.scatter([],[],s=80,marker='*',color='#cc3311',edgecolors='black') for ax in axes]
def update(k):
 for i,(ax,log) in enumerate(zip(axes,logs)):
  k2=min(k,len(log.frames)-1); f=log.frames[k2]
  pos=np.array([x['pos'] for x in log.frames[:k2+1]])
  for d,line in enumerate(trails[i]): line.set_data(pos[:,d,0],pos[:,d,1])
  dots[i].set_offsets(pos[-1,:,:2])
  remaining=[v['pos'][:2] for v in log.victims if v['rescued_step'] is None or v['rescued_step']>f['t']]
  victims[i].set_offsets(np.asarray(remaining).reshape(-1,2))
  label='0% packet loss' if i==0 else 'Communication disabled'
  ax.set_title(f"{label}\nStep {f['t']} | Rescued {f['rescued']}/8 | Coverage {f['coverage']:.0%}",fontsize=12)
 return []
anim=FuncAnimation(fig,update,frames=max(len(x.frames) for x in logs),interval=250,blit=False)
fig.subplots_adjust(top=.82,bottom=.1,wspace=.08)
anim.save(out/'swarm-sar-comparison.mp4',writer=FFMpegWriter(fps=4,bitrate=2200))
update(len(logs[0].frames)-1)
fig.savefig(out/'comparison-preview.png',dpi=150)
print(out/'swarm-sar-comparison.mp4')
