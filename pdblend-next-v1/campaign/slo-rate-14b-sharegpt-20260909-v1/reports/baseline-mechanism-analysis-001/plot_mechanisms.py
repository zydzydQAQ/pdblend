"""Render the measured C 1.5 rps comparison; no measurement or source-data edits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE/'power-mechanisms-evidence.json').read_text())
systems = ['pdblend','mixed','distserve','ecoserve','dynamollm']
names = ['PDBlend','Mixed','DistServe','EcoServe','DynamoLLM']
colors = ['#cb4539','#b88713','#16836e','#6784a0','#8752a8']
rows = [next(r for r in d['rows'] if r['node']=='C' and r['rate_rps']==1.5 and
             r['repeat']==1 and r['system']==s) for s in systems]
fig, axes = plt.subplots(2,2,figsize=(13.5,8.2),layout='constrained')
fig.suptitle('14B ShareGPT | host C | SLO 2x | offered rate 1.50 req/s\n'
             'Same 122 requests; one seed; primary window includes arrivals and actual drain',fontsize=15)
panels=[('SLO attainment (%)',[100*r['slo_attainment'] for r in rows],'.2f'),
        ('Eight-GPU energy (kJ)',[r['energy_j']/1000 for r in rows],'.2f'),
        ('Mean eight-GPU power (W)',[r['average_all8_power_w'] for r in rows],'.0f'),
        ('Primary measurement duration (s)',[r['duration_s'] for r in rows],'.2f')]
for k,(ax,(ylabel,values,fmt)) in enumerate(zip(axes.flat,panels)):
    if k==2:
        base=[r['zero_sampled_util_gpu_power_w'] for r in rows]
        ax.bar(names,base,color='#b6bcc5',label='GPUs with zero sampled utilization')
        ax.bar(names,[v-b for v,b in zip(values,base)],bottom=base,color=colors)
        ax.legend(loc='upper left',fontsize=8,framealpha=.8)
    else:
        ax.bar(names,values,color=colors)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,max(values)*(1.25 if k==2 else 1.13))
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for i,v in enumerate(values):
        if k==0 and v<90:
            ax.text(i,v-4,format(v,fmt),ha='center',va='top',color='white',fontsize=10)
        else:
            ax.text(i,v+max(values)*.025,format(v,fmt),ha='center',fontsize=10)
    if k==0:
        ax.axhline(90,color='#565d66',lw=1,ls='--')
        ax.text(.99,1.02,'Dashed line: 90% target',transform=ax.transAxes,ha='right',fontsize=8)
    ax.tick_params(axis='x',labelsize=10)
for ext in ['png','svg','pdf']:
    fig.savefig(HERE/f'C-r1.5-energy-slo-mechanisms.{ext}',dpi=180)
plt.close(fig)
print(HERE/'C-r1.5-energy-slo-mechanisms.png')
