"""Render the HS-DT/C4 overview and reported findings from tracked result tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
BLUE, TEAL, GOLD, GRAY = '#32668b', '#267d70', '#bc7c38', '#687783'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False})

def box(ax, x, y, w, h, title, detail, color):
    patch = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',
                          facecolor='white',edgecolor=color,linewidth=1.7)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',color=color,weight='bold',fontsize=12)
    ax.text(x+w/2,y+h*.31,detail,ha='center',va='center',color='#26313b',fontsize=10.5)

def arrow(ax, a, b, color=GRAY):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.8,
                'connectionstyle':'arc3,rad=0'})

def architecture():
    fig, ax = plt.subplots(figsize=(12.5,5.1))
    fig.subplots_adjust(left=.015,right=.985,bottom=.09,top=.90)
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    box(ax,.015,.35,.18,.30,'Past 24 hours','7 stations · 34 features\nFixed distance graph',GRAY)
    box(ax,.255,.64,.21,.27,'Eta-only GWN','Residual forecast\n+ temporal context',BLUE)
    box(ax,.255,.14,.21,.27,'Multistate GWN','Residual / current / wave\n+ temporal context',TEAL)
    box(ax,.535,.64,.22,.27,'HS-DT: fixed rule','Leads 1–23: equal average\nLead 24: Multistate',TEAL)
    box(ax,.535,.14,.22,.33,'C4: learned extension','Bounded expert gate\n+ bounded residual correction',GOLD)
    box(ax,.81,.64,.17,.27,'Point forecast','24 residual predictions',TEAL)
    box(ax,.81,.14,.17,.33,'Gaussian forecast','Corrected mean\n+ dynamic scale',GOLD)
    arrow(ax,(.195,.56),(.255,.765)); arrow(ax,(.195,.44),(.255,.265))
    arrow(ax,(.466,.80),(.535,.80),BLUE); arrow(ax,(.466,.35),(.535,.72),TEAL)
    arrow(ax,(.466,.65),(.535,.43),BLUE); arrow(ax,(.466,.25),(.535,.25),TEAL)
    arrow(ax,(.757,.775),(.81,.775),TEAL); arrow(ax,(.757,.30),(.81,.30),GOLD)
    ax.text(.365,.535,'The same trained experts\nare frozen for C4.',ha='center',color=GRAY,fontsize=10)
    fig.suptitle('From two graph experts to residual correction and predictive uncertainty',fontsize=16,weight='bold')
    fig.text(.5,.035,'C4 starts from the HS-DT prediction. Only the gate, correction, and uncertainty head are trained.',ha='center',fontsize=10,color=GRAY)
    fig.savefig(OUT/'hsdt_c4_architecture.png',dpi=180,facecolor='white')
    plt.close(fig)

def findings():
    experts=pd.read_csv(ROOT/'results/main_findings/expert_comparison.csv')
    components=pd.read_csv(ROOT/'results/main_findings/c4_components.csv')
    paired=pd.read_csv(ROOT/'results/main_findings/c4_paired_sequence.csv')
    scales=pd.read_csv(ROOT/'results/controls/scale_scores_by_seed.csv').groupby(['region','scale']).CRPS.mean()
    fig, axes=plt.subplots(2,2,figsize=(12.5,8))
    fig.subplots_adjust(left=.16,right=.98,top=.89,bottom=.20,wspace=.53,hspace=.65)
    for ax,frame,colors,title in [
        (axes[0,0],experts,[BLUE,TEAL,GOLD],'A  Fixed expert combination'),
        (axes[0,1],components,[GRAY,BLUE,TEAL,GOLD],'B  C4 component comparison'),
    ]:
        y=np.arange(len(frame))[::-1]
        ax.scatter(frame.sequence_R2,y,s=90,c=colors,zorder=3)
        ax.set_yticks(y,frame.model)
        ax.set_ylim(-.5,len(frame)-.5)
        ax.set_xlim(.718,.747)
        ax.set_xlabel('Sequence R² · five-run mean')
        ax.set_title(title,loc='left',fontsize=12,weight='bold',pad=14)
        ax.grid(axis='x',alpha=.17)
        for xx,yy in zip(frame.sequence_R2,y):
            ax.text(xx+.0007,yy,f'{xx:.4f}',va='center',fontsize=10)
    ax=axes[1,0]
    yy=np.array([1,0]); d=paired.delta_R2.to_numpy()
    ax.errorbar(d,yy,xerr=np.vstack([d-paired.ci_low,paired.ci_high-d]),fmt='o',color=GOLD,capsize=5,lw=2,markersize=7)
    ax.axvline(0,color=GRAY,lw=1,ls='--')
    ax.set_yticks(yy,paired.region)
    ax.set_ylim(-.6,1.6)
    ax.set_xlim(-.001,.019)
    ax.set_xticks([0, .005, .010, .015], ['0.000', '0.005', '0.010', '0.015'])
    ax.set_xlabel('C4 − matched HS-DT: Δ Sequence R²')
    ax.set_title('C  Paired C4 gains',loc='left',fontsize=12,weight='bold',pad=14)
    ax.grid(axis='x',alpha=.17)
    ax.text(.015,1,'5/5 seeds',va='center',fontsize=9,color=GRAY)
    ax.text(.015,0,'5/5 seeds',va='center',fontsize=9,color=GRAY)
    ax=axes[1,1]
    xs=np.arange(2); width=.31
    f=[scales.loc[(r,'fixed')] for r in ['historical_7','external_10']]
    dy=[scales.loc[(r,'dynamic')] for r in ['historical_7','external_10']]
    ax.bar(xs-width/2,f,width,label='Fixed scale',color=GRAY)
    ax.bar(xs+width/2,dy,width,label='Dynamic scale',color=GOLD)
    for offset,values in [(-width/2,f),(width/2,dy)]:
        for xx,value in zip(xs+offset,values):
            ax.text(xx,value+.0014,f'{value:.4f}',ha='center',fontsize=9)
    ax.set_xticks(xs,['Seven stations','Ten stations'])
    ax.set_ylim(0,.085)
    ax.set_ylabel('CRPS (m) · lower is better')
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax.set_title('D  Same C4 mean, different scale',loc='left',fontsize=12,weight='bold',pad=14)
    fig.suptitle('HS-DT and C4: the main experimental findings',fontsize=17,weight='bold',y=.97)
    fig.text(.5,.048,'A: original historical expert bank. B–C: matched C4 references. C: reported paired 95% intervals; ten-station analysis is post-hoc.\nD: validation-calibrated scales with identical C4 means. These panels summarize different comparisons, not one overall ranking.',ha='center',fontsize=9.2,color=GRAY,linespacing=1.6)
    fig.savefig(OUT/'hsdt_c4_findings.png',dpi=180,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    architecture()
    findings()
    print('Saved HS-DT/C4 figures under docs/figures/')
