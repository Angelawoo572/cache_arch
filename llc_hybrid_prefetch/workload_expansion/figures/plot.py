"""Draw independent native measurements; use the committed CSVs only."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
PANELS=("screen","transfer","transfer/long_google")


def read(path):
    with (STUDY/path).open() as f:
        return list(csv.DictReader(f))


def overview():
    cases=[r["case"] for r in read("intake/registry.csv")
           if r["split"]=="discovery" and r["collection"]=="gtrace_v2"]
    fig,axes=plt.subplots(1,3,sharey=True,figsize=(8.7,4.8))
    fig.subplots_adjust(left=.16,right=.96,top=.75,bottom=.12,wspace=.14)
    fig.suptitle("Fixed-work execution time on seven Google recordings",fontsize=15,weight="bold",y=.96)
    fig.text(.5,.86,"Percent of cycles saved versus Stride. Negative values mean slower execution.",
             ha="center",fontsize=10,color="#425761")
    for ax,folder,subtitle in zip(axes,PANELS,("A  ·  250k measured","B  ·  250k measured","C  ·  2M measured")):
        measures={(r["case"],r["policy"]):int(r["cycles"]) for r in read(folder+"/performance.csv")}
        values=np.array([[100*(1-measures[c,p]/measures[c,"stride"]) for p in ("multi","spp")] for c in cases])
        ax.imshow(values,aspect="auto",cmap="RdBu",norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12))
        for i,row in enumerate(values):
            for j,v in enumerate(row):
                ax.text(j,i,f"{v:+.2f}",ha="center",va="center",color="white" if abs(v)>7 else "#142c36",fontsize=10)
        ax.set_title(subtitle,fontsize=11,weight="bold")
        ax.set_xticks([0,1],["Multi-delta","SPP"])
        ax.set_yticks(range(len(cases)),[s[3:].capitalize() for s in cases])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.savefig(HERE/"google_execution.png",dpi=165)
    plt.close(fig)


def example():
    r=next(r for r in read("screen/producer_examples.csv") if r["case"]=="gg_delta" and r["target_instr"]=="100121")
    fig,ax=plt.subplots(figsize=(8.7,2.9))
    fig.subplots_adjust(left=.14,right=.95,bottom=.22,top=.78)
    fig.suptitle("Google Delta   ·   Two access streams predict physical line "+f'{int(r["physical_line"]):,}',
                 y=.97,fontsize=14,weight="bold")
    ax.axvline(0,lw=1,color="#697378",ls="--")
    for y,(name,color) in enumerate((("spp","#a84629"),("stride","#137665"))):
        t=int(r[name+"_ingress"])
        g=int(r[name+"_generation"])-t
        f=int(r[name+"_fill"])-t
        ax.plot([g,f],[y,y],color=color,lw=3)
        ax.scatter([g,f],[y,y],color=color,s=[20,46],zorder=3)
        ax.text(g,y-.24,f"proposal {g:+}",ha="center",fontsize=9)
        ax.text(f,y+.13,f"fill {f:+}",ha="center",color=color,fontsize=9)
    ax.set_ylim(-.59,1.58);ax.set_xlim(-365,250)
    ax.set_yticks([1,0],["Stride","SPP"])
    ax.set_xlabel("Cycles relative to demand ingress in each native execution")
    ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
    fig.savefig(HERE/"same_target.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    overview()
    example()
