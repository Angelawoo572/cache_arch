"""Draw two neighboring requests from the measured MCF example using a shared cycle axis."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parents[1] / "results"
ids = ("2293","2294")
events = {i: {} for i in ids}
with (results/"examples.csv").open() as f:
    for row in csv.DictReader(f):
        num = row["candidate_id"]
        if num in events and row["case"]=="mcf-cactus-r1" and row["event"] in ("accepted","dispatch","fill","demand_anchor"):
            events[num][row["event"]] = int(row["cycle"])
zero = events[ids[0]]["accepted"]
plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","axes.spines.top":False,
                     "axes.spines.left":False,"axes.spines.right":False,"font.size":10})
fig, ax = plt.subplots(figsize=(9,3.55))
fig.subplots_adjust(top=.77, bottom=.25, left=.16,right=.94)
fig.text(.04,.925,"Two neighboring lines, two different outcomes",fontsize=17,weight="bold",ha="left")
fig.text(.04,.845,"MCF with CactuBSSN  ·  First trace point  ·  Shared LLC",fontsize=10,color="#475467")
for y,(num,role) in enumerate(zip(ids,("Late merge","Timely hit"))):
    v = {k:t-zero for k,t in events[num].items()}
    if len(v)!=4:
        raise ValueError("incomplete candidate timeline "+num)
    lane = 1 - y
    ax.hlines(lane, v["accepted"], max(v.values()),color="#b5c3cc",linewidth=2)
    for stage, marker, color in [("accepted","o","#707b85"),("dispatch","|","#707b85"),("fill","s","#267b76"),("demand_anchor","D","#aa3525")]:
        ax.scatter(v[stage], lane, marker=marker, s=65, color=color, zorder=3)
    prefix= "demand" if y==0 else "fill"
    suffix= "fill" if y==0 else "demand"
    first, second=sorted([v["fill"],v["demand_anchor"]])
    ax.annotate(prefix+" "+str(first),(first,lane),xytext=(2,12),textcoords="offset points",ha="center",color="#364152")
    ax.annotate(suffix+" "+str(second),(second,lane),xytext=(2,12),textcoords="offset points",ha="center",color="#364152")
    ax.text((first+second)/2,lane-.19,f"{second-first} cycles",ha="center",color="#344054",fontsize=10)
ax.set_yticks((1,0),("Late merge","Timely hit"),weight="bold")
ax.set_ylim(-.45,1.53)
ax.tick_params(left=False,bottom=False,labelcolor="#344054")
ax.spines["bottom"].set_color("#a3b0bb")
ax.set_xlim(-13,403)
ax.set_xlabel("Cycles after the first prefetch was accepted",labelpad=8)
fig.text(.04,.04,"Demand = completed LLC lookup. Both addresses were used. Only the second filled before demand.",fontsize=9,color="#475467")
fig.savefig(results/"adjacent_requests.svg",metadata={"Date":None})
