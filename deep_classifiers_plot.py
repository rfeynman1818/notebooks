import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import patheffects as pe

fig=plt.figure(figsize=(12,4.5),dpi=200)
ax=plt.gca()
ax.set_axis_off()
ax.set_xlim(0,1); ax.set_ylim(0,1)

def box(x,y,w,h,txt,fs=11):
    r=Rectangle((x,y),w,h,fill=False,linewidth=2)
    ax.add_patch(r)
    t=ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,wrap=True)
    t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    return r

def arrow(x1,y1,x2,y2):
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=14,linewidth=2)
    ax.add_patch(a)

box(0.03,0.28,0.22,0.44,"FireRisk dataset\n(remote sensing imagery)\n(windowed stream)")
box(0.28,0.28,0.22,0.44,"Vision Transformer (ViT)\nfeature extractor\n[CLS] token")
box(0.56,0.28,0.18,0.44,"Embedding vectors\n(high-dimensional)")
box(0.78,0.60,0.20,0.30,"DRIFTLENS\nPCA + Gaussian\n(μ, Σ)")
box(0.78,0.16,0.20,0.30,"Frechét Distance\n(FDD)\nvs. baseline")
box(0.78,0.02,0.20,0.10,"Drift alert /\nmonitor")

arrow(0.25,0.50,0.28,0.50)
arrow(0.50,0.50,0.56,0.50)
arrow(0.74,0.50,0.78,0.68)
arrow(0.74,0.50,0.78,0.32)
arrow(0.88,0.16,0.88,0.12)
arrow(0.88,0.12,0.88,0.10)

ax.text(0.02,0.90,"DRIFTLENS on FireRisk (ViT): Monitoring drift in deep image classifiers",
        fontsize=13,weight='bold')
ax.text(0.56,0.20,"(window → embedding distribution)",fontsize=9,ha='center')

plt.savefig("driftlens_firerisk_vit_graphic.png",bbox_inches='tight',pad_inches=0.2)
plt.close()

