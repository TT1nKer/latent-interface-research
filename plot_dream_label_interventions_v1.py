"""Render aggregate, descriptive development results; no private question data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent
    old=json.loads((root/'results/dream_label_ablation_v1.analysis.json').read_text())
    new=json.loads((root/'results/dream_label_confidence_v1.analysis.json').read_text())
    data={(r['arm'],r['receiver']):r for report in (old,new) for r in report['by_arm_receiver']}
    arms=['within_class_0','within_class_1','label_prefix','margin_matched','posterior_matched']
    labels=['Same class: donor plan 0','Same class: donor plan 1','Class-only teacher prefix',
            'Same class + margin matching','Same class + eight-score matching']
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'dream-label-interventions-v1'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.1),sharey=True,gridspec_kw={'width_ratios':[1.1,1]})
    for receiver,color,offset in [(0,'#2468a0',-.12),(1,'#b34d26',.12)]:
        rows=[data[a,receiver] for a in arms]
        y=[i+offset for i in range(len(arms))]
        values=[r['accuracy_change']*100 for r in rows]
        low=[r['accuracy_change_descriptive_ci95'][0]*100 for r in rows]
        high=[r['accuracy_change_descriptive_ci95'][1]*100 for r in rows]
        axes[0].errorbar(values,y,xerr=[[x-l for x,l in zip(values,low)],[h-x for x,h in zip(values,high)]],
                         fmt='o',capsize=3,color=color,label=f'Receiver {receiver}',markersize=5)
        changed=[r['changed_label_rate']*100 for r in rows]
        axes[1].plot(changed,y,'o',color=color,markersize=5)
        for x,yy in zip(changed,y):axes[1].annotate(f'{x:.1f}%',(x,yy),xytext=(7,0),textcoords='offset points',va='center',color=color,fontsize=9)
    axes[0].set_yticks(range(len(arms)),labels);axes[0].invert_yaxis()
    axes[0].axvline(0,color='#777777',linewidth=.8)
    axes[0].set_xlabel('Accuracy change from original packet (percentage points)')
    axes[1].set_xlabel('Answers changed from original packet (%)')
    axes[1].set_xlim(0,max(data[a,r]['changed_label_rate']*100 for a in arms for r in (0,1))+2)
    axes[0].legend(loc='lower left',frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.18)
    fig.suptitle('Frozen QASC channel: class-preserving interventions on 1,000 development questions',fontsize=12)
    fig.text(.02,.02,'Intervals are descriptive question bootstraps conditional on fixed donor plans. No held-out confirmation.\nDifferent-class control changed 61.4% / 52.1% of answers for receivers 0 / 1 (omitted from the plotted scale).',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.11,1,.94))
    out=root/'results/figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'dream_label_interventions_v1.png',dpi=180)
    fig.savefig(out/'dream_label_interventions_v1.svg',metadata={'Date':None})
    svg=out/'dream_label_interventions_v1.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
