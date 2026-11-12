"""Deterministic class-preserving nearest donor plans, no gold or outcomes."""
import math

if __package__:
    from .dream_label_plan_v1 import build_plan as random_plan, digest
else:
    from dream_label_plan_v1 import build_plan as random_plan, digest


ARMS = ('matched', 'margin_matched', 'posterior_matched')


def softmax(values):
    v=[math.exp(x-max(values)) for x in values]
    return [x/sum(v) for x in v]


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def pair_plan(ids, labels, values, *, scalar):
    result=[-1]*len(ids)
    for c in sorted(set(labels)):
        group=[i for i,x in enumerate(labels) if x==c]
        assert len(group)>=2
        if scalar:
            ordered=sorted(group,key=lambda i:(values[i][0],ids[i]))
            while len(ordered)>3:
                a,b=ordered[:2];result[a],result[b]=b,a;ordered=ordered[2:]
            remaining=ordered
        else:
            edges=sorted((distance(values[i],values[j]), min(ids[i],ids[j]), max(ids[i],ids[j]), i,j)
                         for k,i in enumerate(group) for j in group[k+1:])
            available=set(group)
            for _,_,_,i,j in edges:
                if len(available)<=3: break
                if i in available and j in available:
                    result[i],result[j]=j,i;available.remove(i);available.remove(j)
            remaining=sorted(available,key=lambda i:ids[i])
        for a,b in zip(remaining,remaining[1:]+remaining[:1]):result[a]=b
    assert sorted(result)==list(range(len(ids)))
    assert all(i!=j and labels[i]==labels[j] for i,j in enumerate(result))
    return result


def errors(values, donors):
    v=sorted(distance(values[i],values[j]) for i,j in enumerate(donors))
    return dict(mean=sum(v)/len(v), p95=v[int(.95*(len(v)-1))], maximum=v[-1])


def build_plan(ids, message_logits):
    base=random_plan(ids,message_logits)
    for rp in base['receivers']:
        scores=[r[rp['sender']] for r in message_logits]
        margins=[[sorted(x,reverse=True)[0]-sorted(x,reverse=True)[1]] for x in scores]
        posteriors=[softmax(x) for x in scores]
        old=rp['donors']
        rp['donors']={'matched':list(range(len(ids))),
            'margin_matched':pair_plan(ids,rp['sender_predicted_labels'],margins,scalar=True),
            'posterior_matched':pair_plan(ids,rp['sender_predicted_labels'],posteriors,scalar=False)}
        rp['matching_quality']={a:{'margin':errors(margins,d),'posterior':errors(posteriors,d)}
                               for a,d in {**old,**rp['donors']}.items() if a!='different_class'}
    base['version']='confidence_v1'
    return base


def resource_projection(baseline):
    assert baseline['completed'] and baseline['numerical_validity_passed'] is True
    assert baseline['examples']==1000
    estimate=1.5*(baseline['setup_seconds']+6*max(baseline['terminal_seconds']))+180
    assert estimate<=900 and baseline['torch_peak_reserved_bytes']<22*1024**3
    return estimate
