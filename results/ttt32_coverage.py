from arc.data import load_dir
from selector.arch_ingest import load_pickles, getter_kgmon, getter_full_probmul_3
tasks = load_dir("data_repo/data/evaluation")
def stats(d):
    g = load_pickles(d); cov=set(); top2=set(); keys=set()
    for (t,i),sols in g.items():
        if t not in tasks or i>=len(tasks[t]["test"]): continue
        gt=tasks[t]["test"][i].get("output"); keys.add((t,i))
        if any(s["grid"]==gt for s in sols.values()): cov.add((t,i))
        top=sorted(sols.values(),key=lambda x:-getter_kgmon(x["guesses"]))[:2]
        if any(s["grid"]==gt for s in top): top2.add((t,i))
    ncand=sum(len(s) for s in g.values())
    return g,keys,cov,top2,ncand
for d in ["cands_shard0","ttt32_shard0/cands"]:
    g,k,c,t2,n=stats(d); globals()[d.split('/')[0]]=(k,c,t2)
    print(d,"subkeys",len(g),"scorable",len(k),"unique cands",n,"coverage",len(c),"kgmon top2",len(t2), "tasks",len({x[0] for x in g}))
b=globals()["cands_shard0"]; t=globals()["ttt32_shard0"]
common=b[0]&t[0]
print("common subkeys",len(common))
print("cov base",len(b[1]&common),"cov ttt32",len(t[1]&common))
print("covered only by ttt32",sorted(t[1]-b[1]))
print("covered only by base",sorted(b[1]-t[1]))
print("top2 gained",sorted(t[2]-b[2]),"top2 lost",sorted(b[2]-t[2]))
print("union coverage",len(b[1]|t[1]))
