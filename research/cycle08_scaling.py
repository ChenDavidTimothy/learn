"""Cycle 08: intervention advantage as dimension grows.

No algorithms receive the hidden parity mask. Both know parity algebra.
Active: repeated canonical-basis membership queries.
Passive: classical noisy random-equation RANSAC + GF(2) solve.
Same oracle-label budget for both, random inputs versus targeted queries.
A comparison of these *specific algorithms*, not a complexity theorem.
"""
import json, math, random, statistics, time
ROOT=313333
DIMS=(32,64,96,128)
EPS=.1
R=11
REPS=18
RANSAC_ITER=1500
EVAL=1000

def oracle(mask,offset,x,rng):
    return (x&mask).bit_count()%2 ^ offset ^ (rng.random()<EPS)

def elimination(xs,ys,d):
    piv={}
    for x,y in zip(xs,ys):
        row=x|(1<<d); rhs=int(y)
        while row:
            j=row.bit_length()-1
            if j in piv:
                old,b=piv[j];row^=old;rhs^=b
            else:
                piv[j]=(row,rhs);break
    if len(piv)!=d+1:return None
    ans=0
    for j in sorted(piv):
        row,rhs=piv[j]
        if rhs^((row&ans).bit_count()%2):ans|=1<<j
    return ans&((1<<d)-1), (ans>>d)&1

def active(mask,offset,d,rng):
    obs=[]
    for i in [-1]+list(range(d)):
        q=0 if i<0 else 1<<i
        votes=sum(oracle(mask,offset,q,rng) for _ in range(R))
        obs.append(int(votes>R//2))
    bias=obs[0]
    support=sum((obs[i+1]^bias)<<i for i in range(d))
    return support,bias

def passive(mask,offset,d,rng):
    n=(d+1)*R
    xs=[rng.getrandbits(d) for _ in range(n)]
    ys=[oracle(mask,offset,x,rng) for x in xs]
    best=(None,None,n+1)
    start=time.perf_counter()
    for i in range(RANSAC_ITER):
        selected=rng.sample(range(n),d+1)
        solution=elimination([xs[j] for j in selected],
                             [ys[j] for j in selected],d)
        if solution is None:continue
        sm,si=solution
        mismatches=sum(((x&sm).bit_count()%2 ^ si)!=y
                       for x,y in zip(xs,ys))
        if mismatches<best[2]:
            best=(sm,si,mismatches)
            if mismatches <= EPS*n+3.5*math.sqrt(n*EPS*(1-EPS)):
                break
    return best,time.perf_counter()-start

def experiment(d,seed):
    rng=random.Random(seed)
    selected=rng.sample(range(d),max(4,d//3))
    truth=sum(1<<i for i in selected)
    bias=rng.randrange(2)
    t=time.perf_counter()
    a=active(truth,bias,d,random.Random(seed+100100))
    a_time=time.perf_counter()-t
    b,btime=passive(truth,bias,d,random.Random(seed+200200))
    # Compare exact recovered parameters; parity recovery = perfect noiseless
    # generalization, and an incorrect mask predicts 50% error on uniform x.
    exact_a=(a==(truth,bias))
    exact_b=(b[0]==truth and b[1]==bias)
    return {"active_exact":int(exact_a),"passive_exact":int(exact_b),
            "active_seconds":a_time,"passive_seconds":btime,
            "queries":(d+1)*R,"passive_best_training_errors":b[2]}

def main():
    output={"config":{"dims":DIMS,"noise":EPS,"repeat_each_basis":R,
                      "ransac_attempts":RANSAC_ITER,
                      "reps_per_dimension":REPS,
                      "equal_queries":True,
                      "latent_parity_algebra_given_both":True,
                      "methods_do_not_have_equal_computation":True},
            "by_dimension":{}}
    clock=time.perf_counter()
    for d in DIMS:
        rows=[experiment(d,ROOT+d*700+i*811) for i in range(REPS)]
        output["by_dimension"][str(d)]={
            k:{"mean":statistics.mean(r[k] for r in rows),
               "SE":statistics.stdev(r[k] for r in rows)/REPS**.5}
            for k in rows[0]}
        q=(1-EPS)**(d+1)
        rank=math.prod(1-2**(-i) for i in range(1,d+1))
        output["by_dimension"][str(d)]["idealized_success_per_ransac_draw"]=q*rank
        print("d",d,"active exact",
              sum(r["active_exact"] for r in rows),"/",REPS,
              "passive exact",sum(r["passive_exact"] for r in rows),
              "/",REPS,"mean passive seconds",
              round(output["by_dimension"][str(d)]["passive_seconds"]["mean"],3),
              "elapsed",round(time.perf_counter()-clock,1),flush=True)
    output["total_seconds"]=time.perf_counter()-clock
    path="/home/dtdav/dev/learn/research/cycle08_scaling.json"
    with open(path,"w") as f:json.dump(output,f,indent=2)
    print("saved",path,"total seconds",round(output["total_seconds"],2))
if __name__=="__main__":
    main()
