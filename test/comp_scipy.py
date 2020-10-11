import time
import numpy as np
import scipy.stats as stats

SIZE = 10000
tot_time = {}
tot_mean = {}
tot_var = {}

def benchmark(name, gen):
    begin = time.perf_counter()
    vs = gen.rvs(SIZE)
    elapsed = time.perf_counter() - begin
    if name not in tot_time: 
        tot_time[name] = []
        tot_mean[name] = []
        tot_var[name] = []
    tot_time[name].append(elapsed)
    tot_mean[name].append(np.average(vs, axis=0))
    if len(vs.shape) == 2:
        zc = vs - np.average(vs, axis=0)[np.newaxis, ...]
        tot_var[name].append((zc.T @ zc) / vs.shape[0])
    else:
        tot_var[name].append(np.std(vs, axis=0) ** 2)

for _ in range(20):
    benchmark('MvNormal/4', stats.multivariate_normal(
        mean=[0, 1, 2, 3], 
        cov=[[1, 1, 0, 0],
            [1, 2, 0, 0],
            [0, 0, 3, 1],
            [0, 0, 1, 2]]
    ))

    r = np.random.normal(size=[100, 100]) * 0.1
    benchmark('MvNormal/100', stats.multivariate_normal(
        mean=np.linspace(0, 99, 100), 
        cov=np.identity(100) * 4 + (r @ r.T)
    ))

    p = np.linspace(1, 4, 4)
    p /= p.sum()
    benchmark('Multinomial/4 t=20', stats.multinomial(20, p))
    benchmark('Multinomial/4 t=1000', stats.multinomial(1000, p))

    p = np.linspace(1, 100, 100)
    p /= p.sum()
    benchmark('Multinomial/100 t=20', stats.multinomial(20, p))
    benchmark('Multinomial/100 t=1000', stats.multinomial(1000, p))

    benchmark('Dirichlet/4', stats.dirichlet(np.linspace(0.1, 0.4, 4)))
    benchmark('Dirichlet/100', stats.dirichlet(np.linspace(0.1, 10, 100)))

    benchmark('Wishart/4', stats.wishart(
        df=4,
        scale=[[2, 1, 0, 0],
            [1, 2, 1, 0],
            [0, 1, 2, 1],
            [0, 0, 1, 2]]
    ))

    r = np.random.random(size=[50, 50]) * 0.1
    benchmark('Wishart/50', stats.wishart(
        df=50,
        scale=np.identity(50) + (r @ r.T)
    ))

    benchmark('InvWishart/4', stats.invwishart(
        df=8,
        scale=[[2, 1, 0, 0],
            [1, 2, 1, 0],
            [0, 1, 2, 1],
            [0, 0, 1, 2]]
    ))

    r = np.random.random(size=[50, 50]) * 0.1
    benchmark('InvWishart/50', stats.invwishart(
        df=54,
        scale=np.identity(50) + (r @ r.T)
    ))

for name, ts in tot_time.items():
    print("{:24} {:7.2f} ({:7.2f})".format(name, np.average(ts) * 1000, np.std(ts) * 1000))

def format_numpy(arr, prefix=''):
    arr = np.array(arr)
    return ('\n' + prefix).join(str(arr).split('\n'))

np.set_printoptions(formatter={'float':lambda x:"{:7.3f}".format(x)})

for name in tot_time:
    if tot_mean[name][-1].shape[0] >= 10: continue
    print("{}\nMean: {}\nVar : {}\n".format(name, 
        format_numpy(tot_mean[name][-1], ' ' * 6), 
        format_numpy(tot_var[name][-1], ' ' * 6)
    ))
