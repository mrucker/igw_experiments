import torch
import coba as cb

from learners import IGWLearner, LinUCBLearner, EpsLearner, CauchyRewardModel

#IGW vs Parametric in nonstationary environment

if __name__ == "__main__":

    n_processes = 12
    lr, tz = 0.01, 100

    if n_processes > 1: torch.set_num_threads(1)

    env = cb.Environments.cache_dir('.coba_cache').from_template('./class208.json',n_take=10_000).where(n_interactions=10_000)[:50].scale('min','minmax').cycle(6000).batch(8)

    lrn = [
        IGWLearner(CauchyRewardModel(2048, .08),"lambda t: 0 + 10*t**(3/4)",lr,tz),
        LinUCBLearner(features=['a','ax']),
    ]

    cb.Experiment(env,lrn).run('experiment2.log',processes=n_processes)
