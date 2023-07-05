import torch
import coba as cb

from learners import CappedIGWLearner, EpsLearner, CauchyRewardModel

class DiscreteRewards:

    class Reward:

        def __init__(self, label) -> None:
            self._label = label

        def eval(self, action):
            return 1 + (action-self._label if self._label > action else self._label-action)

    def __init__(self, n_actions:int) -> None:
        self._n_actions = n_actions

    @property
    def params(self):
        return {"n_actions": self._n_actions}

    def filter(self, interactions):

        interactions = list(interactions)

        all_labels = [i['rewards']._label for i in interactions]

        l_min = min(all_labels)
        l_max = max(all_labels)
        l_rng = l_max-l_min

        actions = [ round(n/(self._n_actions-1),2) for n in range(self._n_actions)]

        for old in interactions:
            new = old.copy()
            new['rewards'] =DiscreteRewards.Reward((new['rewards']._label - l_min)/l_rng)
            new['actions'] = actions
            yield new

if __name__ == "__main__":

    n_processes = 12
    lr, tz = 0.01, 100
    eta, tmin, tmax, nalgos = .3, 2, 100, 12

    if n_processes > 1: torch.set_num_threads(1)

    env = cb.Environments.cache_dir('.coba_cache').from_openml(43939,take=2000).scale('min','minmax').filter([DiscreteRewards(n) for n in [10, 100, 1000, 2000, 8000]]).shuffle(n=2).batch(8)

    lrn = [
        CappedIGWLearner(CauchyRewardModel(512,.03),"lambda t: 20*t**(3/4)",lr,tz,24,eta,tmin,tmax,nalgos),
        EpsLearner(CauchyRewardModel(512,.03),.05,lr,tz)
    ]

    cb.Experiment(env,lrn,evaluation_task=cb.OnPolicyEvaluation(['reward','time'])).run('experiment3.log',processes=n_processes)
