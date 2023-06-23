import math
import torch
import numpy as np
import coba as cb
from coba.encodings import InteractionsEncoder

class CauchyRewardModel(torch.nn.Module):
    def __init__(self, numrff, sigma):
        self.args = (numrff, sigma)
        super().__init__()

    @property
    def params(self):
        return {"argmax":"cauchy", "nrff": self.args[0], "sigma": self.args[1]}

    def define(self,context,actions):
        (numrff, sigma) = self.args
        n_features = self.featurize(context,actions).shape[1]
        self.rffW = torch.nn.Parameter(torch.empty(n_features, numrff).cauchy_(sigma=sigma), requires_grad=False)
        self.rffb = torch.nn.Parameter((2 * math.pi * torch.rand(numrff)), requires_grad=False)
        self.sqrtrff = torch.nn.Parameter(torch.Tensor([math.sqrt(numrff)]), requires_grad=False)
        self.linear = torch.nn.Linear(in_features=numrff, out_features=1)

    def featurize(self,context,actions):
        I = InteractionsEncoder(['a','xa'])
        X = [ I.encode(x=x,a=a) for x,A in zip(context,actions) for a in A]
        return torch.tensor(X,dtype=torch.float32)

    def forward(self,context,actions):
        X = self.featurize(context,actions)
        with torch.no_grad():
            rff = (torch.matmul(X, self.rffW) + self.rffb).cos() / self.sqrtrff
        return self.linear(rff)

class PolyRewardModel(torch.nn.Module):

    def define(self,context,actions):
        in_features = self.featurize(context,actions).shape[1]
        self.layers = torch.nn.Linear(in_features=in_features, out_features=1)

    @property
    def params(self):
        return {"argmax":"Poly"}

    def featurize(self,context,actions):
        I = InteractionsEncoder(['a','xa'])
        X = [ I.encode(x=x,a=a) for x,A in zip(context,actions) for a in A]
        return torch.tensor(X,dtype=torch.float32)

    def forward(self,context,actions):
        X = self.featurize(context,actions)
        return self.layers(X)

class MlpRewardModel(torch.nn.Module):

    def __init__(self, n_layers:int):
        self._n_layers = n_layers
        self._defined = False
        super().__init__()

    def define(self, context, actions):
        in_features = self.featurize(context,actions).shape[1]
        layers = []
        for _ in range(self._n_layers):
            layers.append(torch.nn.Linear(in_features=in_features, out_features=in_features))
            layers.append(torch.nn.LeakyReLU())
        self.layers = torch.nn.Sequential(*layers, torch.nn.Linear(in_features=in_features, out_features=1) )

    @property
    def params(self):
        return {"argmax":"MLP", "n_layers":self._n_layers}

    def featurize(self,context,actions):
        I = cb.InteractionsEncoder(['a','xa'])
        X = [ I.encode(x=x,a=a) for x,A in zip(context,actions) for a in A]
        return torch.tensor(X,dtype=torch.float32)
    
    def forward(self,context,actions):
        X = self.featurize(context,actions)
        #X = torch.concat((context,self.embeddings(actions.argmax(dim=1))),dim=1)
        return self.layers(X)

class IGWLearner:
    def __init__(self, model, gamma_str: str, initlr:float, tzero:float, importance=True, n_repeats=1, v=1):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.t          = 0
        self.model      = model
        self.gamma_str  = gamma_str
        self.initlr     = initlr
        self.tzero      = tzero
        self.opt        = None
        self.scheduler  = None
        self.importance = importance
        self.n_repeats  = n_repeats
        self.loss       = torch.nn.MSELoss(reduction='none')
        self.v          = v

    @property
    def params(self):
        return {**self.model.params, "lr":self.initlr, "tz":self.tzero, "importance":self.importance, "n_repeats": self.n_repeats, "gs": self.gamma_str, 'v':self.v }

    def define(self,context,actions):
        torch.manual_seed(1)
        self.gamma = eval(self.gamma_str)
        self.model.define(context,actions)
        self.opt = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:math.sqrt(self.tzero)/math.sqrt(self.tzero+t))

    def predict(self, context, actions):
        n_batch = len(context)

        with torch.no_grad():

            if not self.opt: self.define(context,actions)

            mu = len(actions[0])
            gamma = self.gamma(self.t)

            rewards = torch.reshape(self.model(context,actions), (n_batch,-1))
            rwdmax  = rewards.max(axis=1,keepdim=True)[0]
            rwdgap  = rwdmax-rewards
            probs   = 1/(mu+gamma*rwdgap)

            bests = rwdgap == 0
            br,bc = bests.nonzero(as_tuple=True)
            n_max = (bests).sum(axis=1)
            p_max = (1-(probs*(rwdgap!=0)).sum(axis=1))/n_max
            probs[br,bc] = p_max[br]

        return probs.tolist()

    def learn(self, context, actions, action, reward, probs):
        self.t += 1

        if not self.opt: self.define(context,actions)
        reward = torch.tensor(reward,dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor([1 if not self.importance else min(1./p,5) if p else 5 for p in probs],dtype=torch.float32).unsqueeze(1)

        for _ in range(self.n_repeats):

            self.opt.zero_grad()
            pred_reward = self.model(context, [[a] for a in action])
            loss_mean   = (weights*self.loss(pred_reward, reward)).mean()
            loss_mean.backward()
            self.opt.step()

        if self.v == 1:
            self.scheduler.step()

class EpsLearner:
    def __init__(self, model, epsilon:float, initlr:float, tzero:float, importance=True, n_repeats=1, v=1):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.t          = 0
        self.model      = model
        self.epsilon    = epsilon
        self.initlr     = initlr
        self.tzero      = tzero
        self.opt        = None
        self.scheduler  = None
        self.importance = importance
        self.n_repeats  = n_repeats
        self.loss       = torch.nn.MSELoss(reduction='none')
        self.v          = v

    @property
    def params(self):
        return {**self.model.params, "lr":self.initlr, "tz":self.tzero, "importance":self.importance, "n_repeats": self.n_repeats, "e": self.epsilon, 'v':self.v }

    def define(self,context,actions):
        torch.manual_seed(1)
        self.model.define(context,actions)
        self.opt = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:math.sqrt(self.tzero)/math.sqrt(self.tzero+t))

    def predict(self, context, actions):
        n_batch   = len(context)
        n_actions = len(actions[0])

        with torch.no_grad():

            if not self.opt: self.define(context,actions)

            rewards = torch.reshape(self.model(context,actions), (n_batch,-1))
            rwdmax  = rewards.max(axis=1,keepdim=True)[0]
            rwdgap  = rwdmax-rewards
            probs   = torch.tensor([[self.epsilon/n_actions]*n_actions]*n_batch)

            bests = rwdgap == 0
            br,bc = bests.nonzero(as_tuple=True)
            n_max = (bests).sum(axis=1)
            probs[br,bc] += (1-self.epsilon)/n_max

        return probs.tolist()

    def learn(self, context, actions, action, reward, probs):
        self.t += 1

        if not self.opt: self.define(context,actions)
        reward = torch.tensor(reward,dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor([1 if not self.importance else min(1./p,5) if p else 5 for p in probs],dtype=torch.float32).unsqueeze(1)

        for _ in range(self.n_repeats):

            self.opt.zero_grad()
            pred_reward = self.model(context, [[a] for a in action])
            loss_mean   = (weights*self.loss(pred_reward, reward)).mean()
            loss_mean.backward()
            self.opt.step()

        if self.v == 1:
            self.scheduler.step()

class LinUCBLearner:
    """A contextual bandit learner that represents expected reward as a
    linear function of context and action features. Exploration is carried
    out according to upper confidence bound estimates.

    This is an implementation of the Chu et al. (2011) LinUCB algorithm using the
    `Sherman-Morrison formula`__ to iteratively calculate the inversion matrix. This
    implementation's computational complexity is linear with respect to feature count.

    Remarks:
        The Sherman-Morrsion implementation used below is given in long form `here`__.

    References:
        Chu, Wei, Lihong Li, Lev Reyzin, and Robert Schapire. "Contextual bandits
        with linear payoff functions." In Proceedings of the Fourteenth International
        Conference on Artificial Intelligence and Statistics, pp. 208-214. JMLR Workshop
        and Conference Proceedings, 2011.

    __ https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    __ https://research.navigating-the-edge.net/assets/publications/linucb_alternate_formulation.pdf
    """

    def __init__(self, alpha: float = 1, features  = [1, 'a', 'ax']) -> None:
        """Instantiate a LinUCBLearner.

        Args:
            alpha: This parameter controls the exploration rate of the algorithm. A value of 0 will cause actions
                to be selected based on the current best point estimate (i.e., no exploration) while a value of inf
                means that actions will be selected based solely on the bounds of the action point estimates (i.e.,
                we will always take actions that have the largest bound on their point estimate).
            features: Feature set interactions to use when calculating action value estimates. Context features
                are indicated by x's while action features are indicated by a's. For example, xaa means to cross the
                features between context and actions and actions.
        """

        self._alpha = alpha

        self._X = features
        self._X_encoder = InteractionsEncoder(features)

        self._theta = None
        self._A_inv = None

    @property
    def params(self):
        return {'family': 'LinUCB', 'alpha': self._alpha, 'features': self._X}

    def predict(self, context, actions):

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))

        context = context or []
        features = np.array([self._X_encoder.encode(x=context,a=action) for action in actions]).T

        if(self._A_inv is None):
            self._theta = np.zeros(features.shape[0])
            self._A_inv = np.identity(features.shape[0])

        point_estimate = self._theta @ features
        point_bounds   = np.diagonal(features.T @ self._A_inv @ features)

        action_values = point_estimate + self._alpha*np.sqrt(point_bounds)
        max_indexes   = np.where(action_values == np.amax(action_values))[0]

        return [int(ind in max_indexes)/len(max_indexes) for ind in range(len(actions))]

    def learn(self, context, actions, action, reward: float, probability: float) -> None:

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))

        context = context or []
        features = np.array(self._X_encoder.encode(x=context,a=action)).T

        if(self._A_inv is None):
            self._theta = np.zeros((features.shape[0]))
            self._A_inv = np.identity(features.shape[0])

        r = self._theta @ features
        w = self._A_inv @ features
        v = features    @ w

        self._A_inv = self._A_inv - np.outer(w,w)/(1+v)
        self._theta = self._theta + (reward-r)/(1+v) * w

torch.set_num_threads(1)

if __name__ == "__main__":

    n_processes = 12
    lr, tz = 0.02, 100

    env = cb.Environments.cache_dir('.coba_cache').from_template('./208_multiclass.json',n_take=10_000).where(n_interactions=10_000)
    env = env[:50].scale('min','minmax') #filter down to a smaller batch of datasets so it completes sooner
    env = (env.cycle(0) + env.cycle(6000)).batch(8) #mini batches of size 8

    lrn = [
        IGWLearner(CauchyRewardModel(2048, .08),"lambda t: 0 + 10*t**(3/4)",lr/2,tz,False,2,2),
        LinUCBLearner(features=['a','ax']),
        EpsLearner(CauchyRewardModel(2048, .08),.05,lr/2,tz,False,2,2)
    ]

    cb.Experiment(env,lrn).run('out.log',processes=n_processes)
