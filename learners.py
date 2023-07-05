import math
import torch
import numpy as np
import coba as cb
from coba.encodings import InteractionsEncoder

import scipy.optimize as so

class CappedIGWLearner:
    def __init__(self, fhat, gamma_str:str, initlr:float, tzero:float, kappa_infty:float, eta:float, tau_min:float, tau_max:float, nalgos:int):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.gamma_str = gamma_str
        self.sampler   = CappedIGWSampler(gamma_str,kappa_infty,eta,tau_min,tau_max,nalgos)
        self.fhat      = fhat
        self.initlr    = initlr
        self.tzero     = tzero
        self.opt       = None
        self.scheduler = None
        self.losses    = []

    @property
    def params(self):
        samp_params = self.sampler.params if self.sampler else {}
        return {**samp_params, **self.fhat.params, "g": self.gamma_str, "lr":self.initlr, "tz":self.tzero}

    def initialize(self,context,actions):
        torch.manual_seed(1)
        self.fhat.define(context,actions)
        self.opt = torch.optim.Adam((p for p in self.fhat.parameters() if p.requires_grad ), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:np.sqrt(self.tzero)/np.sqrt(self.tzero+t))

    def predict(self, context, actions):
        if not self.opt: self.initialize(context,actions)

        #WILL BE REPLACED TO SELECT FROM THE POWERSET OF EXAMPLES
        generators = [ lambda _A=A: _A[np.random.randint(len(_A))] for A in actions ]

        with torch.no_grad():
            _actions, _density, _algo, _invpalgo = self.sampler.sample(self.fhat, context, generators)
            return _actions, _density, {'algo':_algo, 'invpalgo':_invpalgo}

    def learn(self, context, actions, action, reward, probs, algo, invpalgo):
        reward  = torch.tensor(reward,dtype=torch.float32).unsqueeze(1)

        if not self.opt: self.initialize(context,actions)

        with torch.no_grad(): self.sampler.update(algo, invpalgo, reward)

        self.opt.zero_grad()
        pred = self.fhat(context, [action])
        loss = self.loss(pred, (1-reward))
        loss.mean().backward()
        self.opt.step()
        self.scheduler.step()

class IGWLearner:
    def __init__(self, fhat, gamma_str: str, initlr:float, tzero:float):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.t          = 0
        self.fhat      = fhat
        self.gamma_str  = gamma_str
        self.initlr     = initlr
        self.tzero      = tzero
        self.opt        = None
        self.scheduler  = None
        self.loss       = torch.nn.MSELoss(reduction='none')

    @property
    def params(self):
        return {**self.fhat.params, "g": self.gamma_str, "lr":self.initlr, "tz":self.tzero}

    def define(self,context,actions):
        torch.manual_seed(1)
        self.gamma = eval(self.gamma_str)
        self.fhat.define(context,actions)
        self.opt = torch.optim.Adam((p for p in self.fhat.parameters() if p.requires_grad), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:np.sqrt(self.tzero)/np.sqrt(self.tzero+t))

    def predict(self, context, actions):
        n_batch = len(context)

        with torch.no_grad():

            if not self.opt: self.define(context,actions)

            mu = len(actions[0])
            gamma = self.gamma(self.t)

            rewards = torch.reshape(self.fhat(context,actions), (n_batch,-1))
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

        self.opt.zero_grad()
        pred_reward = self.fhat(context, [[a] for a in action])
        loss_mean   = self.loss(pred_reward, reward).mean()
        loss_mean.backward()
        self.opt.step()
        self.scheduler.step()

class EpsLearner:
    def __init__(self, fhat, epsilon:float, initlr:float, tzero:float):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.t         = 0
        self.fhat      = fhat
        self.epsilon   = epsilon
        self.initlr    = initlr
        self.tzero     = tzero
        self.opt       = None
        self.scheduler = None
        self.loss      = torch.nn.MSELoss(reduction='none')

    @property
    def params(self):
        return {**self.fhat.params, "lr":self.initlr, "tz":self.tzero, "e": self.epsilon}

    def define(self,context,actions):
        torch.manual_seed(1)
        self.fhat.define(context,actions)
        self.opt = torch.optim.Adam((p for p in self.fhat.parameters() if p.requires_grad), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:np.sqrt(self.tzero)/np.sqrt(self.tzero+t))

    def predict(self, context, actions):
        n_batch   = len(context)
        n_actions = len(actions[0])

        with torch.no_grad():

            if not self.opt: self.define(context,actions)

            rewards = torch.reshape(self.fhat(context,actions), (n_batch,-1))
            rwdmax  = rewards.max(axis=1,keepdim=True)[0]
            rwdgap  = rwdmax-rewards
            probs   = torch.tensor([[self.epsilon/n_actions]*n_actions]*n_batch)

            bests = (rwdgap == 0).nonzero()
            for row in range(n_batch):
                best_cols = bests[bests[:,0]==row,1]
                probs[row,best_cols] += (1-self.epsilon)/len(best_cols)

        return probs.tolist()

    def learn(self, context, actions, action, reward, probs):
        self.t += 1

        if not self.opt: self.define(context,actions)
        reward = torch.tensor(reward,dtype=torch.float32).unsqueeze(1)

        self.opt.zero_grad()
        pred_reward = self.fhat(context, [[a] for a in action])
        loss_mean   = self.loss(pred_reward, reward).mean()
        loss_mean.backward()
        self.opt.step()
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
        self.rffb = torch.nn.Parameter((2 * np.pi * torch.rand(numrff)), requires_grad=False)
        self.sqrtrff = torch.nn.Parameter(torch.Tensor([np.sqrt(numrff)]), requires_grad=False)
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

class CappedIGWSampler:

    def __init__(self, gamma_str, kappa_infty, eta, tau_min, tau_max, nalgos):

        self.lamb        = 1
        self.args        = (eta,tau_min,tau_max,nalgos,kappa_infty)
        self.eta         = eta / nalgos
        self.gamma_str   = gamma_str
        self.gamma       = None
        self.taus        = torch.Tensor(np.geomspace(tau_min, tau_max, nalgos))
        self.invpalgo    = torch.Tensor([self.taus.shape[0]] * self.taus.shape[0])
        self.T           = 0
        self.kappa_infty = kappa_infty

    @property
    def params(self):
        return {"sampler":"new", **dict(zip(['eta','tau_min','tau_max','n_taus','k_inf'],self.args))}

    def update(self, algo, invprop, reward):

        reward = reward.round(decimals=8)
        assert torch.all(reward >= 0) and torch.all(reward <= 1), reward

        weightedlosses = self.eta * (-reward.squeeze(1)) * invprop.squeeze(1)
        newinvpalgo = torch.scatter_reduce(input=self.invpalgo, dim=0, index=algo, src=weightedlosses, reduce='sum')

        # just do this calc on the cpu
        invp = newinvpalgo.cpu().numpy()
        invp += 1 - np.min(invp)
        Zlb = 0
        Zub = 1

        while (np.sum(1 / (invp + Zub)) > 1):
            Zlb = Zub
            Zub *= 2 

        root, res = so.brentq(lambda z: 1 - np.sum(1 / (invp + z)), Zlb, Zub, full_output=True)
        assert res.converged, res

        self.invpalgo = torch.Tensor(invp + root, device=self.invpalgo.device)

    def sample(self, fhat, X, G):

        if self.gamma is None: self.gamma = eval(self.gamma_str)

        if 'samples_martin' not in cb.CobaContext.learning_info:
            cb.CobaContext.learning_info['samples_martin'] = []
        if 'samples_reject' not in cb.CobaContext.learning_info:
            cb.CobaContext.learning_info['samples_reject'] = []

        #This is a "batch" of data.. So we need to pick a base for each of the N items in the batch
        N = len(X)

        algos    = torch.distributions.categorical.Categorical(probs=1.0/self.invpalgo, validate_args=False).sample((N,))
        invpalgo = torch.gather(input=self.invpalgo.unsqueeze(0).expand(N, -1), dim=1, index=algos.unsqueeze(1))
        tau      = torch.gather(input=self.taus.unsqueeze(0).expand(N, -1), dim=1, index=algos.unsqueeze(1))

        selected_actions  = []
        selected_density  = []

        self.T += 1 

        gamma = self.gamma(self.T)

        for t,x,generator in zip(tau,X,G):
            t = t.item()
            beta = self.find_beta_martingale(fhat,gamma,x,t,generator)
            m_t = lambda a: 1/(self.lamb+gamma*torch.clamp(fhat([x],[a])-beta,min=0))

            def batched_rejection_sampler():
                while True:
                    actions   = [generator() for _ in range(100)]
                    densities = m_t(actions).squeeze()
                    keep      = torch.tensor(np.random.rand(100)) < densities

                    yield from zip(keep,actions,densities)

            for n, (keep,action,density) in enumerate(batched_rejection_sampler()):
                if keep: break

            selected_actions.append(action)
            selected_density.append(density)

            cb.CobaContext.learning_info['samples_reject'].append(n)

        return selected_actions, selected_density, algos, invpalgo

    def find_beta_martingale(self,fhat,gamma,x,tau,generator):

        clip_min = np.core.umath.maximum

        alpha = .05
        g     = lambda f,beta: tau/clip_min(f-beta,self.lamb)
        cs    = BettingNormCS(g=g, tau=tau, gamma=gamma, alpha=alpha, lb=1/self.kappa_infty)

        N  = 5_000 #10_000 # we shouldn't ever hit this but just in case...
        fs = []

        def batched_base_action_sampler():
            while True:
                yield from fhat([x],[[generator() for _ in range(100)]]).squeeze().tolist()

        for n,f in zip(range(N),batched_base_action_sampler()):

            fs.append(f)
            cs.addobs(self.lamb+gamma*f)

            if n % 10 == 0: #Improves performance... I'm not sure if it hurts convergence...
                cs.updatelowercs()
                cs.updateuppercs()
                l, u = cs.getci()
                if l > u: break

        #print(t)
        cb.CobaContext.learning_info['samples_martin'].append(n)
        return min(u,l) + torch.rand(size=(1,)).item()*abs(u-l)

class BettingNormCS:
    def __init__(self, *, g, tau, gamma, alpha, lb):

        #g     -- a function which accepts a beta and an action
        #tau   -- the same tau from our paper
        #gamma -- the same gamma from our paper
        #alpha -- the level of significance we desire

        #note: This is based on https://arxiv.org/abs/2010.09686
        #   In the original paper the goal is to estimate the mean from random variables
        #   in our work we want to design a random variables that has a desired mean. That
        #   means we fix our mean and then update our random variable until the confidence
        #   interval for our random variable equalling our desired mean is small

        assert tau > 1, tau
        assert gamma > 0, gamma
        assert 0 < alpha < 1, alpha

        self.g = g
        self.tau = tau
        self.betamin = gamma*(1 - tau) / gamma
        self.betamax = gamma*1
        self.t = 0
        self.alist = np.array([])
        self.lamlist = np.array([0])
        self.nulist = np.array([0])
        self.lamgrad = 0
        self.nugrad = 0
        self.betaminus = self.betamin
        self.betaplus = self.betamax
        self.alpha = alpha
        self.thres = -np.log(alpha/2)
        self.gamma = gamma

        self.lb = lb

    def betlowercs(self):

        g = self.g(self.alist[-1], self.betaminus)
        nabla = (1 - g) / (1 + self.nulist[-1] * (1 - g))
        self.nugrad += nabla**2
        G = self.tau - 1
        D = 1/(2 * (self.tau - 1))
        gamma = 1 / (4 * G * D)
        epsilon = 1 / (gamma**2 * D**2)

        ytp1 = self.nulist[-1] + nabla / (gamma * (epsilon + self.nugrad))
        xtp1 = max(0, min(1/(2 * (self.tau - 1)), ytp1))
        self.nulist = np.append(self.nulist,xtp1)

    def betuppercs(self):
        g = self.g(self.alist[-1], self.betaplus)
        nabla = (g - self.lb) / (1 + self.lamlist[-1] * (g - self.lb))
        self.lamgrad += nabla**2

        G = self.tau - self.lb
        D = 1
        gamma = 1 / (4 * G * D)
        epsilon = 1 / (gamma**2 * D**2)

        ytp1 = self.lamlist[-1] + nabla / (gamma * (epsilon + self.lamgrad))
        xtp1 = max(0, min(1, ytp1))
        self.lamlist = np.append(self.lamlist,xtp1)

    def addobs(self, a):
        self.alist = np.append(self.alist,a)
        self.t += 1

        self.betlowercs()
        self.betuppercs()

    def uppercswealth(self, beta):
        return np.log(1+(self.g(self.alist,beta)-self.lb)*self.lamlist[:-1]).sum()

    def updateuppercs(self):
        if self.betaplus <= self.betamin:
            return

        maxbeta = self.betaplus
        maxbetawealth = self.uppercswealth(maxbeta)
        if maxbetawealth < self.thres:
            return

        minbeta = self.betamin
        minbetawealth = self.uppercswealth(minbeta)
        if minbetawealth > self.thres:
            self.betaplus = self.betamin
            return

        res = so.root_scalar(f = lambda beta: self.uppercswealth(beta) - self.thres, method = 'brentq', bracket = [minbeta, maxbeta])
        assert res.converged, res
        self.betaplus = res.root

    def lowercswealth(self, beta):
        return np.log(1+(1-self.g(self.alist,beta))*self.nulist[:-1]).sum()

    def updatelowercs(self):
        if self.betaminus >= self.betamax:
            return

        minbeta = self.betaminus
        minbetawealth = self.lowercswealth(minbeta)
        if minbetawealth < self.thres:
            return

        maxbeta = self.betamax
        maxbetawealth = self.lowercswealth(maxbeta)
        if maxbetawealth > self.thres:
            self.betaminus = self.betamax
            return

        res = so.root_scalar(f = lambda beta: self.lowercswealth(beta)-self.thres, method = 'brentq', bracket = [ minbeta, maxbeta ])
        assert res.converged, res
        self.betaminus = res.root

    def getci(self):
        return self.betaminus/self.gamma, self.betaplus/self.gamma
