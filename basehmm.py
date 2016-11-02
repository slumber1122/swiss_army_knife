import numpy


class BaseHMM:
    def __init__(self, m, n, precision=numpy.double):
        '''
        m is the num of state in HMM
        n is the num of observed state in HMM
        '''
        self.m = m
        self.n = n
        self.precision = precision
        self.A = None
        self.B = None
        self._eta = self._eta1

    def _eta1(self, t, T):
        return 1.0

    def save():
        pass

    def train(self, observations, iterations=1, epsilon=0.0001):
        '''
        update the HMM parameters given a new set of observations.
        observations: can either be a single array of observed symbols

        Training is repeated 'iterations' times, or until log likelihook of the
        model increases by less than 'epsilon'
        '''
        self._mapB(observations)

        for i in range(iterations):
            prob_old, prob_new = self.trainiter(observations)
            if abs(prob_new - prob_old) < epsilon:
                break

    def trainiter(self, observations):
        '''
        EM algorithm: return the log likelihood of the old model and the new.
        '''
        prob_old = self.log_likelihood(observations)
        new_model = self._baumwelch(observations)
        self._updatemodel(new_model)
        prob_new = self.log_likelihood(observations)

        return prob_old, prob_new

    def _baumwelch(self, observations):
        stats = self._calcstats(observations)
        return self._reestimate(stats, observations)

    def _reestimate(self, stats, observations):
        '''
        performs the `M` step of the Baum-Welch algorithm.
        '''
        new_model = {}
        new_model['pi'] = stats['gamma'][0]
        new_model['A'] = self._reestimateA(observations, stats['xi'], stats['gamma'])
        return new_model

    def _reestimateA(self, observations, xi, gamma):
        A_new = numpy.zeros((self.n, self.n), dtype=self.precision)
        for i in range(self.n):
            for j in range(self.n):
                numer = 0.0
                denom = 0.0
                for t in range(len(observations)-1):
                    numer += (self._eta(t, len(observations)-1)*xi[t][i][j])
                    denom += (self._eta(t, len(observations)-1)*gamma[t][i])
                A_new[i][j] = numer/denom
        return A_new


    def _calcstats(self, observations):
        '''
        calc required statistics of the current model, as part of
        the Baum-Welch `E` step.
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        '''
        stats = {}
        stats['alpha'] = self._calcalpha(observations)
        stats['beta'] = self._calcbeta(observations)
        stats['xi'] = self._calcxi(observations, stats['alpha'], stats['beta'])
        stats['gamma'] = self._calcgamma(stats['xi'], len(observations))

        return stats

    def _calcalpha(self, observations):
        '''
        calc 'alpha' the forward variable.
        The alpha variable is a numpy array indexed by time, then state (TxN).
        alpha[t][i] = the probability of being in state 'i' after observing the
        first t symbols.
        '''
        alpha = numpy.zeros((len(observations), self.n), dtype=self.precision)

        for x in range(self.n):
            alpha[0][x] = self.pi[x]*self.B_map[x][0]

        for t in range(1, len(observations)):
            for j in range(self.n):
                for i in range(self.n):
                    alpha[t][j] += alpha[t-1][i]*self.A[i][j]
                alpha[t][j] *= self.B_map[j][t]
        return alpha

    def _calcbeta(self, observations):
        '''
        calc 'beta' the backword variable.
        beta[t][i] = the probability of bing in state 'i' and then observing
        the symbols from t+1 to the end.
        '''
        beta = numpy.zeros((len(observations), self.n), dtype=self.precision)

        for s in range(self.n):
            beta[len(observations)-1][s] = 1.0

        for t in range(len(observations)-2, -1, -1):
            for i in range(self.n):
                for j in range(self.n):
                    beta[t][i] += self.A[i][j]*self.B_map[j][t+1]*beta[t+1][j]
        return beta

    def _calcxi(self, observations, alpha=None, beta=None):
        '''
        calc 'xi' the joint probability from the 'alpha' and 'beta' variables.
        xi[t][i][j] = the probability of being in state `i` at time `t`, and
        `j` at time `t+1` given the entire observations sequence.
        '''
        if alpha is None:
            alpha = self._calcalpha(observations)

        if beta is None:
            beta = self._calcbeta(observations)
        xi = numpy.zeros((len(observations), self.n, self.n), dtype=self.precision)

        for t in range(len(observations)-1):
            denom = 0.0
            for i in range(self.n):
                for j in range(self.n):
                    thing = alpha[t][i] * self.A[i][j] * self.B_map[j][t+1] * beta[t+1][j]
                    denom += thing
            for i in range(self.n):
                for j in range(self.n):
                    numer = alpha[t][i] * self.A[i][j] * self.B_map[j][t+1] * beta[t+1][j]
                    xi[t][i][j] = numer/denom
        return xi

    def _calcgamma(self, xi, seqlen):
        '''
        calc `gamma` from xi.
        gamma is (TxN) numpy array,
        where gamma[t][i] = the probability of being in state `i` at time `t`
        given the full observation sequence
        '''
        gamma = numpy.zeros((seqlen, self.n), dtype=self.precision)
        for t in range(seqlen):
            for i in range(self.n):
                gamma[t][i] = sum(xi[t][i])
        return gamma

    def log_likelihood(observations):
        pass

    def _updatemodel(self, new_model):
        self.pi = new_model['pi']
        self.A = new_model['A']

    def _mapB(self, observations):
        raise NotImplementedError("B(observable probabilities) must be implemented by Deriving class")
