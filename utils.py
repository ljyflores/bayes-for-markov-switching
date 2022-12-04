import pandas as pd
from scipy import linalg
import numpy as np
import math
from tqdm import tqdm

def hamFilt(x,y,S1,S2,phi0,phi1,P,S,mu):
    """
    Perform Hamilton Filtering to (1) compute the likelihood of each state, 
    and (2) sample the states S_{1:T} according to these likelihoods

    Outputs a list of length T with the sampled states
    """
    t = len(y)
    num_lags = x.shape[1]-1
    A = np.vstack((np.diag([1,1])-P, np.ones((1,2))))
    E = np.array([0,0,1])

    ett11 = linalg.inv(A.T @ A) @ A.T @ E
    filter = np.zeros((t,2))
    total_likelihood = 0 

    for i in range(t):
        prev_mu = [mu[idx] for idx in S[i-num_lags:i]] if i>=num_lags else [mu[idx] for idx in S[:i]] # Get state means for prev num_lags states
        if len(prev_mu) < num_lags: prev_mu = [0]*(num_lags-len(prev_mu)) + prev_mu # Pad in case too short
        prev_mu = prev_mu + [0] # Add [0] for the constant

        # First, compute likelihood of y_i given the past states (g(y_i|S_i))
        em1 = (y[i] - mu[0]) - ((x.iloc[i,].values - prev_mu) @ phi0)  # Compute residual
        em2 = (y[i] - mu[1]) - ((x.iloc[i,].values - prev_mu) @ phi1)  # Compute residual
        
        neta1 = (1/np.sqrt(2*math.pi*S1)) * np.exp((-1/(2*S1)) * (em1**2))  # Gaussian likelihood
        neta2 = (1/np.sqrt(2*math.pi*S2)) * np.exp((-1/(2*S2)) * (em2**2))  # Gaussian likelihood
        
        ett10 = P @ ett11

        ##update step##
        ett11 = np.multiply(ett10, np.array([neta1, neta2]).reshape(2))
        fit = np.sum(ett11)
        ett11 = ett11/fit
        filter[i,] = ett11
        total_likelihood += np.log(fit)

    # Perform sampling by drawing from Unif(0,1)
    # Then assign state based on sampled value and transition matrix cutoffs

    # First, sample the last state
    S = np.zeros(t)
    p1 = filter[-1,0]
    p2 = filter[-1,1]
    p = p1/(p1+p2)
    u = np.random.uniform(0,1,1) # Unif(0,1), draw 1
    S[-1] = 1 if u>=p else 0

    # Then, backward sample the rest
    for i in range(t-2,-1,-1):
        if S[i]==0:
            p00 =P[0,0] * filter[i,0]
            p01 =P[0,1] * filter[i,1]
        
        if S[i]==1:
            p00 =P[1,0] * filter[i,0]
            p01 =P[1,1] * filter[i,1]
        
        u = np.random.uniform(0,1,1) # Unif(0,1), draw 1
        p = p00/(p01+p00)
        S[i] = 0 if u<p else 1

    S = S.astype(int) # Use integers!
    return(S)


def lag_matrix(s,lag, dropna=True):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    Source: https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
    '''
    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in range(1,lag+1):
                new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
        res=pd.DataFrame(new_dict,index=s.index)

    elif type(s) is pd.Series:
        res=pd.concat([s.shift(i) for i in range(1,lag+1)],axis=1)
        res.columns=['lag_%d' %i for i in range(1,lag+1)]
    else:
        print('Only works for DataFrame or Series')
        return None

    if dropna:
        return res.dropna()
    else:
        return res 

def switchcount(s,g):
    """
    Count the number of transitions from state i->j
    For all pairs (i,j)
    """
    swt = np.zeros((len(g),len(g))) # Matrix of transition counts

    for t in range(len(s)):
        st_prev, st_curr = int(s[t-1]), int(s[t]) # Get previous and current states
        swt[st_prev][st_curr] += 1
    return swt



def transmat_post(u00,u01,u11,u10,S,G):
    """
    Sample new transition probabilities from beta distribution
    Use prior beta distribution with params u00,u01,u11,u10
    Conditioned on previous states S
    """

    tranmat = switchcount(s=S,g=G) # Count the number of transitions for each pair (i,j)
    
    # Unpack transition counts
    N00 = tranmat[0][0]
    N01 = tranmat[0][1]
    N10 = tranmat[1][0]
    N11 = tranmat[1][1]

    #draw for beta distribution
    p = np.random.beta(N00 + u00, N01 + u01, size=1)
    q = np.random.beta(N10 + u10, N11 + u11, size=1)
    pmat = np.array([[p,1-p],[1-q,q]]).reshape(2,2)

    return(pmat)


def beta_post(x, y, S, sig0, sig1, A0, a0, B0, b0, v0, d0, mu, phi0, phi1):
    """
    Sample phi (slope coefs), mu (state means), and sigma (state variances)
    """
    chol_coef = x.shape[1]
    num_lags = x.shape[1]-1

    # Sample mu = (mu_0, mu_1)
    s0 = []
    for i in range(x.shape[0]):
        # For each timestep, sum the coefficients belonging to that state 
        # See Markov-Switching Models and Gibbs-Sampling, Kim, 2017 (p. 231)
        s_row_0 = 1.0 if S[i]==0 else 0.0 
        s_row_1 = 1.0 if S[i]==1 else 0.0

        lagged_state_lst = S[i-num_lags:i] if i>=num_lags else np.array([0]*(num_lags-i)+list(S[:i]))

        for lag_idx, lag_state in enumerate(lagged_state_lst):
            if lag_state==0: 
                s_row_0 -= phi0[lag_idx]

            elif lag_state==1: 
                s_row_1 -= phi1[lag_idx]

            else: assert False
        s0.append([s_row_0, s_row_1])
    s0 = np.array(s0) # Gives us T x num_lags matrix, each item represents a state variance

    phi_lst = np.array([phi0 if s==0 else phi1 for s in S]).reshape(-1, chol_coef) # Get the coefficients (phi) of each state
    sig_lst = np.array([sig0 if s==0 else sig1 for s in S]) # Get the variance (sigma) of each state

    y0 = (y - np.multiply(x.values, phi_lst).sum(axis=1)) # Gives us a T x 1 matrix, each row represents the residual
    # print(s0, sig_lst)
    s0 = (s0.T/sig_lst).T # Divide row i by sigma_{S_i}, which is the state variance for time i
    y0 = np.divide(y0, sig_lst) # Divide residual i by sigma_{S_i}, which is the state variance for time i
    
    M = linalg.inv(linalg.inv(A0) + (s0.T @ s0)) @ ((linalg.inv(A0) @ a0) + (s0.T @ y0.values.reshape(-1,1)))
    V = linalg.inv(linalg.inv(A0) + (s0.T @ s0)) 
    mu = np.random.multivariate_normal(M.reshape(-1), V, 1)[0]

    # Sample phi_0
    x0, y0 = (x.iloc[S==0,]-mu[0]).values, (y[S==0]-mu[0]).values # Obtain data for state 0, remove mean
    if len(x0)==B0.shape[0]: x0 = np.array(x0).T

    M = linalg.inv(linalg.inv(B0) + (1/sig0) * (x0.T @ x0)) @ ((linalg.inv(B0) @ b0)+ ((1/sig0) * x0.T @ y0).reshape(-1,1))
    V = linalg.inv(linalg.inv(B0) + (1/sig0) * (x0.T @ x0)) 
    phi0 = np.random.multivariate_normal(M.reshape(-1), V, 1)[0]
    # print("phi0", phi0.shape)

    # Sample phi_1
    x1, y1 = (x.iloc[S==1,]-mu[1]).values, (y[S==1]-mu[1]).values # Obtain data for state 1, remove mean
    if len(x1)==B0.shape[0]: x1 = np.array(x1).T

    M = linalg.inv(linalg.inv(B0) + (1/sig1) * (x1.T @ x1)) @ ((linalg.inv(B0) @ b0) + ((1/sig1) * x1.T @ y1).reshape(-1,1))
    V = linalg.inv(linalg.inv(B0) + (1/sig1) * (x1.T @ x1)) 
    phi1 = np.random.multivariate_normal(M.reshape(-1), V, 1)[0]
    # print("phi1", phi1.shape)

    # Sample sigma_0
    e0 = y0 - (x0 @ phi0)
    D0 = (d0 + (e0.T @ e0))/2
    T0 = (v0 + e0.shape[0])/2
    sigma0 = 1 / np.random.gamma(shape=T0, scale=1/D0, size=1)[0]

    # Sample sigma_1
    e1 = y1 - (x1 @ phi1)
    D1 = (d0 + (e1.T @ e1))/2
    T1 = (v0 + e1.shape[0])/2
    sigma1 = 1 / np.random.gamma(shape=T1, scale=1/D1, size=1)[0]

    return {'phi0': phi0, 'phi1': phi1, 'sigma0': sigma0, 'sigma1': sigma1, 'mu': mu}

def preprocess_ts(y, lags):
    y = (y-min(y))/(max(y)-min(y)) # Standardize
    x = lag_matrix(y, lags, dropna=False) # Get lags
    x['constant'] = 1

    x = x.iloc[lags:].reset_index(drop=True)
    y = y[lags:].reset_index(drop=True)
    return x, y

#MCMC
def estimate(x, y, reps, num_states, sig0, sig1, p, q, d0, v0, u00, u01, u10, u11):
    
    ### Priors ###
    b0 = np.zeros((x.shape[1],1)) # num_lags x 1
    B0 = np.diag([100]*x.shape[1]) # num_lags x num_lags

    a0 = np.zeros((num_states,1)) # num_states x 1
    A0 = np.diag([100]*num_states) # num_states x num_states

    phi0 = np.zeros((x.shape[1],1))
    phi1 = np.zeros((x.shape[1],1))
    pmat = np.array([[p,1-p],[1-q,q]])

    S = np.random.binomial(1, 0.5, size=x.shape[0])
    mu = (x.iloc[:,:-1].mean(axis=0)).values

    ### Estimate ###
    out1 = [] #States/Regimes
    out2 = [] #Transition Probabilities
    out3 = [] #Beta/Variance 

    for i in tqdm(range(1, reps+1)):    
        # Sample probabilities of states per timestep using Hamilton Filter
        S = hamFilt(x=x,
                    y=y,
                    S1=sig0,
                    S2=sig1,
                    phi0=phi0,
                    phi1=phi1,
                    P=pmat,
                    S=S,
                    mu=mu) 
        # Sample transition probabilities from beta distribution
        pmat = transmat_post(u00 = u00,
                            u01 = u01,
                            u11 = u11,
                            u10 = u10, 
                            S   = S,
                            G   = np.array([1,2]).reshape(2))
        # Sample means (mu), variances (sigma), and coefficients (phi)
        new_params = beta_post(x=x, 
                            y=y, 
                            S=S, 
                            sig0=sig0, 
                            sig1=sig1, 
                            A0=A0, 
                            a0=a0, 
                            B0=B0, 
                            b0=b0, 
                            v0=v0, 
                            d0=d0, 
                            mu=mu, 
                            phi0=phi0.reshape(-1),
                            phi1=phi1.reshape(-1))
        phi0 = new_params['phi0'].reshape(-1,1)
        phi1 = new_params['phi1'].reshape(-1,1)
        sig0 = new_params['sigma0']
        sig1 = new_params['sigma1']
        mu   = new_params['mu']

        # Impose identification restriction on sigma
        if sig0 > sig1:
            S = 1-S
            pmat = pmat[::-1][:,::-1]
            phi0, phi1 = phi1, phi0
            sig0, sig1 = sig1, sig0
            mu = mu[::-1]
            new_params = {'phi0':phi0, 'phi1':phi1, 'sigma0':sig0, 'sigma1':sig1, 'mu':mu}

        out1.append(S)
        out2.append(pmat)
        out3.append(new_params)

        if i>50:
            out1 = out1[-50:]
            out2 = out2[-50:]
            out3 = out3[-50:]
    return out1, out2, out3