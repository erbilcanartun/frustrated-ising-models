import numpy as np
from scipy.misc import derivative

# References:
# [1] McKay, Berker, Kirkpatrick (1982) "Spin-Glass Behavior in Frustrated Ising Models with Chaotic Renormalization-Group Trajectories", PRL.

# Parameters:
# Couplings: -βℋ(ij) = K.σ(i)σ(j)
# K: coupling constant
# m1 (m2): number of bonds in the short (long) strand in unit Ⓑ (Ⓒ).
# p: number of parallel bonds in unit Ⓑ.
# pb (pc): number of parallel units Ⓑ (Ⓒ).


# Equation given in Ref.[1]
def renormalize_close_form(K, m1, m2, p, pb, pc):

    t = np.tanh(K)
    t_ = np.tanh(p * K)
    tb_ = (2 * t**2 * (1 - t_)) / (1 + t**4 - 2 * t**2 * t_)
    
    return pb * np.arctanh(tb_) + pc * (np.arctanh(t**m1) - np.arctanh(t**m2))


# Renormalization group function
def decimate(j1, j2):
    
    """
    Decimation procedure with rescaling factor b = 2.
    
    It is the exact renormalization-group transformation of
    spin-1/2 Ising model without external field in one dimension.
    """
    
    e1 = j1 + j2
    e2 = -j1 - j2
    emax = np.amax([e1, e2])
    
    lnR1 = emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))
    
    e1 = j1 - j2
    e2 = -j1 + j2
    emax = np.amax([e1, e2])
    
    lnR2 =  emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

    return (lnR1 - lnR2) / 2


def renormalize(K, m1, m2, p, pb, pc):
    
    # Unit Ⓑ
    
    e1 = (4 - p) * K
    e2 = -(4 + p) * K
    e3 = p * K
    emax = np.amax([e1, e2, e3])
    
    lnR1 = emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax) + 2*np.exp(e3 - emax))
    
    e1 = -p * K
    e2 = p * K
    emax = np.amax([e1, e2])
    
    lnR2 = emax + np.log(2 * np.exp(e1 - emax) + 2*np.exp(e2 - emax))
    
    B = (lnR1 - lnR2) / 2
    
    # Unit Ⓒ
    
    # The shorter strand with m1 bonds
    if m1 >= 2:
        
        k1 = decimate(K, K)
        for i in range(m1 - 2):
            k1 = decimate(K, k1)
            
    elif m1 == 1:
        k1 = K
    
    else:
        k1 = 0
    
    # The longer strand with m2 bonds
    if m2 >= 2:
        
        k2 = decimate(K, -K)
        for i in range(m2 - 2):
            k2 = decimate(K, k2)
            
    elif m2 == 1:
        k2 = -K
        
    else:
        k2 = 0
    
    C = k1 + k2
    
    return pb * B + pc * C

def renormalization_group_flow(K, m1, m2, p, pb, pc, n=100):

    flow = [[0, K]]
    for k in range(1, n):
        
        K = renormalize(K, m1, m2, p, pb, pc)
        flow.append([k, K])

    return np.array(flow)

# Derivative of the recursion relation in Ref.[1]
def renormalization_derivative(K, m1, m2, p, pb, pc):
    
    return pc*((m1*tanh(K)**(m1 - 1)*(tanh(K)**2 - 1))/(tanh(K)**(2*m1) - 1) - (m2*tanh(K)**(m2 - 1)*(tanh(K)**2 - 1))/(tanh(K)**(2*m2) - 1)) - (pb*((2*tanh(K)**2*(tanh(K*p) - 1)*(4*tanh(K*p)*tanh(K)*(tanh(K)**2 - 1) - 4*tanh(K)**3*(tanh(K)**2 - 1) + 2*p*tanh(K)**2*(tanh(K*p)**2 - 1)))/(tanh(K)**4 - 2*tanh(K*p)*tanh(K)**2 + 1)**2 + (2*p*tanh(K)**2*(tanh(K*p)**2 - 1))/(tanh(K)**4 - 2*tanh(K*p)*tanh(K)**2 + 1) + (4*tanh(K)*(tanh(K)**2 - 1)*(tanh(K*p) - 1))/(tanh(K)**4 - 2*tanh(K*p)*tanh(K)**2 + 1)))/((4*tanh(K)**4*(tanh(K*p) - 1)**2)/(tanh(K)**4 - 2*tanh(K*p)*tanh(K)**2 + 1)**2 - 1)


# Lyapunov exponent calculator
def lyapunov_exponent(m1=7, m2=8, p=4, pb=40, pc=1, iteration_number=10000, transient=100):

    K = 1
    K_series = []
    for i in range(iteration_number):
        
        K_series.append(K)
        K = renormalization(K, m1, m2, p, pb, pc)
        
    lyapunov_series = []
    for K in K_series:
        
        lyapunov = derivative(renormalize, x0=K, dx=1e-6, args=(m1, m2, p, pb, pc,))
        lyapunov_series.append(np.log(abs(lyapunov)))
    
    # Averaging with dropping off the first 100 iterations to allow for an initial transient
    lyapunov_exponent = np.mean(lyapunov_series[100:])
        
    return lyapunov_exponent