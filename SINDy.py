import numpy as np
from scipy.special import binom
from scipy.integrate import odeint
import pdb  

def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

def sindy_library(X, poly_order, include_sine=False):
    symbs = ['x', 'y', 'z', '4', '5', '6', '7']
    m, n = X.shape
    l = library_size(n, poly_order, include_sine, True)
    library = np.ones((m, l))
    sparse_weights = np.ones((l, n))
    
    index = 1
    names = ['1']

    for i in range(n):
        library[:,index] = X[:,i]
        sparse_weights[index, :] *= 1
        names.append(symbs[i])
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                sparse_weights[index, :] *= 2
                names.append(symbs[i]+symbs[j])
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    sparse_weights[index, :] *= 3
                    names.append(symbs[i]+symbs[j]+symbs[k])
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        sparse_weights[index, :] *= 4
                        names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q])
                        index += 1

    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            sparse_weights[index, :] *= 5
                            names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q]+symbs[r])
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            names.append('sin('+symbs[i]+')')
            index += 1
        
    return_list = [library]
    return return_list


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine=False, exact_features=False):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine, include_names=False, exact_features=exact_features), Xi).reshape((n,))
    x = odeint(f, x0, t)
    return x





#TO ONLY GET NAMES OF OUR VARIABLES WITHOUT COMPUTING THE LIBRARY
def sindy_library_names(latent_dim, poly_order, include_sine=False):
    # Upgrade to combinations
    symbs = ['x', 'y', 'z', '4', '5', '6', '7']
     
    n = latent_dim
    index = 1
    names = ['1']

    for i in range(n):
        names.append(symbs[i])
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                names.append(symbs[i]+symbs[j])
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    names.append(symbs[i]+symbs[j]+symbs[k])
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q])
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q]+symbs[r])
                            index += 1

    if include_sine:
        for i in range(n):
            names.append('sin('+symbs[i]+')')
            index += 1
    
    return names