# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:08:11 2016

@author: jan

Library contains the following Quasi-Newton methods for Hessian updates:
    BFGS
    DBFGS
    SRone
    DFP
For help call help(bfgs), help(dbfgs), help(srone) or help(dfp).
"""

#import numpy, scipy and linalg
import numpy as np
from scipy import linalg as la

#   DEFINITIONS OF FUNCTIONS FOLLOWS

def bfgs(M, s, y, inv = False, tol = 0):
    """
    Function bfgs(M, s, y, inv = False, tol = 0) computes a quasi-Newton
    update by the BFGS method. One uses it when the Hessian is symmetric
    positive definite.
    
    PARAMETERS
    ----------
    M : is an `n` by `n` symmetric positive definite matrix.
    s : is a vector with `n` components such that a new iterate `x^+ = x + s`, 
    where `s` is a descent direction.
    y: is a vector with `n` components such that `y = grad f^+ - grad f`,
    where `grad f` is a gradient of an objective function. Here `+` denostes
    a data in the next iteration.
    inv : is an optional parameter. When `inv = False` the function computes
    a quasi-Newton update for matrix `M`. When `inv = True` it computes
    a quasi-Newton update for the inverse `M^{-1}`.
    tol : is an optional parameter and whenever `s.T*y < tol` the quasi-Newton
    update gets skipped.
    
    RETURNS
    -------
    M : is an `n` by `n` symmetric positive definite matrix
    flag : gives `1 or 0` marking the update was `skipped or updated`.
    
    RAISES
    ------
    Function raises no error and warnings. However, it does not check whether
    dimensions of `M`, `s` and `y` are correct and that `M or M^{-1}` is an
    symmetric positive definite matrix.
    
    IMPORTANT
    ---------
    When `inv = False` then we use the function bfgs(M, s, y, ....) to get
    new `M`. However, when `inv = True` one needs to call the function by
    bfgs(M^{-1}, s, y, ...) to get new `M^{-1}`!
    
    NOTES
    -----
    There is a reference worth looking at: J. Nocedal, S. J. Wright: Numerical
    Optimization (Second edition), 2006. Formulas and ideas for this script is
    taken from Chapter 6.
    You can also look on Wikipedia for the `BFGS method`.
    """
    yds = np.dot(y, s)
    #To update or not to update?
    if yds > tol:
        rho = 1/yds
        #   Update the Hessian and select the right scheme
        if inv:
            #   Use the formula for updating the inverse of the Hessian
            row, col = np.shape(M)
            identity = np.eye(row)
            syT = np.outer(s, y)
            ysT = np.outer(y, s)
            ssT = np.outer(s, s)
            #   Get three matrices for the final formula
            A = identity - rho*syT
            B = identity - rho*ysT
            C = rho*ssT
            #   Return the updated matrix, flag
            return np.dot(A, np.dot(M, B)) + C, 0
        else:
            #   Use the formula for a standard update
            yyT = np.outer(y,y)
            Ms = np.dot(M, s)
            sTM = np.dot(s, M)
            #   Return the updated matrix, flag
            return M - np.dot(Ms, sTM)/np.dot(s, Ms) + rho*yyT, 0
    else:
        #   Skip the update
        return M, 1
        
def srone(M, s, y, inv = False, tol = 0):
    """
    Function srone(M, s, y, inv = False, tol = 0) computes a quasi-Newton
    update by the SR-1 method. One uses it when the Hessian is indefinite.
    
    PARAMETERS
    ----------
    M : is an `n` by `n` symmetric indefinite definite matrix.
    s : is a vector with `n` components such that a new iterate `x^+ = x + s`, 
    where `s` is a descent direction.
    y: is a vector with `n` components such that `y = grad f^+ - grad f`,
    where `grad f` is a gradient of an objective function. Here `+` denostes
    a data in the next iteration.
    inv : is an optional parameter. When `inv = False` the function computes
    a quasi-Newton update for matrix `M`. When `inv = True` it computes
    a quasi-Newton update for the inverse `M^{-1}`.
    tol : is an optional parameter and says then the quasi-Newton update gets 
    skipped.
    
    RETURNS
    -------
    M : is an `n` by `n` symmetric indefinite matrix
    flag : gives `1 or 0` marking the update was `skipped or updated`.
    
    RAISES
    ------
    Function raises no error and warnings. However, it does not check whether
    dimensions of `M`, `s` and `y` are correct and that `M or M^{-1}` is an
    symmetric indefinite matrix.
    
    IMPORTANT
    ---------
    When `inv = False` then we use the function srone(M, s, y, ....) to get
    new `M`. However, when `inv = True` one needs to call the function by
    srone(M^{-1}, s, y, ...) to get new `M^{-1}`!
    
    NOTES
    -----
    There is a reference worth looking at: J. Nocedal, S. J. Wright: Numerical
    Optimization (Second edition), 2006. Formulas and ideas for this script is
    taken from Chapter 6.
    You can also look on Wikipedia for the `SR-1 method`. 
    """
    r = 1e-8
    #   Prepare data
    if inv:
        #   The inverse scheme
        My = np.dot(M, y)
        v = s - My
        rho = np.dot(v, y)
        lb = r*la.norm(y)*la.norm(v)
    else:
        #   Standard update
        Ms = np.dot(M, s)
        v = y - Ms
        rho = np.dot(v, s)
        lb = r*la.norm(s)*la.norm(v)
    #   Condition on the application of an update
    if tol:
        #   Use users tolerance
        cond = np.abs(rho) >= tol
    else:
        #   If no tolerance is given, use this heuresitc
        cond = np.abs(rho) >= lb
    #   To update or not to update
    if cond:
        #   Update the Hessian
        return M + rho*np.outer(v,v), 0
    else:
        #   Skip the update
        return M, 1
        
def dfp(M, s, y, inv = False, tol = 0):
    """
    Function dfp(M, s, y, inv = False, tol = 0) computes a quasi-Newton
    update by the DFP method. One uses it when the Hessian is symmetric
    positive definite.
    
    PARAMETERS
    ----------
    M : is an `n` by `n` symmetric positive definite matrix.
    s : is a vector with `n` components such that a new iterate `x^+ = x + s`, 
    where `s` is a descent direction.
    y: is a vector with `n` components such that `y = grad f^+ - grad f`,
    where `grad f` is a gradient of an objective function. Here `+` denostes
    a data in the next iteration.
    inv : is an optional parameter. When `inv = False` the function computes
    a quasi-Newton update for matrix `M`. When `inv = True` it computes
    a quasi-Newton update for the inverse `M^{-1}`.
    tol : is an optional parameter and whenever `s.T*y < tol` the quasi-Newton
    update gets skipped.
    
    RETURNS
    -------
    M : is an `n` by `n` symmetric positive definite matrix
    flag : gives `1 or 0` marking the update was `skipped or updated`.
    
    RAISES
    ------
    Function raises no error and warnings. However, it does not check whether
    dimensions of `M`, `s` and `y` are correct and that `M or M^{-1}` is an
    symmetric positive definite matrix.
    
    IMPORTANT
    ---------
    When `inv = False` then we use the function bfgs(M, s, y, ....) to get
    new `M`. However, when `inv = True` one needs to call the function by
    bfgs(M^{-1}, s, y, ...) to get new `M^{-1}`!
    
    In a sence it is a dual method to the BFGS method. We intechange roles
    of `s` and `y`.
    
    NOTES
    -----
    There is a reference worth looking at: J. Nocedal, S. J. Wright: Numerical
    Optimization (Second edition), 2006. Formulas and ideas for this script is
    taken from Chapter 6.
    You can also look on Wikipedia for the `DFP method`.
    """
    if inv:
        #   Use the bfgs function for `y->s` and `s->y`, `inv = False`
        return bfgs(M, y, s, False, tol)
    else:
        #   Use the bfgs function for `y->s` and `s->y`, `inv = True`
        return bfgs(M, y, s, True, tol)
        
def dbfgs(M, s, y, tol = 0.2):
    """
    Function dbfgs(M, s, y, tol = 0.2) computes a quasi-Newton
    update by the DBFGS method. One uses it when the Hessian is symmetric
    positive definite.
    
    PARAMETERS
    ----------
    M : is an `n` by `n` symmetric positive definite matrix.
    s : is a vector with `n` components such that a new iterate `x^+ = x + s`, 
    where `s` is a descent direction.
    y: is a vector with `n` components such that `y = grad f^+ - grad f`,
    where `grad f` is a gradient of an objective function. Here `+` denostes
    a data in the next iteration.
    tol : is an optional parameter. Function uses parameter `tol` in this 
    fashion: it checks if `s.T*y >=tol*s.T*M*s`. If this holds it takes a 
    BFGS update.
    
    RETURNS
    -------
    M : is an `n` by `n` symmetric positive definite matrix
    
    RAISES
    ------
    Function raises no error and warnings. However, it does not check whether
    dimensions of `M`, `s` and `y` are correct and that `M is an
    symmetric positive definite matrix.
    
    NOTES
    -----
    There is a reference worth looking at: J. Nocedal, S. J. Wright: Numerical
    Optimization (Second edition), 2006. Formulas and ideas for this script is
    taken from Chapter 18.
    You can also look on Wikipedia for the `DBFGS method`.
    """
    #   Prepare data
    sTy = np.dot(s, y)
    Ms = np.dot(M, s)
    sTM = np.dot(s, M)
    sTMs = np.dot(s, Ms)
    #   Get parameter theta
    if sTy >= tol*sTMs:
        theta = 1
    else:
        theta = 0.8*sTMs/(sTMs - sTy)
    #   Get vector r we later substitute for `y` in BFGS formula
    r = theta*y + (1 - theta)*Ms
    rrT = np.outer(r, r)
    rho = 1/np.dot(s, r)
    #   Use BFGS cheme
    return M - np.outer(Ms, sTM)/sTMs + rho*rrT