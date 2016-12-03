import numpy as np
def mat_H(matrix_Hessian, s, y):
    #  Function approximates the Hessian of the Lagrangian by BFGS method
    #  http:#en.wikipedia.org/wiki/BFGS_method
    
    #  INPUT:
    #      matrix_Hessian - Hessian matrix from the previous iteration
    #      s - a direction such that s = x_new - x_old
    #      y = gxL(x_new, lam_new) - gxL(x_old, lam_new)
    
    #  OUTPUT:
    #      matrix_Hessian_next - a new Hessian approximation
    
    #=========================================================================

    
    global Hessian_approximation; #todo wut?
    
    if ( Hessian_approximation == "DBFGS"):
        #      DBFGS approximation scheme
        if (abs(s*y) >= 0.2*s*matrix_Hessian*s): #todo transpo abs(s'*y) >= 0.2*s'*matrix_Hessian*s then
            theta = 1
        else:
            theta = (0.8*s*matrix_Hessian*s)/(s*matrix_Hessian*s - s*y) #todo transpo theta = (0.8*s'*matrix_Hessian*s)/(s'*matrix_Hessian*s - s'*y);
        
        r = theta*y + (1-theta)*matrix_Hessian*s
        #todo transpo matrix_Hessian_next = matrix_Hessian - (matrix_Hessian*s*s'*matrix_Hessian)/(s'*matrix_Hessian*s) + (r*r')/(s'*r);
        matrix_Hessian_next = matrix_Hessian - (matrix_Hessian*s*s*matrix_Hessian)/(s*matrix_Hessian*s) + (r*r)/(s*r)
    elif ( Hessian_approximation == "SR1"):
#        disp(1e-8*norm(s)*norm(y - matrix_Hessian*s))
        #      SR-1 approximation scheme (indefinite matrix)
        if (abs(s*(y - matrix_Hessian*s)) < 1e-8*np.norm(s)*np.norm(y - matrix_Hessian*s)): #todo transpo abs(s'*(y - matrix_Hessian*s)) < 1e-8*norm(s)*norm(y - matrix_Hessian*s) then
            matrix_Hessian_next = matrix_Hessian
        elif (np.norm(y - matrix_Hessian*s) < 1e-8):
            matrix_Hessian_next = matrix_Hessian
        else:
            #todo transpo matrix_Hessian_next = matrix_Hessian + ((y-matrix_Hessian*s)*(y-matrix_Hessian*s)')/((y-matrix_Hessian*s)'*s);
            matrix_Hessian_next = matrix_Hessian + ((y-matrix_Hessian*s)*(y-matrix_Hessian*s))/((y-matrix_Hessian*s)*s)
    else:
    #elif ( Hessian_approximation == "BFGS"):
        #  BFGS approximation scheme
        if (s*y > 0): #todo transp s'*y
            matrix_Hessian_next = matrix_Hessian + (y*y')/(y'*s) - (matrix_Hessian*s*s'*matrix_Hessian)/(s'*matrix_Hessian*s);
        else:
            matrix_Hessian_next = matrix_Hessian
    

  return matrix_Hessian_next

def my_gillmurr(A, tol):
    #  Function computes the Cholesky factorization of a SPD matrix "A + E"
    #  so that A+E = R^TR, where E is an update
    #  INPUT:  a matrix "A", tol - a tolerance
    #  OUTPUT: an upper-triangular factor R, update D
    
    #  Init
    B = A
    n = size(A,1)
    R = np.zeros(n,n)
    E = np.zeros(n,n)
    max_absdiag = max(abs(diag(B)))
    bet = np.sqrt(max_absdiag) + 0.1    #  will change
    
    #  Iterate
    for k = in range(1,n):
        l = k+1
        gam = max(abs(B(k,l:-1)))
        ro = sqrt(max([abs(B(k,k)), (gam/bet)^2, tol^2]))
        R(k,k) = ro
        E(k,k) = ro^2 - B(k,k)
        if ro < tol:
            R(k,l:-1) = 0
        else:
            R(k,l:-1) = B(k,l:-1)/ro
        
        
        #  Update the rest
        B(l:,l:) = B(l:,l:) - R(k,l:)*R(k,l:) #todo transpose B(l:,l:) = B(l:,l:) - R(k,l:)'*R(k,l:)
    
  return [R, E]