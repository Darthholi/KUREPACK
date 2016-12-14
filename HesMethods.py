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

    
    global Hessian_approximation #todo wut?
    
    if ( Hessian_approximation == "DBFGS"):
        #      DBFGS approximation scheme
        Hs = np.dot(matrix_Hessian,s)
        sTHs=np.vdot(s,Hs)
        if (abs(np.vdot(s,y)) >= 0.2*sTHs): #was: abs(s'*y) >= 0.2*s'*matrix_Hessian*s then
            theta = 1
        else:
            theta = (0.8*sTHs)/(sTHs - np.vdot(s,y)) #was transpo theta = (0.8*s'*matrix_Hessian*s)/(s'*matrix_Hessian*s - s'*y);
        
        r = theta*y + (1-theta)*Hs
        #was matrix_Hessian_next = matrix_Hessian - (matrix_Hessian*s*s'*matrix_Hessian)/(s'*matrix_Hessian*s) + (r*r')/(s'*r);
        matrix_Hessian_next = matrix_Hessian - (np.vdot(Hs,Hs))/(sTHs) + (np.vdot(r,r))/(np.vdot(s,r))
    elif ( Hessian_approximation == "SR1"):
#        disp(1e-8*norm(s)*norm(y - matrix_Hessian*s))
        #      SR-1 approximation scheme (indefinite matrix)
        Hs = np.dot(matrix_Hessian,s)
        if (abs(np.vdot(s,Hs) < 1e-8*np.norm(s)*np.norm(y - Hs)): #was abs(s'*(y - matrix_Hessian*s)) < 1e-8*norm(s)*norm(y - matrix_Hessian*s) then
            matrix_Hessian_next = matrix_Hessian
        elif (np.norm(y - Hs) < 1e-8):
            matrix_Hessian_next = matrix_Hessian
        else:
            #was matrix_Hessian_next = matrix_Hessian + ((y-matrix_Hessian*s)*(y-matrix_Hessian*s)')/((y-matrix_Hessian*s)'*s);
            matrix_Hessian_next = matrix_Hessian + (np.vdot(y-Hs,y-Hs))/(np.vdot(y-Hs,s))
    else:
    #elif ( Hessian_approximation == "BFGS"):
        #  BFGS approximation scheme
        if (np.vdot(s,y) > 0): #was transp s'*y
            #was matrix_Hessian_next = matrix_Hessian + (y*y')/(y'*s) - (matrix_Hessian*s*s'*matrix_Hessian)/(s'*matrix_Hessian*s);
            Hs = np.dot(matrix_Hessian, s)
            matrix_Hessian_next = matrix_Hessian + (np.vdot(y, y)) / (np.vdot(y, s)) - (np.vdot(Hs, Hs)) / (np.vdot(s, Hs))
        else:
            matrix_Hessian_next = matrix_Hessian
    

    return matrix_Hessian_next

def my_gillmurr(A, tol):
    #  Function computes the Cholesky factorization of a SPD matrix "A + E"
    #  so that A+E = R**TR, where E is an update
    #  INPUT:  a matrix "A", tol - a tolerance
    #  OUTPUT: an upper-triangular factor R, update D
    
    #  Init
    B = A
    n = size(A,1)
    R = np.zeros(n,n)
    E = np.zeros(n,n)
    max_absdiag = np.max(np.abs(np.diag(B)))
    bet = np.sqrt(max_absdiag) + 0.1    #  will change
    
    #  Iterate
    for k in range(0,n-1):
        l = k+1
        gam = max(abs(B[k,l:-1]))
        ro = np.sqrt(max([abs(B(k,k)), (gam/bet)**2, tol**2]))
        R[k,k] = ro
        E[k,k] = ro**2 - B[k,k]
        if ro < tol:
            R[k,l:-1] = 0
        else:
            R[k,l:-1] = B[k,l:-1]/ro
        
        
        #  Update the rest
        B[l:,l:] = B[l:,l:] - np.dot(np.transpose(R[k,l:]),R[k,l:]) #todo transpose B(l:,l:) = B(l:,l:) - R(k,l:)'*R(k,l:)
    
    return [R, E]

def mat_H_block(matrix_Hessian, blocks_OLD, s_v, y_v, statespace_dimension, number_of_segments):
    #  Function approximates the Hessian of the Lagrangian by BFGS method
    #  http:#en.wikipedia.org/wiki/BFGS_method
    #  This version keeps the sparsity of the Matrix matrix_Hessian. BFGS method is applied
    #  on each diagonal block separately!
    
    #  INPUT:
    #      matrix_Hessian - Hessian matrix from the previous iteration
    #      blocks_OLD - block structure from the last iteration
    #      s_v - a direction such that s = x_new - x_old
    #      y_v = gxL(x_new, lam_new) - gxL(x_old, lam_new)
    #      statespace_dimension - dimension of the state space
    #      number_of_segments - number of segments
    
    #  OUTPUT:
    #      matrix_Hessian_new - a new Hessian approximation
    #      blocks_NEW - I need to keep current blocks for the next iteration
    
    #=========================================================================
    
    #  Data initialization
    matrix_Hessian_new = []
    blocks_NEW = []
    number_of_blocks = size(blocks_OLD, 1)/size(blocks_OLD, 2)
    size_of_blocks = size(blocks_OLD, 2)
    
    #  We split vectors s and y = lag_grad_next-lag_grad into number_of_segments vectors, for we have number_of_segments blocks
    #if (PROBLEM_FORMULATION == 2):
    #    #  Lengths are fixed => we have only Nn parameters
    #    S = matrix(s_v, statespace_dimension, number_of_segments)
    #    Y = matrix(y_v, statespace_dimension, number_of_segments)
    #else:
    #    #  Lengths are not fixed => we have N(n+1) parameters
    S = matrix(s_v, statespace_dimension+1, number_of_segments)
    Y = matrix(y_v, statespace_dimension+1, number_of_segments)
    

    #  I update every blocks so that blocks_NEW = blocks_OLD + UPDATE
    if (number_of_blocks==1):
        #  This is the full BFGS update
        [blocks_NEW] = mat_H(blocks_OLD, s_v, y_v)
        
        #  Construct the Hessian matrix for the KKT system
        matrix_Hessian_new = blocks_NEW

    elif (number_of_blocks == number_of_segments):
        #  This is the block-diagonal update
        for i in range(number_of_blocks): #i = 1:number_of_blocks
            #  Extract one block
            Hb = blocks_OLD(size_of_blocks*(i-1)+1:i*size_of_blocks, :) #todo
            
            #  Block-wise BFGS
            [Hb_update] = mat_H(Hb, S[:,i], Y[:,i])
            blocks_NEW = [blocks_NEW, Hb_update]
            matrix_Hessian_new = sysdiag(matrix_Hessian_new, Hb_update)
        

    elif (number_of_blocks == number_of_segments-1):
        #  data
        ZB_SMALL = np.zeros(size_of_blocks, size_of_blocks)

        #  This is the banded Hessian with width of the band 2*n or 2*(n+1)
        for i in range(number_of_blocks): #= 1:number_of_blocks
            #  Extract one block
            Hb = blocks_OLD(size_of_blocks*(i-1) + 1: i*size_of_blocks, :) #todo
            
            #  Block-wise BFGS (overlapping blocks
            [Hb_update] = mat_H(Hb, [[S[:, i]; S[:, i+1]], [Y[:, i], Y[:, i+1]]])
            blocks_NEW = [blocks_NEW, Hb_update]
            ZB_BIG = np.zeros(size(matrix_Hessian_new, 1) - size_of_blocks/2, size(matrix_Hessian_new, 1) - size_of_blocks/2)
            matrix_Hessian_new = sysdiag(matrix_Hessian_new, ZB_SMALL) + sysdiag(ZB_BIG, Hb_update)
            ZB_SMALL = np.zeros(size_of_blocks/2, size_of_blocks/2)
        
    
    return [matrix_Hessian_new, blocks_NEW]

def my_Hess_linode(lam, seg_init, seg_len, ode_A, cen_U, ell_I, ell_U, my_ode):
        #  This function approximates the Hessian of the Lagrangian using
        #  the structure and information I have at hand. I try to avoid
        #  using BFGS or any other for the Hessian approximation
        
        #  OUTPUT: a block-diagonal SPD Hessian approximation
        
        
        #  initialize
        [statespace_dim, number_of_seg] = size(seg_init)
        AT_sqr = (ode_A*ode_A)
        Hess_new = []
        lam_I = lam(0)
        lam_U = lam(-1)
        xambda = lam
        xlambda[0] = []
        xlambda[-1] = []
        D = np.zeros(statespace_dim, statespace_dim)
                
        #  iterate over blocks 1...number_of_seg
        #  the last update at the end
        for k = xrange(number_of_seg-2): #k=1:numberofseg-1
            #  get e**{At_i}**T
            sen_mat = expm(ode_A*seg_len(k))
            #  stack mixed second derivatives (space-time)
            V_stack = -(ode_A*sen_mat)';
            #  stack second derivatives with respect to time
#            alpha_stack = -seg_init(:,k)'*sen_mat*AT_sqr; --- # BAD
            alpha_stack = -AT_sqr*sen_mat*seg_init(:,k);
            #  perform the index contraction by lambda
            v_i = V_stack*xlambda((k-1)*statespace_dim+1:k*statespace_dim);
            alpha_i = alpha_stack'*lambda((k-1)*statespace_dim+1:k*statespace_dim);
            #  build the block of the Hessian matrix
            if k ==1 then
                H_block = [lam_I*ell_I, v_i; v_i', alpha_i + 1];
            else
                H_block = [D, v_i; v_i', alpha_i + 1];
            end
            #  add this block to previously computed ones
            Hess_new = sysdiag(Hess_new, H_block);
        end
        #  the last block
        eAt = expm(ode_A*seg_len($));
        eAtx = eAt*seg_init(:,$);
        AeAt = ode_A*eAt;
        AeAtx = AeAt*seg_init(:,$);
        eAtxc = eAtx - cen_U;
        AAeAtx = ode_A*AeAtx;
        #  create blocks in the last block
        M = eAt'*ell_U*eAt;
        v_i = AeAt'*ell_U*eAtxc + eAt'*ell_U*AeAtx;
        alpha_i = AAeAtx'*ell_U*eAtxc + AeAtx'*ell_U*AeAtx;
        H_block = lam_U*[M, v_i; v_i', alpha_i]
        H_block($,$) = H_block($,$) + 1;
        Hess_new = sysdiag(Hess_new, H_block);
        return [Hess_new]


function [Hess_new, D_new] = my_Hess_nonlinode(lam, seg_init, seg_len, ode_A, D_old,...
                                    s, y, cen_U, ell_I, ell_U, my_ode, FINITE_DIFFERENCE,...
                                    ODE_LIST)
        #  This function approximates the Hessian of the Lagrangian using
        #  the structure and information I have at hand. I try to avoid
        #  using BFGS or any other for the Hessian approximation
        
        #  OUTPUT: a block-diagonal SPD Hessian approximation
        
        
        #  initialize
        [statespace_dim, number_of_seg] = size(seg_init);
        Hess_new = [];
        lam_I = lam(1);
        lam_U = lam($);
        lambda = lam;
        lambda(1) = [];
        lambda($) = [];
        D_new = [];
        
        #  remove time and keep only state variables in s, y
        s = matrix(s,statespace_dim+1, number_of_seg);
        s($,:) = [];
        y = matrix(y,statespace_dim+1, number_of_seg);
        y($,:) = [];
        
        #  I will get 1st - (N-1)st block in the Hessian
        for k = 1: number_of_seg-1
            #  Get the end point and the sensitivity function for \phi(t, x_0)
            [sen_mat, x] = sen_init(seg_init(:,k), seg_len(k),...
                                    ode_A, FINITE_DIFFERENCE, ODE_LIST)
            #  get the Jacobian of the rhs of ODE's
            [J] = numderivative(list(ode_rhs, ode_A, ODE_LIST), x)
            
            #  stack mixed second derivatives (space-time)
            V_stack = -(J*sen_mat)';
            
            #  stack second derivatives with respect to time
            fx = ode_rhs(x, ode_A, ODE_LIST)
            alpha_stack = -fx'*J';
            
            #  perform the index contraction by lambda
            v_i = V_stack*lambda((k-1)*statespace_dim+1:k*statespace_dim);
            alpha_i = alpha_stack*lambda((k-1)*statespace_dim+1:k*statespace_dim);
            
            #  Get the k-th block
            if k ==1 then
                #  get s, y and D_i
                D_k = D_old((k-1)*statespace_dim+1:k*statespace_dim,:);
                [D] = mat_H(D_k, s(:,k), y(:,k));
                D_new = [D_new; D];
#                H_block = [D, v_i; v_i', alpha_i + 1];
#                #  I can be clever
                H_block = [lam_I*ell_I, v_i; v_i', alpha_i + 1];
            else
                #  get s, y and D_i
                D_k = D_old((k-1)*statespace_dim+1:k*statespace_dim,:);
                [D] = mat_H(D_k, s(:,k), y(:,k));
                D_new = [D_new; D];
                #  form H_block
                H_block = [D, v_i; v_i', alpha_i + 1];
            end
            #  Add it
            Hess_new = sysdiag(Hess_new, H_block);
        end
        
        #  The last block
        #  Get the end point and the sensitivity function for \phi(t, x_0)
        [sen_mat, x] = sen_init(seg_init(:,$), seg_len($),...
                                    ode_A, FINITE_DIFFERENCE, ODE_LIST)
                                    
        #  get the Jacobian of the rhs of ODE's
        [J] = numderivative(list(ode_rhs, ode_A, ODE_LIST), x)
        
        #  get data in matrix
        fx = ode_rhs(x, ode_A, ODE_LIST)
        v_i = (J*sen_mat)'*ell_U*(x - cen_U) + sen_mat'*ell_U*fx;
        alpha_i = fx'*ell_U*fx + (J*fx)'*ell_U*(x - cen_U);
        
        #  Get D
        D_k = D_old((number_of_seg-1)*statespace_dim+1:number_of_seg*statespace_dim,:);
        [D] = mat_H(D_k, s(:,$), y(:,$));
        D_new = [D_new; D];
        H_block = [D, lam_U*v_i; lam_U*v_i', lam_U*alpha_i]
        
#        #  Clever updates
#        H_block = [D + lam_U*sen_mat'*ell_U*sen_mat, lam_U*v_i; lam_U*v_i', lam_U*alpha_i]

        H_block($,$) = H_block($,$) + 1;
        Hess_new = sysdiag(Hess_new, H_block);
endfunction 