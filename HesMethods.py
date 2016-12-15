import numpy as np
def mat_H(matrix_Hessian, s, y, Hessian_approximation = "BFGS"):
    #  Function approximates the Hessian of the Lagrangian by BFGS method
    #  http:#en.wikipedia.org/wiki/BFGS_method
    
    #  INPUT:
    #      matrix_Hessian - Hessian matrix from the previous iteration
    #      s - a direction such that s = x_new - x_old
    #      y = gxL(x_new, lam_new) - gxL(x_old, lam_new)
    
    #  OUTPUT:
    #      matrix_Hessian_next - a new Hessian approximation
    
    #=========================================================================

    
    #global Hessian_approximation #todo wut?
    
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
        if (abs(np.vdot(s,Hs) < 1e-8*np.norm(s)*np.norm(y - Hs))): #was abs(s'*(y - matrix_Hessian*s)) < 1e-8*norm(s)*norm(y - matrix_Hessian*s) then
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
    n = A.shape()[0]
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
    number_of_blocks = blocks_OLD.shape()[0]/blocks_OLD.shape()[1]
    size_of_blocks = blocks_OLD.shape()[1]
    
    #  We split vectors s and y = lag_grad_next-lag_grad into number_of_segments vectors, for we have number_of_segments blocks
    #if (PROBLEM_FORMULATION == 2):
    #    #  Lengths are fixed => we have only Nn parameters
    #    S = matrix(s_v, statespace_dimension, number_of_segments)
    #    Y = matrix(y_v, statespace_dimension, number_of_segments)
    #else:
    #    #  Lengths are not fixed => we have N(n+1) parameters
    #S = matrix(s_v, statespace_dimension+1, number_of_segments)
    #Y = matrix(y_v, statespace_dimension+1, number_of_segments)
    S = np.empty( statespace_dimension+1, number_of_segments) #matrix(s_v,)
    Y = np.empty( statespace_dimension+1, number_of_segments) #matrix(y_v, statespace_dimension+1, number_of_segments)
    S[:,:] = np.reshape(s_v, statespace_dimension+1, number_of_segments)
    Y[:, :] = np.reshape(y_v, statespace_dimension + 1, number_of_segments)
    

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
            Hb = blocks_OLD[size_of_blocks * i:(i + 1) * size_of_blocks - 1,:]  #bylo #Hb = blocks_OLD(size_of_blocks*(i-1)+1:i*size_of_blocks, :) #todo
            
            #  Block-wise BFGS
            [Hb_update] = mat_H(Hb, S[:,i], Y[:,i])
            blocks_NEW = [blocks_NEW, Hb_update]
            #matrix_Hessian_new = sysdiag(matrix_Hessian_new, Hb_update)
            matrix_Hessian = linalg.block_diag([matrix_Hessian_new, Hb_update])
        

    elif (number_of_blocks == number_of_segments-1):
        #  data
        ZB_SMALL = np.zeros(size_of_blocks, size_of_blocks)

        #  This is the banded Hessian with width of the band 2*n or 2*(n+1)
        for i in range(number_of_blocks): #= 1:number_of_blocks
            #  Extract one block
            Hb = blocks_OLD[size_of_blocks * i : (i+1) * size_of_blocks-1,:] #Hb = blocks_OLD(size_of_blocks*(i-1) + 1: i*size_of_blocks, :)
            
            #  Block-wise BFGS (overlapping blocks
            #[Hb_update] = mat_H(Hb, [[S[:, i]; S[:, i+1]], [Y[:, i], Y[:, i+1]]]) #concat: , as rows, ; as columns
            newmat =  [[S[:, i], S[:, i + 1]], [Y[:, i], Y[:, i + 1]]]
            #throw!!
            Hb_update = mat_H(Hb, newmat)
            blocks_NEW = [blocks_NEW, Hb_update]
            ZB_BIG = np.zeros(matrix_Hessian_new.shape[0] - size_of_blocks/2, matrix_Hessian_new.shape[0] - size_of_blocks/2)
            #matrix_Hessian_new = sysdiag(matrix_Hessian_new, ZB_SMALL) + sysdiag(ZB_BIG, Hb_update)
            matrix_Hessian_new = linalg.block_diag([matrix_Hessian_new, ZB_SMALL]) + linalg.block_diag([ZB_BIG, Hb_update])
            ZB_SMALL = np.zeros(size_of_blocks/2, size_of_blocks/2)
        
    
    return [matrix_Hessian_new, blocks_NEW]

def my_Hess_linode(lam, seg_init, seg_len, ode_A, cen_U, ell_I, ell_U, my_ode):
        #  This function approximates the Hessian of the Lagrangian using
        #  the structure and information I have at hand. I try to avoid
        #  using BFGS or any other for the Hessian approximation
        
        #  OUTPUT: a block-diagonal SPD Hessian approximation
        
        
        #  initialize
        statespace_dim = seg_init.shape()[0]
        number_of_seg = seg_init.shape()[1]
        AT_sqr = (ode_A*ode_A)
        Hess_new = []
        lam_I = lam[0]
        lam_U = lam[-1]
        xlambda = lam
        xlambda[0] = []
        xlambda[-1] = []
        D = np.zeros(statespace_dim, statespace_dim)
                
        #  iterate over blocks 1...number_of_seg
        #  the last update at the end
        for k in xrange(number_of_seg-2): #k=1:numberofseg-1
            #  get e**{At_i}**T
            sen_mat = scipy.linalg.expm(ode_A*seg_len[k])
            #  stack mixed second derivatives (space-time)
            V_stack = -np.dot(ode_A,sen_mat).T #(ode_A*sen_mat)';                  #np.dot as matrix multiplication
            #  stack second derivatives with respect to time
#            alpha_stack = -seg_init(:,k)'*sen_mat*AT_sqr; --- # BAD
            alpha_stack = -np.dot(np.dot(AT_sqr,sen_mat),seg_init[:,k]) #-AT_sqr*sen_mat*seg_init(:,k);
            #  perform the index contraction by lambda
            v_i = V_stack*xlambda[k*statespace_dim:(k+1)*statespace_dim-1]
            alpha_i = np.dot(alpha_stack.T,xlambda[k*statespace_dim:(k+1)*statespace_dim-1]) #alpha_stack'*lambda((k-1)*statespace_dim+1:k*statespace_dim);
            #  build the block of the Hessian matrix
            if k == 1:
                H_block = [[lam_I*ell_I, v_i], [v_i.T, alpha_i + 1]]#[lam_I*ell_I, v_i; v_i.T, alpha_i + 1]
            else:
                H_block = [[D, v_i], [v_i.T, alpha_i + 1]]#[D, v_i; v_i.T, alpha_i + 1]
            
            #  add this block to previously computed ones
            Hess_new = linalg.block_diag([Hess_new, H_block])#sysdiag(Hess_new, H_block);
        
        #  the last block
        eAt = scipy.linalg.expm(np.dot(ode_A,seg_len[-1]))
        eAtx = eAt*seg_init[:,-1]
        AeAt = ode_A*eAt
        AeAtx = AeAt*seg_init[:,-1]
        eAtxc = eAtx - cen_U
        AAeAtx = ode_A*AeAtx
        #  create blocks in the last block
        M = eAt.T*ell_U*eAt#eAt'*ell_U*eAt;
        v_i = AeAt.T*ell_U*eAtxc + eAt.T*ell_U*AeAtx #AeAt'*ell_U*eAtxc + eAt'*ell_U*AeAtx;
        alpha_i = AAeAtx.T*ell_U*eAtxc + AeAtx.T*ell_U*AeAtx #AAeAtx'*ell_U*eAtxc + AeAtx'*ell_U*AeAtx;
        H_block = lam_U*[[M, v_i], v_i.T, alpha_i] #lam_U*[M, v_i; v_i', alpha_i]
        H_block[-1,-1] = H_block[-1,-1] + 1
        Hess_new = linalg.block_diag([Hess_new, H_block])
        return Hess_new


def my_Hess_nonlinode(lam, seg_init, seg_len, ode_A, D_old,
                                    s, y, cen_U, ell_I, ell_U, my_ode, FINITE_DIFFERENCE):
  #  This function approximates the Hessian of the Lagrangian using
  #  the structure and information I have at hand. I try to avoid
  #  using BFGS or any other for the Hessian approximation
  
  #  OUTPUT: a block-diagonal SPD Hessian approximation
  
  
  #  initialize
  statespace_dim=seg_init.shape()[0]
  number_of_seg=seg_init.shape()[1]
  Hess_new = []
  lam_I = lam[1]
  lam_U = lam[-1]
  xlambda = lam
  xlambda[1] = []
  xlambda[-1] = []
  D_new = []
  
  #  remove time and keep only state variables in s, y
  s = matrix(s,statespace_dim+1, number_of_seg) #todo
  s[-1,:] = []
  y = matrix(y,statespace_dim+1, number_of_seg) #todo
  y[-1,:] = []
  
  #  I will get 1st - (N-1)st block in the Hessian
  for k in range(number_of_seg-2) #k = 1: number_of_seg-1
      #  Get the end point and the sensitivity function for \phi(t, x_0)
      [sen_mat, x] = sen_init(seg_init[:,k], seg_len(k),
                              ode_A, FINITE_DIFFERENCE)
      #  get the Jacobian of the rhs of ODE's
      [J] = numderivative(list(ode_rhs, ode_A), x) #todo
      
      #  stack mixed second derivatives (space-time)
      V_stack = -np.dot(J,sen_mat).T #-(J*sen_mat)';
      
      #  stack second derivatives with respect to time
      fx = ode_rhs(x, ode_A)
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
                              ode_A, FINITE_DIFFERENCE)
                              
  #  get the Jacobian of the rhs of ODE's
  [J] = numderivative(list(ode_rhs, ode_A), x)
  
  #  get data in matrix
  fx = ode_rhs(x, ode_A)
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
  return [Hess_new, D_new]

function fx = ode_rhs(x, ODE_MATRIX_A)
    //  Function gives the rhs of \dot{x} = f(t, x)
    
    //  INPUT:
    //      x - state variable, i.e. x(t, x0)
    //      ODE_MATRIX_A - constant matrix for the lienar part of f(t,x)
    
    //  OUTPUT:
    //      fx - f(t, x(t,x0))
    
    //=========================================================================
    
    //  Choose what kind of dynamics you need
    select ODE_LIST
    case 1 then
        fx = ODE_MATRIX_A*x;
    case 2 then
        fx = sin(x);
    case 3 then
        fx = sin(x)+cos(x);
    case 4 then
        fx = sin(x($:-1:1,:))+cos(x);
    case 5 then
        fx = 2*(sin(x($:-1:1,:))+cos(x));
    case 6 then
        fx = sin(x) + 2*(sin(x($:-1:1,:))+cos(x));
    case 7 then
        fx = ODE_MATRIX_A*x + sin(x($:-1:1,:));
    case 8 then
        //  Lorenz system http://en.wikipedia.org/wiki/Lorenz_system
        //  For the following choice of parameters sigma, ro and beta, the system
        //  has chaotic solutions
        
        //  parameters
        sig = 10;
        ro = 28;
        bet = 8/3;
        
        fx = [sig*(x(2,:) - x(1,:))
        x(1,:).*(ro - x(3,:)) - x(2,:)
        x(1,:).*x(2,:) - bet*x(3,:)];
    case 9 then
        // Rossler attractor http://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
        //  For the following choice of parameters a, b and c, the system
        //  has chaotic solutions
        
        //  parameters
        a = 0.2;
        b = 0.2;
        c = 13;
        
        fx = [-x(1,:) - x(3,:)
        x(1,:) + a*x(2,:)
        b + x(3,:).*(x(1,:) - c)];
    case 10 then
        //  Competitive Lotka-Volterra equations 4-dimensional example
        //  http://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations
        //  x - size of the population, r - growth rate, K - carrying capacity
        //  M - effects of species on each other a_i,j effect of j-th species
        //  on the i-th species
        
        //  parameters - gives a chaotic system
        r = [1
        0.72
        1.53
        1.27];
        
        M = [1 1.09 1.52 0
        0 1 0.44 1.36
        2.33 0 1 0.47
        1.21 0.51 0.35 1];
        
        K = 1;
        
        //  ode
        for i = 1:4
            temp = 0;
            for j = 1:4
                temp = temp + M(i,j)*x(j,:);
            end
            fx(i,:) = r(i)*x(i,:).*(1 - temp/K);
        end
    case 11 then
        //  Tumor-imune dynamics - Kirschner model
        //  https://www.math.unl.edu/~bdeng1/Teaching/math943/Topics/Cancer%20Modeling/References/Kirshner98.pdf
        
        //  x(1) - activated imune-system cells
        //  x(2) - tumor cells
        //  x(3) - concentration of IL-2
        
        //  paramaters
        c = 0.05;
        mu2 = 0.03;
        p1 = 0.1245;
        g1 = 2e7;
        g2 = 1e5;
        r2 = 0.18;
        b = 1e-9;
        a = 1;
        mu3 = 10;
        p2 = 5;
        g3 = 1e3;
        s1 = 1;
        s2 = 1;
        
        //  ode
        fx(1,:) = c*x(2,:) - mu2*x(1,:) + (p1*x(1,:).*x(3,:))./(g1 + x(3,:)) + s1;
        fx(2,:) = r2*(1-b*x(2,:)).*x(2,:) - (a*x(1,:).*x(2,:))./(g2 + x(2,:));
        fx(3,:) = (p2*x(1,:).*x(2,:))./(g3 + x(2,:)) - mu3*x(3,:) + s2;
    case 12 then
        //  Predator and top predator
        //  population size
        p = 8;
        //  birth rate of rabits
        r = 5;
        //  searching efficiencies
        a = [4;2];
        //  conversion efficiency
        c = [1;0.3];
        //  handling times
        h = [3;8];
        //  mortality rates
        m = [0.2; 0.05];
        
        fx = [r*x(1,:).*(1-x(1,:)/p) - (a(1)*x(1,:).*x(2,:))./(1 + a(1)*h(1)*x(1,:))
        c(1)*(a(1)*x(1,:).*x(2,:))./(1 + a(1)*h(1)*x(1,:)) - m(1)*x(2,:) - (a(2)*x(2,:).*x(3,:))./(1 + a(2)*h(2)*x(2,:))
        c(2)*(a(2)*x(2,:).*x(3,:))./(1 + a(2)*h(2)*x(2,:)) - m(2)*x(3,:)];
    case 13 then
        //  Nonlinear Systems by Khalil, p. 378, 9.25
        fx = [x(2,:) + x(2,:).*x(3,:).^3
        -x(1,:) - x(2,:) - x(1,:).^2
        x(1,:) + x(2,:) - x(3,:).^3];
    case 14 then
        //  Nonlinear Systems by Khalil, p. 187, 4.32/1
        fx = [-x(1,:) + x(1,:).^2
        -x(2,:) + x(3,:).^2
        x(3,:) - x(1,:).^2];
    case 15 then
        //  Nonlinear Systems by Khalil, p. 187, 4.32/4
        fx = [-x(1,:)
        -x(1,:) - x(2,:) - x(3,:) - x(1,:).*x(3,:)
        (x(1,:) + 1).*x(2,:)];
    case 16 then
        //  Nonlinear Systems by Khalil, p. 334, 8.5/3
        fx = [-x(2,:) + x(1,:).*x(3,:)
        x(1,:) + x(2,:).*x(3,:)
        -x(3,:) - (x(1,:).^2 + x(2,:).^2) + x(3,:).^2];
    case 17 then
        l = size(ODE_MATRIX_A, 1);
        fx = ODE_MATRIX_A*x + sin(x($:-1:1,:)) + 2*cos([x(l:-1:1,:); x($:-1:l+1,:)]);
    case 18 then
        //  Chemical reaction of Robertson from
        //  https://en.wikipedia.org/wiki/Stiff_equation
        fx =[-0.04*x(1,:) + 1e4*x(2,:).*x(3,:)
        0.04*x(1,:) - 1e-4*x(2,:).*x(3,:) - 3e-7*x(2,:).^2
        3e7*x(2,:).^2];
    case 19 then
        //  The Navigational Benchmark with 9 location
        //  --- a nonlinear version of the benchmark
        //  The position in 3x3 grid is given in a vector ODE_MATRIX_A = [i;j]
        
        //  The dynamics is fixed in data A, B and the grid C!!!!!!!
        //  \dot{x} = Ax - Bu - nonlin term
        C = [1 3 3
        7 5 2
        6 4 4];
        //  get the position in the grid where to evaluate f(x)
        width = 2;
        pos_in_grid = ceil(x(1:2)/width);
        //  get the orientation for the flow
        piCq = %pi*C(pos_in_grid(1), pos_in_grid(2))/4;
        //  get the input from the flow in the grid
        u = [sin(piCq); cos(piCq)];
        //  the nonlin term
        nlt = [zeros(2, length(x(1,:))); 0.1*x(3:$,:).^2];
        //  The ODE
       A = [0 0 1 0
       0 0 0 1
       0 0 -1.2 0.1
       0 0 0.1 -1.2];
       B = [0 0
       0 0
       -1.2 0.1
       0.1 -1.2];
       fx = A*x - nlt - repmat(B*u, 1, length(x(1,:)));
   case 20 then
       //  The Navigational Benchmark with 9 location
        //  --- a nonlinear version of the benchmark
        //  The position in 3x3 grid is given in a vector ODE_MATRIX_A = [i;j]
        
        //  The dynamics is fixed in data A, B and the grid C!!!!!!!
        //  \dot{x} = Ax - Bu - nonlin term
        C = [1 3 3
        7 5 2
        6 4 4];
        //  get the position in the grid where to evaluate f(x)
        pos_in_grid = ceil(x(1:2)/2);
        //  get the orientation for the flow
        piCq = %pi*C(pos_in_grid(1), pos_in_grid(2))/4;
        //  get the input from the flow in the grid
        u = [sin(piCq); cos(piCq)];
        //  The ODE
       A = [0 0 1 0
       0 0 0 1
       0 0 -1.2 0.1
       0 0 0.1 -1.2];
       B = [0 0
       0 0
       -1.2 0.1
       0.1 -1.2];
       fx = A*x - repmat(B*u, 1, length(x(1,:)));
    else
        disp("You need to select ODEs for your system. Error in ode_rhs.sci")
    end
endfunction

"""
function [sensitivity_function, endstate_traj] = sen_init(INITIAL_STATE, LENGTH_TRAJ, ode_matrix_A, FINITE_DIFFERENCE_SCHEME)
    //  Function computes sensitivity of the final state of a trajectory to
    //  the change of its initial state, i.e. it computes the first variational
    //  equation.
    
    //  INPUT:
    //      INITIAL_STATE - initial state of a trajectory
    //      LENGTH_TRAJ - the length of a trajectory
    //      ode_matrix_A - a matrix that defines linear dynamics of the dynamical system
    //      FINITE_DIFFERENCE_SCHEME - a switch between forward and central difference for numerical
    //          differentiantion; FINITE_DIFFERENCE_SCHEME = 1/2 = forward/central
    
    //  OUTPUT:
    //      sensitivity_function - sensitivity function
    //      endstate_traj - final state of a trajectory x(LENGTH_TRAJ; xO), where \dot{x} = f(t, x)
    
    //=========================================================================
    
    
    //  statespace_dimension - state space dimension
    statespace_dimension = length(INITIAL_STATE);
    
    //  delta for finite difference computation
    delta = norm(INITIAL_STATE)*sqrt(%eps);
    
    //  initialization
    x0_stack = repmat(INITIAL_STATE, 1, statespace_dimension);
    sensitivity_function = [];
    
    //  Select ODE
//    my_ode = list(ode_lin, ode_matrix_A);
    
    select FINITE_DIFFERENCE_SCHEME
    case 1 then
        //  forward difference scheme
        ic = [INITIAL_STATE, x0_stack + delta*eye(statespace_dimension, statespace_dimension)];
        
        //  update global variables
//        ode_calls = ode_calls + [0; 1]
//        ode_total_length = ode_total_length + LENGTH_TRAJ;
        
        [X, flag] = ode_simul(ic, LENGTH_TRAJ, ode_matrix_A, 0)
        for i = 1:statespace_dimension
            sensitivity_function = [sensitivity_function, (X(:,i+1) - X(:,1))/delta];
        end
    case 2 then
        //  delta for the entral difference scheme
        delta = norm(INITIAL_STATE)*%eps^(1/3);
        
        //  central difference scheme
        ic = [INITIAL_STATE, x0_stack + delta*eye(statespace_dimension, statespace_dimension), x0_stack - delta*eye(statespace_dimension, statespace_dimension)];
        
//        ode_calls = ode_calls + [0; 1]
//        ode_total_length = ode_total_length + LENGTH_TRAJ;
        
//        [X, flag] = ode_simul(ic, LENGTH_TRAJ, ode_matrix_A, 0)
        
        //  The part for HDS testing
        //====================================================================
        X = [];
        for i = 1:size(ic, 2)
            [sol, flag] = ode_simul(ic(:,i), LENGTH_TRAJ, ode_matrix_A, 0);
            X = [X sol];
        end
        //====================================================================
        
        for i = 1:statespace_dimension
            sensitivity_function = [sensitivity_function, (X(:,i+1) - X(:,statespace_dimension+1+i))/(2*delta)]
        end
    case 3 then
        //  This is an analytic approach. We solve coupled ODEs to get the sensitivity function
        ic = [INITIAL_STATE; matrix(eye(statespace_dimension, statespace_dimension),...
        statespace_dimension^2, 1)];
        //  Select ODEs with sensitivities
        [X, flag] = ode_simul(ic, LENGTH_TRAJ, ode_matrix_A, 1);
        //  Sensitivity function
        sensitivity_function = matrix(X(statespace_dimension+1:$),statespace_dimension,...
        statespace_dimension);
        X(statespace_dimension+1:$) = [];
    else
        disp("Error in sen_init.sci. Select FINITE_DIFFERENCE_SCHEME = 1/2");
        break
    end
    
    //  The final state of a trajectory
    endstate_traj = X(:,1);
endfunction
"""