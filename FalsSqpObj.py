import numpy as np
import HesMethods as hmat
#todo [] zavorky jsou nutne vsude?
#todo - new const = sqrt(epsconst)
#todo 0 based idexing (-1 is already ok)
epsconst = 2**(-52)

class FalsSqpObj:
  #def _init_(self):
  def get_fcl(self,INITIAL_STATES_SEGMENTS,LENGTHS_SEGMENTS, ell_I, ell_U, cen_IU, ode_matrix_A,
              FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST,xlambda)
    
    # Function evaluates the objective function fx, its gradient gfx,
    # vector of constraints cx, its Jacobian Bx
    # Lagrangian L and its gradient with respect to x gxL
    
    # objective function part
    [fx, gfx] = object_fx(INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, ell_I, ell_U, cen_IU, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST)
    
    # vector of constraints part
    [cx, Bx, T] = const_cx(INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST, ell_I, ell_U, cen_IU)
    
    # Lagrangian part
    [L, gxL] = lagrangian(fx, gfx, cx, Bx, xlambda)
    
    return [fx, gfx, cx, Bx, L, gxL]
  
  def gmerit_fcn(self,direction_x, obj_fun_grad, const_val, const_Jac, xlambda, MERIT_FUNCTION_SIGMA):
    #  Function computes the gradient of a merit function for the step length
    #  selection process
    #  m = F(x) + lambda^T*const_val(x) + (MERIT_FUNCTION_SIGMA/2)*||const_val(x)||_2^2
    #  gxm = direction_x^T*(gxF(x) + const_Jac(x)*lambda) + MERIT_FUNCTION_SIGMA*direction_x^T*const_Jac(x)*const_val(x)
    
    #  Source: http:#www.cs.cas.cz/luksan/luksan/saddle.ps
    
    #  INPUT:
    #      direction_x - direction vector such that x_new = x_old + a_k*direction_x
    #      obj_fun_grad - gradient of the objective function gxF(x)
    #      const_val - vector of constraints const_val(x)
    #      const_Jac - Jacobian of the constraints const_Jac(x)
    #      lambda - vector of Lagrange multipliers
    #      MERIT_FUNCTION_SIGMA - a parameter enforcing const_val(x) = 0
    
    #  OUTPUT:
    #  merit_fun_grad - gradient of the merit function
    
    #=========================================================================
    
    
    #  lambda = lam_o + d_l, x = x_o!!!!!!!!!!!!!!!!!!!!!!!!
    # orig: merit_fun_grad = direction_x'*(obj_fun_grad + const_Jac*lambda) + MERIT_FUNCTION_SIGMA*direction_x'*const_Jac*const_val #todo transpose
    merit_fun_grad = direction_x*(obj_fun_grad + const_Jac*xlambda) + MERIT_FUNCTION_SIGMA*direction_x*const_Jac*const_val #todo transpose like orig:
    return [merit_fun_grad]
  
  def merit_fcn(self,obj_fun_val, const_val, xlambda, mer_fun_sigma):
    #  Function evaluates the merit function for given x and lambda
    #  mer_fun_val = F(x) + xlambda^T*const_val(x) + (mer_fun_sigma/2)*||const_val(x)||_2^2
    
    #  INPUT:
    #      obj_fun_val - value of the objective function F(x)
    #      const_val - vector of constraints const_val(x)
    #      xlambda - vector of Lagrange multipliers
    #      mer_fun_sigma - a parameter enforcing const_val(x) = 0
    
    #  OUTPUT:
    #      mer_fun_val - value of the merit function
    
    #=========================================================================
    
    
    #  Evaluate mer_fun_val(x, lambda)
    #orig mer_fun_val = obj_fun_val + xlambda'*const_val + 0.5*mer_fun_sigma*norm(const_val)^2;
    mer_fun_val = obj_fun_val + xlambda*const_val + 0.5*mer_fun_sigma*np.norm(const_val)^2 # todo transposition by orig...
    return [mer_fun_val]
  
  def xt_from_s(self,INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, s, PROBLEM_FORMULATION):
    #  Function computes x_new = x_old + a_k*d_x = x_old + s
    
    #  INPUT:
    #      INITIAL_STATES_SEGMENTS - old initial states of segments, INITIAL_STATES_SEGMENTS(:,i) is the initial
    #              state of the i-th segment
    #      LENGTHS_SEGMENTS - old lengths of segments, LENGTHS_SEGMENTS(i) is the length of the 
    #              i-th segment
    #      s - a direction such that s = x_new - x_old
    #      #      PROBLEM_FORMULATION - a switch between various formulation of the objective function f(x)
    #          and the vectors of constraints c(x)
    
    #  OUTPUT:
    #      si_o - new initial states corresponding to x_new
    #      sl_o - new lengths of segements corresponding to x_new
    
    #=========================================================================
    
    #  statespace_dimension - state space dimension, number_of_segments - number of segments
    [statespace_dimension, number_of_segments] = size(INITIAL_STATES_SEGMENTS)
    
    if (PROBLEM_FORMULATION== 2):
            x_n = matrix(s, statespace_dimension, number_of_segments)
            si_o = INITIAL_STATES_SEGMENTS + x_n;
            sl_o = LENGTHS_SEGMENTS;
    else:
            xat = matrix(s, statespace_dimension+1, number_of_segments)
            si_o = INITIAL_STATES_SEGMENTS + xat[1:statespace_dimension,:]
            sl_o = LENGTHS_SEGMENTS + xat[-1,:];
    return [si_o, sl_o]
  
  def step(self,INITIAL_STATES_SEGMENTS, direction_x, xlambda, direction_lambda, LENGTHS_SEGMENTS, ell_I, ell_U, cen_IU, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, TO_STOP, PROBLEM_FORMULATION, ODE_LIST, MERIT_FUNCTION_SIGMA, mer_fun_val, mer_fun_grad):
    #  Function computes the length of the step size_step so that x_new = x_old + size_step*direction_x
    #  The merit function we use is: 
    #  number_of_segments(size_step) = F(x+size_step*direction_x) + (lambda + direction_lambda)^T*c(x + size_step*direction_x) + (sigma/2)*||c(x + size_step*direction_x)||_2^2
    
    #  INPUT:
    #      INITIAL_STATES_SEGMENTS - initial states of segments stored column-wise; that is
    #                 INITIAL_STATES_SEGMENTS(:,i) is the initial state of the i-th segment
    #      direction_x - is the direction so that x_new = x_old + size_step*direction_x
    #      lambda - vector of Lagrange multipliers
    #      direction_lambda - is the direction so that lam_new = lam_old + size_step*direction_lambda
    #      LENGTHS_SEGMENTS - length of segments; sl_o(i) is the length of the i-th 
    #                  segment
    #      ell_I - a SPD matrix giving the shape of the ellipsoid of the
    #              initial states I
    #      ell_U - a SPD matrix giving shape of the ellipsoid of the unsafe
    #              states U
    #      cen_IU - centres of ellipsoids I and U; stored solumn-wise
    #      ode_matrix_A - a matrix that defines linear dynamics of the dynamical system
    #      FINITE_DIFFERENCE_SCHEME - a switch between forward and central difference for numerical 
    #          differentiantion; FINITE_DIFFERENCE_SCHEME = 1/2 = forward/central
    #      TO_STOP - when to stop, TO_STOP = [stop_lag;stop_c;stop_it;stop_step]
    #      PROBLEM_FORMULATION - a switch between various formulation of the objective function f(x)
    #          and the vectors of constraints c(x)
    #      ODE_LIST - a switch between dynamical systems; for testing
    #      MERIT_FUNCTION_SIGMA - a paramater forcing the constraint c(x) to be satisfied
    
    #  OUTPUT:    
    #      size_step - the length of a step for x_n = x_o + size_step*direction_x, lam_n = lam_o + size_step*direction_lambda
    
    #==========================================================================
    
    #  statespace_dimension - dimension of the state space, number_of_segments - number of segments
#    [statespace_dimension, number_of_segments] = size(INITIAL_STATES_SEGMENTS);
    
    #  boolean for while loop
    go_on = True
    
    #  initial length
    size_step = 2
    
    #  delta for the contraction of size_step
    delta = 0.5

#  FINISH
#    #  evaluate F(x_old), gxF(x_old), c(x_old), B(x_old)
#    [obj_fun_val, obj_fun_grad] = object_fx(INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, ell_I, ell_U, cen_IU, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST);
#    [const_val] = const_only_cx(INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST, ell_I, ell_U, cen_IU);
#    #  value of the merit function in x_old, lam_n = lambda + direction_lambda
#    [mer_fun_val] = merit_fcn(obj_fun_val, const_val, lambda + direction_lambda, MERIT_FUNCTION_SIGMA);
#    #  gradient of the merit function in lam_n = lam_o + direction_lambda, x = x_old
#    [mer_fun_grad] = gmerit_fcn(direction_x, obj_fun_grad, const_val, const_Jac, lambda + direction_lambda, MERIT_FUNCTION_SIGMA);
    

    #  iterate
    while go_on:
        #  reduce the length of size_step (divide by 2)
        size_step = size_step*delta
        
        #  Displey the step-size
#        disp(size_step, "Step-size:")
        
        #  evaluate merit function in x_new = x_old + size_step*direction_x, lambda + direction_lambda
        s_k = size_step*direction_x
        [new_seg_init, new_seg_length] = self.xt_from_s(INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, s_k, PROBLEM_FORMULATION)
        [obj_fun_val_next] = self.object_only_fx(new_seg_init, new_seg_length, ell_I, ell_U, cen_IU, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST)
        [const_val_next] = self.const_only_cx(new_seg_init, new_seg_length, ode_matrix_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST, ell_I, ell_U, cen_IU)
        [mer_fun_val_next] = self.merit_fcn(obj_fun_val_next, const_val_next, xlambda + direction_lambda, MERIT_FUNCTION_SIGMA)
        
        #  Check if we descend
        if ((mer_fun_val_next - mer_fun_val) < 1e-4*size_step*mer_fun_grad):
            go_on = False
        
        
        #  stop because of the size of the step drops to TO_STOP($)
        if (size_step < TO_STOP[-1]):
            go_on = False
        
    return size_step

  def fals_sqp(self,INITIAL_STATES_SEGMENTS, LENGTHS_SEGMENTS, ell_I, ell_U, cen_IU, ODE_MATRIX_A, TO_STOP, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION = 3, ODE_LIST = None, MERIT_FUNCTION_SIGMA = None):
    #  Function runs the SQP algorithm
    
    #  INPUT:
    #      INITIAL_STATES_SEGMENTS - initial states of segments stored column-wise; that is
    #                 INITIAL_STATES_SEGMENTS(:,i) is the initial state of the i-th segment
    #      LENGTHS_SEGMENTS - length of segments; sl_o(i) is the length of the i-th
    #                  segment
    #      ell_I - a SPD matrix giving the shape of the ellipsoid of the
    #              initial states I
    #      ell_U - a SPD matrix giving shape of the ellipsoid of the unsafe
    #              states U
    #      cen_IU - centres of ellipsoids I and U; stored solumn-wise
    #      ODE_MATRIX_A - a matrix that defines linear dynamics of the dynamical system
    #      FINITE_DIFFERENCE_SCHEME - a switch between forward and central difference for numerical
    #          differentiantion; FINITE_DIFFERENCE_SCHEME = 1/2 = forward/central
    #      PROBLEM_FORMULATION - a switch between various formulation of the objective function f(x)
    #          and the vectors of constraints c(x)
    #      ODE_LIST - a switch between dynamical systems; for testing
    #      MERIT_FUNCTION_SIGMA - a parameter enforcing c(x) = 0
    
    #  OUTPUT:
    #      si_o - optimal initial states of segments stored column-wise; that is
    #                 INITIAL_STATES_SEGMENTS(:,i) is the initial state of the i-th segment
    #      sl_o - optimal lengths of segments; sl_o(i) is the length of the i-th
    #                  segment
    #      number_of_iterations - number of iterations
    #      cKKT - cKKT(i) is the condition number of KKT matrix in the i-th interation
    #      cH - cH(i) is the condition number of the Hessian in the i-th iteration
    #      cHb - cHb(k,i) is the condition number of k-th block of matrix_Hessian in the i-th iteration
    #      vM - value of the merit function number_of_segments(x, xlambda, MERIT_FUNCTION_SIGMA)
    #      nL - nL(i) norm of the Lagrangian
    #      cB - cB(i) condition number of the const_Jac matrix in the i-th iteration
    #      nc - nc(i) is the norm of the constraints in the i-th iteration
    #      ss - ss(i) is the step size in the i-th iteration
    #      of - of(i) is the value of the objective function in the i-th iteration
    #      sis - sis(i) the maximum condition number of the sensitivity function in
    #          the i-th iteration
    #      sl_i - sl_i(:,i) constains lengths of segments in the i-th iteration
    #      si_i - si_i(1:k*statespace_dimension,:) are initial states from the k-th iteration
    #      flag - 1/2/3/4 tells me why we stopped [stop_lag;stop_c;stop_it;stop_step]
    
    #==========================================================================
    
    #  global variables
    Hessian_computation = None
    min_eig_it = None
    #  iterate
    go_on = True
    
    #  the total number of restarts
    tnor = 0
    
    #  counter of iterations
    i = 0
    
    #  statespace_dimension - state space dimension, number_of_segments - number of segments
    [statespace_dimension, number_of_segments] = len(INITIAL_STATES_SEGMENTS)#size(INITIAL_STATES_SEGMENTS);
    
    #  initial Hessian approximation and initial xlambda
    #  Hb = matrix_Hessian => FULL BFGS!!!
    #  Hb = repmat(*, number_of_segments, 1) => block-diagonal Hessian
    #  Hb = repmat(*, number_of_segments - 1, 1) => banded Hessian
    if (PROBLEM_FORMULATION == 1):
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = np.tile(np.eye(statespace_dimension+1, statespace_dimension+1), (number_of_segments, 1))
        xlambda = np.zeros(statespace_dimension*(number_of_segments-1), 1)
    elif (PROBLEM_FORMULATION == 2):
        matrix_Hessian = np.eye(number_of_segments*statespace_dimension, number_of_segments*statespace_dimension)
        Hb_old = np.tile(np.eye(statespace_dimension, statespace_dimension), (number_of_segments, 1))
        xlambda = np.zeros(statespace_dimension*(number_of_segments-1), 1)
    elif (PROBLEM_FORMULATION == 3):
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = np.tile(np.eye(statespace_dimension+1, statespace_dimension+1), (number_of_segments, 1))
        #  I've got two more constraints
        xlambda = np.zeros(statespace_dimension*(number_of_segments-1) + 2, 1)
    elif (PROBLEM_FORMULATION == 4):
        #  Try to control lengths of segments
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = np.tile(np.eye(statespace_dimension+1, statespace_dimension+1), (number_of_segments, 1))
        #  I've got two more constraints
        xlambda = np.zeros(statespace_dimension*(number_of_segments-1), 1)
    elif (PROBLEM_FORMULATION == 5):
        #  No control over the lengths
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = matrix_Hessian
        #  I've got two more constraints
        xlambda = np.zeros(2, 1)
    elif (PROBLEM_FORMULATION == 6):
        #  No control over the lengths
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = matrix_Hessian
        xlambda = []
    elif (PROBLEM_FORMULATION == 7):
        #  This should not work!!!!!!!!!
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = matrix_Hessian
        #  I've got two more constraints
        xlambda = np.zeros(2, 1)
    elif (PROBLEM_FORMULATION == 8):
        #  This should not work!!!!!!!!!
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = matrix_Hessian
        xlambda = []
    elif (PROBLEM_FORMULATION == 9):
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = np.tile(np.eye(2*statespace_dimension+2, 2*statespace_dimension+2), (number_of_segments-1, 1))
        #  I've got two more constraints
        xlambda = np.zeros(2, 1)
    elif (PROBLEM_FORMULATION == 10):
        matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
        Hb_old = matrix_Hessian
        #  I've got two more constraints
        xlambda = np.zeros(2, 1)
    else:
        print "You need to select nd = 1/2/3/4/5/6/7/8/9/10! ERROR in fals_sqp.sci"
        
    
    
    #  initialization
    cKKT = []
    cH = []
    cHb = []
    cZHZ = []
    nL = []
    cB = []
    nc = []
    vM = []
    ss = []
    sis = []
    f_I = []
    f_U = []
    si_o = INITIAL_STATES_SEGMENTS
    sl_o = LENGTHS_SEGMENTS
    #  The chnage in lengths of segments during iterations (column wise)
    sl_i = [sl_o] #todo transpose
    #  Store all the initial states that are computed during the optimization
    si_i = [INITIAL_STATES_SEGMENTS]
    
    while go_on:
        #  add one to a counter
        i = i + 1
        
        #  Implement a restart strategy
        restart = True
        tries = 0
        
        while restart:
            #  Get values of Fx fFx, cx, Bx, L , gxL
            [obj_fun_val, obj_fun_grad, const_val, const_Jac, lag_val, lag_grad] = self.get_fcl(si_o,
              sl_o, ell_I, ell_U, cen_IU, ODE_MATRIX_A,
              FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST, xlambda)
            
            #  rhs for the KKT
            rhs_b = [-lag_grad, -const_val]
            
            #  Solve KKT system
            [direction_x, direction_xlambda, matrix_KKT] = self.solve_kkt(matrix_Hessian, const_Jac,
                                                                    rhs_b, statespace_dimension, number_of_segments)

#            #    Sparse pattern of the KKT matrix matrix_KKT
#            PlotSparse(sparse(matrix_KKT), 'b.');
            
            #  Get gradient of the merit function
            [mer_fun_grad] = self.gmerit_fcn(direction_x, obj_fun_grad, const_val, const_Jac, xlambda, MERIT_FUNCTION_SIGMA)
            
                       
            if (-mer_fun_grad < 1e-5*np.norm(direction_x)*np.norm(lag_grad)):
                #  Restart the Hessian approximation
                if (PROBLEM_FORMULATION==2):
                    matrix_Hessian = np.eye(number_of_segments*statespace_dimension, number_of_segments*statespace_dimension)
                else:
                    matrix_Hessian = np.eye(number_of_segments*(statespace_dimension+1), number_of_segments*(statespace_dimension+1))
                    Hb_old = np.tile(np.eye(statespace_dimension+1, statespace_dimension+1), (number_of_segments, 1))
                
                #  increase the counter for restarts
                tnor = tnor + 1
                
                #  Get new directions
                [direction_x, direction_xlambda, matrix_KKT] = self.solve_kkt(matrix_Hessian, const_Jac,
                                                                        rhs_b, statespace_dimension, number_of_segments)
                
            else:
                restart = False
            
            tries = tries + 1
            
            if tries == 2:
                restart = False
            
#           disp(i, "RESTART in ITER")
        

        #  value of the merit function in x_old, lam_n = xlambda + direction_xlambda
        [mer_fun_val] = self.merit_fcn(obj_fun_val, const_val, xlambda + direction_xlambda, MERIT_FUNCTION_SIGMA)
        #  gradient of the merit function in lam_n = lam_o + direction_xlambda, x = x_old
        [mer_fun_grad] = self.gmerit_fcn(direction_x, obj_fun_grad, const_val, const_Jac, xlambda + direction_xlambda, MERIT_FUNCTION_SIGMA)
#        disp(mer_fun_grad,"Mer fun grad");

        #  compute size_step, the length of a step
        [size_step] = self.step(si_o, direction_x, xlambda, direction_xlambda, sl_o, ell_I, ell_U, cen_IU, ODE_MATRIX_A, FINITE_DIFFERENCE_SCHEME, TO_STOP, PROBLEM_FORMULATION, ODE_LIST, MERIT_FUNCTION_SIGMA, mer_fun_val, mer_fun_grad)
#
#
#        #  Show the step-size
#        disp(size_step,"Step-size:")

        #  Compute s_k = x_new - x_old
        s_k = size_step*direction_x
        
        #  Get the maximum condition number from the sensitivity functions
        #  This is not necessary!!!! It is here just for me ;-)
#        temp = [];
#        for j = 1:number_of_segments-1
#            [x_sen, x_end] = sen_init(si_o(:,j), sl_o(j), ODE_MATRIX_A, FINITE_DIFFERENCE_SCHEME, ODE_LIST);
#            temp = [temp; cond(x_sen)];
#        end
        
        
        #  gradient of L(x_old, lam_new)
        [obj_fun_val, obj_fun_grad, const_val, const_Jac, lag_val, lag_grad] = self.get_fcl(si_o,
                                                                                       sl_o, ell_I, ell_U, cen_IU, ODE_MATRIX_A,
                                                                                       FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST, xlambda + size_step*direction_xlambda)

        #  gradient of L(x_new, lam_new)
        [si_o, sl_o] = self.xt_from_s(si_o, sl_o, s_k, PROBLEM_FORMULATION)
        
        [obj_fun_val_next, obj_fun_grad_next, const_val_next,
         const_Jac_next, lag_val_next, lag_grad_next] = self.get_fcl(si_o,
                                                                sl_o, ell_I, ell_U, cen_IU, ODE_MATRIX_A,
                                                                FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST,
                                                                xlambda + size_step*direction_xlambda)

#        #  data for investigation
#        cKKT = [cKKT; cond(matrix_KKT)];
        cH = [cH, cond(matrix_Hessian)]
#        cB = [cB; cond(const_Jac)];
        

        nc = [nc, np.norm(const_val_next)]
        nL = [nL, np.norm(lag_grad_next)]

#=====================================================================================
        if ( Hessian_computation == "FullApprox"):
            #  Full Hessian approximation by (BFGS, DBFGS, SR1)
            [matrix_Hessian] = hmat.mat_H(matrix_Hessian, s_k, lag_grad_next - lag_grad)
        elif ( Hessian_computation == "FullApproxGM"):
            #  Full Hessian approximation by (BFGS, DBFGS, SR1)
            [matrix_Hessian] = hmat.mat_H(matrix_Hessian, s_k, lag_grad_next - lag_grad)
            [R, E] = hmat.my_gillmurr(matrix_Hessian, np.sqrt(epsconst))
            matrix_Hessian = matrix_Hessian + E

        elif (Hessian_computation == "BlockApprox"):
            #  Block Hessian approximation by (BFGS, DBFGS, SR1)
            [matrix_Hessian, Hb_new] = mat_H_block(matrix_Hessian, Hb_old, s_k,
                                                   lag_grad_next - lag_grad, statespace_dimension,
                                                   number_of_segments, PROBLEM_FORMULATION)
            Hb_old = Hb_new
            dim = size(Hb_old, 2)
            min_eig = []
#            #  Study the behaviour of eigenvalues in each block when BFGS/SR1 is used
#            for j = 1:number_of_segments
#                H_block = Hb_old((j-1)*dim +1:j*dim,:);
#                v = real(spec(H_block))
#                min_eig = [min_eig; min(v)];
#            end
#            min_eig_it = [min_eig_it min_eig];
        elif (Hessian_computation == "MyHessLin"):
            #  MY HESSIAN APPROXIMATION SCHEME - analytic formulae
            [matrix_Hessian] = my_Hess_linode(xlambda, si_o, sl_o,
                                                    ODE_MATRIX_A, cen_IU(:,2), ell_I, ell_U, ODE_LIST)
        elif (Hessian_computation == "MyHessLinGM"):
            #  MY HESSIAN APPROXIMATION SCHEME for lin ODE's + G-M update
            [matrix_Hessian] = my_Hess_linode(xlambda, si_o, sl_o,
                                                    ODE_MATRIX_A, cen_IU(:,2), ell_I, ell_U, ODE_LIST)
            [R, E] = my_gillmurr(matrix_Hessian, np.sqrt(epsconst))
            matrix_Hessian = matrix_Hessian + E
        elif (Hessian_computation == "MyHessNonLin"):
            #  MY HESSIAN APPROXIMATION SCHEME - NONLINER ODE. I use exact formulae + BFGS/damped BFGS/SR1
            if (i == 1):
                Hb_old = np.tile(np.eye(statespace_dimension, statespace_dimension), (number_of_segments, 1))
            
            [matrix_Hessian, Hb_new] = my_Hess_nonlinode(xlambda, si_o, sl_o, ODE_MATRIX_A, Hb_old,
                                                               s_k, lag_grad_next - lag_grad, cen_U, ell_I, ell_U,
                                                         ODE_LIST, FINITE_DIFFERENCE_SCHEME)
            Hb_old = Hb_new
        elif (Hessian_computation == "BlockApproxGM"):
            #  Block Hessian approximation by (BFGS, DBFGS, SR1)
            [matrix_Hessian, Hb_new] = mat_H_block(matrix_Hessian, Hb_old, s_k,
                                                   lag_grad_next - lag_grad, statespace_dimension,
                                                   number_of_segments, PROBLEM_FORMULATION)
            Hb_old = Hb_new
            #  Applay Gill-Murray modified Cholesky
            [R, E] = my_gillmurr(matrix_Hessian, np.sqrt(epsconst))
            matrix_Hessian = matrix_Hessian + E
        elif (Hessian_computation == "TrueHess"):
            #  Use builtin function to get the second derivatives
            #  !!!!! Very expensive    !!!!!!!
            funkce = list(lagrangian_complet, statespace_dimension, number_of_segments, #todo list python?
                          xlambda + size_step*direction_xlambda, ell_I, ell_U, cen_IU, ODE_MATRIX_A,
                                 FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST)
            sol = np.matrix([si_o, sl_o], number_of_segments*(statespace_dimension +1), 1)
            [grad_L_opt, matrix_Hessian] = numderivative(funkce, sol, [], [], "blockmat")
        else:
            #  Full Hessian approximation by (BFGS, DBFGS, SR1)
            [matrix_Hessian] = mat_H(matrix_Hessian, s_k, lag_grad_next - lag_grad)
        
        
##      Hessian computation by the finite difference scheme "VERY EXPENSIVE"
#        funkce = list(lagrangian_complet, statespace_dimension, number_of_segments,...
#                 xlambda + size_step*direction_xlambda, ell_I, ell_U, cen_IU, ODE_MATRIX_A, ...
#                FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST);
#        sol = matrix([si_o; sl_o], number_of_segments*(statespace_dimension +1), 1);
#        [grad_L, matrix_Hessian] = numderivative(funkce, sol, [], [], "blockmat");

#        #  Check the structure of the Hessian
#        PlotSparse(sparse(matrix_Hessian),"b.")
#====================================================================================
        
#        ss = [ss; size_step];
#        sis = [sis; max(temp)];
        sl_i = [sl_i,sl_o] #sl_ot #todo transpose
        si_i = [si_i,si_o]
        

        #  Get the null-space of B'(x)
#        Q = kernel(const_Jac_next');
        

        #  Compute the condition numbers for each block of matrix_Hessian
#        [H_cond] = mat_H_block_cond(matrix_Hessian, statespace_dimension, number_of_segments, PROBLEM_FORMULATION);
#        cHb = [cHb H_cond];
#        if Q == [] then
#            cZHZ = [cZHZ; []];
#        else
#            cZHZ = [cZHZ; [cond(Q'*matrix_Hessian*Q), cond(const_Jac_next'*const_Jac_next)]];
##            #  Get all eigenvalues of the projected Hessian
##            disp(spec(Q'*matrix_Hessian*Q), "Eigenvalues of the projected Hessian")
##            #  Get all eigenvalues of the Hessian
##            disp(spec(matrix_Hessian), "Eigenvalues of the Hessian")
#        end
        
        
        
#        #  I will evaluate the merit function and store it
#        [mer_fun_val] = merit_fcn(obj_fun_val_next, const_val_next, xlambda+direction_xlambda, MERIT_FUNCTION_SIGMA)
#        vM = [vM; mer_fun_val];
        
        #  update xlambda; lam_new = lam_old + size_step*direction_xlambda
        xlambda = xlambda + size_step*direction_xlambda


        #  Get the value of the objective function f_I + f_U
        v = si_op[:,1] - cen_IU[:,1]
        f_I = [f_I; v*ell_I*v] #v^T*ell_I*v #todo transpose
#        my_ode = list(ode_lin, ODE_MATRIX_A, ODE_LIST(1));
#        X = ode(si_o(:,$), 0, sl_o($), my_ode);
        [X, flag] = ode_simul(si_o[:,-1], sl_o[-1],ODE_LIST, ODE_MATRIX_A, 0)
        v = X - cen_IU[:,-1]
        f_U = [f_U, v*ell_U*v] #v^T *ell_U*v #todo transpose
        of = [f_I, f_U]
        
         
        #  norm of the lagrangian and vector of constraints
        if (nL[-1] < TO_STOP(1) & nc[-1] < TO_STOP(2)):
            go_on = False
            flag = 1
        
        #  maximum number of iterations is reached
        if  (i == TO_STOP(3)):
            go_on = False
            flag = 2
        
        #  length of a step is too small
        if (size_step <= TO_STOP[-1]):
            go_on = False
            flag = 3
        
        
#        #  Display the iteration number
#        disp(i,"Iteration No.")
#        #  Draw segments
#        dummy = draw_segments(si_o, sl_o, i, [1 2], ODE_LIST,...
#                ODE_MATRIX_A, 0)
        
        
#        if  nc($) < TO_STOP(2) then
#            go_on = false;
#            flag = 4;
#        end
        
        
#        #  I add an extra stopping criterion - GAUK STOP!!!!
#        if nc($) < TO_STOP(2) then
#            go_on = false;
#            flag = -1;
#        end
        
#
#        #  Get the real Hessian and the gradient for the output
#        if go_on == false then
##            #  get nullspace of const_Jac'
#            [QB, RB] = qr(const_Jac_next);
#            Q = QB(:,size(const_Jac_next,2)+1:$);
##            PlotSparse(sparse(Q),"bo")
#            disp(spec(Q'*matrix_Hessian*Q),"test SPD on KerB^T")
#
#            #  Get the Hessian
#            x = [si_o; sl_o];
#            [statespace_dimension, number_of_segments] = size(si_o);
#            lambd = xlambda;
#            x = matrix(x, number_of_segments*(statespace_dimension+1), 1);
#            fce = list(lagrangian_complet, statespace_dimension, number_of_segments, lambd, ell_I, ell_U, cen_IU, ODE_MATRIX_A, FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST);
#            [J_o, H_o] = numderivative(fce, x);
#            H_o = matrix(H_o, (statespace_dimension+1)*number_of_segments, (statespace_dimension+1)*number_of_segments);
#            H_o = Q'*H_o*Q;
#            disp(spec(H_o),"True Hessian on KerB^T", norm(J_o),"Norm of the gradient")
#        end
#
    #end
    lam_o = xlambda
    number_of_iterations = i
    
    #  disp tnor
#    disp(tnor)
    
#    #  Getting the structure of the Hessian
#    Hess = matrix_Hessian
##    [u,v] = find(abs(Hess) < 1e-4);
##    for k = 1:length(u)
##        Hess(u(k), v(k)) = 0;
##    end
#    PlotSparse(sparse(Hess), 'bo');
#
#    #  Project
#    kerBT = kernel(const_Jac_next');
#    Hess_proj = kerBT'*Hess*kerBT;
#    disp(spec(Hess_proj));
#
#    #  pause
#    pause
    
#    disp(spec(matrix_Hessian),"========", cond(matrix_Hessian))
#    #  Check that the true Hessian is SPD if projected on ker(B')
#    kerBT = kernel(const_Jac_next');
#    funkce = list(lagrangian_complet, statespace_dimension, number_of_segments,...
#                 lam_o, ell_I, ell_U, cen_IU, ODE_MATRIX_A, ...
#                FINITE_DIFFERENCE_SCHEME, PROBLEM_FORMULATION, ODE_LIST);
#    sol = matrix([si_o; sl_o], number_of_segments*(statespace_dimension +1), 1);
#    [grad_L, Hess_L] = numderivative(funkce, sol, [], [], "blockmat");
#    Hess_L_proj = kerBT'*Hess_L*kerBT;
##    R = chol(Hess_L_proj);
#    disp(spec(Hess_L_proj))
    return [si_o, sl_o, lam_o, number_of_iterations, cKKT, cH, cHb, vM, nL, nc, ss, of, sis, sl_i, si_i, cB, cZHZ, flag]
#endfunction

  def solve_kkt(self,matrix_Hessian, Bx, b, statespace_dimension, number_of_segments):
    #  Function solves the KKT system, where a matrix matrix_KKT = [matrix_Hessian Bx; Bx^T 0];
    
    #  INPUT:
    #      matrix_Hessian - Hessian matrix (its approximation) of the Lagrangian
    #      lambda - vector of Lagrange multipliers
    #      INITIAL_STATES_SEGMENTS - initial states of segments stored column-wise; that is
    #             INITIAL_STATES_SEGMENTS(:,i) is the initial state of the i-th segment
    #      LENGTHS_SEGMENTS - length of segments; LENGTHS_SEGMENTS(i) is the length of the i-th segment
    #      ell_I - a SPD matrix giving the shape of the ellipsoid of the
    #              initial states I
    #      ell_U - a SPD matrix giving shape of the ellipsoid of the unsafe
    #              states U
    #      cen_IU - centres of ellipsoids I and U; stored solumn-wise
    #      ode_matrix_A - a matrix that defines linear dynamics of the dynamical system
    #      FINITE_DIFFERENCE_SCHEME - a switch between forward and central difference for numerical
    #          differentiantion; FINITE_DIFFERENCE_SCHEME = 1/2 = forward/central
    #      PROBLEM_FORMULATION - a switch between various formulation of the objective function f(x)
    #          and the vectors of constraints c(x)
    #      ODE_LIST - a switch between dynamical systems; for testing
    
    #  OUTPUT:
    #      direction_x - a direction so that x_new = x_old + alpha*d_x
    #      direction_lambda - a direction so that lam_new = lam_old + alpha*d_l
    #      matrix_KKT - KKT matrix of the form matrix_KKT = [matrix_Hessian Bx; Bx^T 0]
    
    #=========================================================================
    
    H_size = matrix_Hessian.size();#todo size
    
    #  Form KKT system matrix_KKT = [matrix_Hessian Bx, Bx^T 0]
    if ( PROBLEM_FORMULATION == 3):#todo transpose [Bx bylo Bx^T
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, zeros(statespace_dimension*(number_of_segments-1)+2, statespace_dimension*(number_of_segments-1)+2)]];
    elif (PROBLEM_FORMULATION == 4):
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, zeros(statespace_dimension*(number_of_segments-1), statespace_dimension*(number_of_segments-1))]];
    elif (PROBLEM_FORMULATION == 5):
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, np.zeros(2, 2)]];
    elif (PROBLEM_FORMULATION == 6):
        matrix_KKT = matrix_Hessian
    elif (PROBLEM_FORMULATION == 7):
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, np.zeros(2, 2)]];
    elif (PROBLEM_FORMULATION == 8):
        matrix_KKT = matrix_Hessian
    elif (PROBLEM_FORMULATION == 9):
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, np.zeros(2, 2)]];
    elif (PROBLEM_FORMULATION == 10):
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, np.zeros(2, 2)]];
    else:
        matrix_KKT = [[matrix_Hessian, Bx], [Bx, np.zeros(statespace_dimension*(number_of_segments-1), statespace_dimension*(number_of_segments-1))]];
    
    
#    m_KKT_bak = matrix_KKT;
    
    #  Get a fix for the conditioning ff the (1,1) block H
#    w = norm(matrix_Hessian)/norm(Bx)^2
#    matrix_KKT(1:H_size(1), 1:H_size(1)) = matrix_Hessian + w*Bx*Bx';
#    b(1:H_size(1)) = b(1:H_size(1)) + w*Bx*b(H_size(1)+1:$);
    
    #  Solve My = b, where y = [direction_x; direction_lambda] by BACKSLASH
#    y = matrix_KKT\b;
    
#    #  Solve the KKT system by the SCHUR_COMPLEMENT METHOD
#    c = -gxL;
#    d = -cx;
#    S = Bx'*inv(matrix_Hessian)*Bx;
#    b = Bx'*inv(matrix_Hessian)*c-d;
#    [y_l, fail, err, iter, res] = conjgrad(S, b);
#    b = c - Bx*y_l;
#    [y_x, fail, err, iter, res] = conjgrad(matrix_Hessian, b);
#    y = [y_x;y_l];

#      #    Solve KKT using the NULL-SPACE METHOD
#     Q = kernel(Bx');
#     c = b(1:size(matrix_Hessian, 1);
#     d = b(size(matrix_Hessian, 1)+1:$);
#     x_P = Bx'\d;
#     A = Q'*matrix_Hessian*Q;
#     b = Q'*(c - matrix_Hessian*x_P)
#     x_N = A\b;
#     y_x = x_P + Q*x_N;
#     A = Bx'*Bx;
#     b = Bx'*(c - matrix_Hessian*y_x)
#     y_l = A\b;
#     y = [y_x;y_l];

#    #  Solve the KKT system by the Null-space Method
#    [x_nsm, y_nsm] = kkt_NullSpaceMethod(matrix_Hessian, Bx, matrix_KKT, -gxL, -cx);
#    y = [x_nsm;y_nsm];
     
#     #    Solve KKT using the PROJECTED PCG with residual update
     [x_o, y_o] = ppcg_lada(matrix_Hessian, Bx, matrix_KKT, b(1:size(matrix_Hessian, 1)), b(size(matrix_Hessian, 1)+1:-1))
     y = [x_o,y_o]
     
#     disp(norm(y-y_PCG), cond(matrix_KKT))

#    matrix_KKT = m_KKT_bak;
    
    #  Get directions direction_x and direction_lambda from the solution vector y
    if (PROBLEM_FORMULATION == 1):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    elif (PROBLEM_FORMULATION == 2):
        direction_x = y(1:number_of_segments*statespace_dimension)
        direction_lambda = y(number_of_segments*statespace_dimension + 1:-1)
    elif (PROBLEM_FORMULATION == 3):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    elif (PROBLEM_FORMULATION == 4):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    elif (PROBLEM_FORMULATION == 5):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    elif (PROBLEM_FORMULATION == 6):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = []
    elif (PROBLEM_FORMULATION == 7):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    elif (PROBLEM_FORMULATION == 8):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = []
    elif (PROBLEM_FORMULATION == 9):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    elif (PROBLEM_FORMULATION == 10):
        direction_x = y(1:number_of_segments*(statespace_dimension+1))
        direction_lambda = y(number_of_segments*(statespace_dimension+1)+1:-1)
    else:
        print "You need to select PROBLEM_FORMULATION = 1/2/3/4/5/6/7/8/9/10! ERROR in solve_kkt.sci"
        
  return [direction_x, direction_lambda, matrix_KKT]



"""
function [dummy] = draw_segments(seg_i, seg_l, num_fig, which_dim, ode_list, ode_A, sen)
    #  This function plots segments in a figure denoted by num_fig
    #  It returns 2D plot, so in which_sim there is marked what components
    #  one intends to draw.
    
    #  INPUT: seg_i - initial states, seg_l - lengths, num_fig - number of the
    #          figure to draw segments in, which_dim - which components to draw
    #          ode_list, ode_A and sen - data needed for simulation!!!
    #  OUTPUT: dummy - whatever I put there ;-)
    
    for j = 1:length(seg_l)
        #  solve ODE
        [X, flag] = ode_simul_show(seg_i(:,j), seg_l(j), ode_list, ode_A, sen);
        scf(num_fig)
        plot2d(X(which_dim(1),:), X(which_dim($),:));
        a = gca();
        a.data_bounds = [0,0;6,6];
        a.box = "on";
        a.grid = [1,1];
        a.sub_ticks = [1,1];
        xax = a.x_ticks;
        xax.locations = [0.0;2.0;4.0;6.0];
        xax.labels = ["0.0";"2.0";"4.0";"6.0"];
        a.x_ticks = xax;
        yay = a.y_ticks;
        yay.locations = [0.0;2.0;4.0;6.0];
        yay.labels = ["0.0";"2.0";"4.0";"6.0"];
        a.y_ticks = yay;
    end
    dummy = "OK"
endfunction
"""