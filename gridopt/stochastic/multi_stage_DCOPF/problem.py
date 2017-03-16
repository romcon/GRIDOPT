#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.    #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

from __future__ import print_function
import csv
import time
import pfnet as pf
import numpy as np
from .utils import ApplyFunc
from numpy.linalg import norm
from gridopt.power_flow import new_method
from optalg.opt_solver.opt_solver_error import *
from optalg.stoch_solver import StochProblemMS
from optalg.opt_solver import OptSolverIQP,QuadProblem
from scipy.sparse import triu,tril,bmat,coo_matrix,eye,block_diag,spdiags

class MS_DCOPF_Problem(StochProblemMS):
    
    # Parameters
    parameters = {'cost_factor' : 1e1,   # factor for determining fast gen cost
                  'infinity'    : 1e4,   # infinity
                  'flow_factor' : 1.0,   # factor for relaxing thermal limits
                  'p_ramp_max'  : 0.01,  # factor for constructing ramping limits for slow gens (fraction of pmax)
                  'r_ramp_max'  : 0.10,  # factor for constructing ramping limits for renewables (fraction of rmax)
                  'r_ramp_freq' : 0.10,  # renewable ramping frequency 
                  'r_eps'       : 1e-3,  # smallest renewable injection
                  'num_samples' : 1000,  # number of samples
                  'draw': False,         # drawing flag
                  'name': ''}            # name

    def __init__(self,net,forecast,parameters={}):
        """
        Class constructor.
        
        Parameters
        ----------
        net : PFNET Network
        forecast : dict
        parameters : dict
        """
        
        # Check forecast
        assert('vargen' in forecast)
        assert('load' in forecast)
        assert('size' in forecast)
        assert(len(forecast['load']) == net.num_loads)
        assert(len(forecast['vargen']) == net.num_vargens)
        assert(set([len(v) for v in list(forecast['load'].values())]) == set([forecast['size']]))
        assert(set([len(v) for v in list(forecast['vargen'].values())]) == set([forecast['size']]))
        
        # Parameters
        self.parameters = MS_DCOPF_Problem.parameters.copy()
        self.set_parameters(parameters)
        
        # Save info
        self.T = forecast['size']
        self.corr_value = net.vargen_corr_value
        self.corr_radius = net.vargen_corr_radius

        # Branch flow limits
        for br in net.branches:
            if br.ratingA == 0.:
                br.ratingA = self.parameters['infinity']
            else:
                br.ratingA *= self.parameters['flow_factor']

        # Initial state
        for load in net.loads:
            load.P = forecast['load'][load.index][0]
        for vargen in net.var_generators:
            vargen.P = forecast['vargen'][vargen.index][0]
        dcopf = new_method('DCOPF')
        dcopf.set_parameters({'quiet': True, 'vargen_curtailment': True})
        dcopf.solve(net)
        assert(dcopf.results['status'] == 'solved')
        dcopf.update_network(net)
                
        # Counters
        num_w = net.num_buses-net.get_num_slack_buses() # voltage angles
        num_p = net.get_num_P_adjust_gens()             # adjustable generators
        num_r = net.num_vargens                         # renewable generators
        num_l = net.num_loads                           # loads
        num_bus = net.num_buses                         # buses
        num_br = net.num_branches                       # branches
        
        # Variables
        net.clear_flags()
        net.set_flags('bus',
                      'variable',
                      'not slack',
                      'voltage angle')
        net.set_flags('generator',
                      'variable',
                      'adjustable active power',
                      'active power')
        net.set_flags('load',
                      'variable',
                      'any',
                      'active power')
        net.set_flags('variable generator',
                      'variable',
                      'any',
                      'active power')

        # Current values
        x = net.get_var_values()
        
        # Projections
        Pw = net.get_var_projection('bus','voltage angle')
        Pp = net.get_var_projection('generator','active power')
        Pl = net.get_var_projection('load','active power')
        Pr = net.get_var_projection('variable generator','active power')
        assert(Pw.shape == (num_w,net.num_vars))
        assert(Pp.shape == (num_p,net.num_vars))
        assert(Pl.shape == (num_l,net.num_vars))
        assert(Pr.shape == (num_r,net.num_vars))

        # Power flow equations
        pf_eq = pf.Constraint('DC power balance',net)
        pf_eq.analyze()
        pf_eq.eval(x)
        A = pf_eq.A.copy()
        b = pf_eq.b.copy()

        # Branch flow limits
        fl_lim = pf.Constraint('DC branch flow limits',net)
        fl_lim.analyze()
        fl_lim.eval(x)
        G = fl_lim.G.copy()
        hl = fl_lim.l.copy()
        hu = fl_lim.u.copy()
        assert(np.all(hl < hu))
        
        # Generation cost
        cost = pf.Function('generation cost',1.,net)
        cost.analyze()
        cost.eval(x)
        H = (cost.Hphi + cost.Hphi.T - triu(cost.Hphi))/net.base_power # symmetric, scaled
        g = cost.gphi/net.base_power - H*x                             # scaled

        # Bounds
        l = net.get_var_values('lower limits')
        u = net.get_var_values('upper limits')
        assert(np.all(Pw*l < Pw*u))
        assert(np.all(Pp*l < Pp*u))
        assert(np.all(Pl*l <= Pl*u))
        assert(np.all(Pr*l < Pr*u))

        # Renewable covariance
        from scikits.sparse.cholmod import cholesky
        r_cov = Pr*net.create_vargen_P_sigma(net.vargen_corr_radius,net.vargen_corr_value)*Pr.T
        r_cov = r_cov.tocoo()
        assert(np.all(r_cov.row >= r_cov.col))
        r_cov = (r_cov+r_cov.T-triu(r_cov)).tocsc()
        factor = cholesky(r_cov)
        L,D = factor.L_D()
        P = factor.P()
        PT = coo_matrix((np.ones(P.size),(P,np.arange(P.size))),shape=D.shape)
        P = P.T
        D = D.tocoo()
        Dh = coo_matrix((np.sqrt(D.data),(D.row,D.col)),shape=D.shape)
        L = PT*L*Dh

        # Problem data
        self.num_p = num_p
        self.num_q = num_p
        self.num_w = num_w
        self.num_s = num_r
        self.num_r = num_r
        self.num_y = num_p
        self.num_z = num_br
        self.num_l = num_l
        self.num_bus = num_bus
        self.num_br = num_br
        self.num_x = self.num_p+self.num_q+self.num_w+self.num_s+self.num_y+self.num_z # stage vars

        self.p_max = Pp*u
        self.p_min = Pp*l
        self.p_prev = Pp*x
        
        self.q_max = Pp*u
        self.q_min = Pp*l
        
        self.w_max = self.parameters['infinity']*np.ones(self.num_w)
        self.w_min = -self.parameters['infinity']*np.ones(self.num_w)

        self.r_max = Pr*u

        self.z_max = hu
        self.z_min = hl

        dp = np.maximum(self.p_max-self.p_min,5e-2)
        self.y_max = self.parameters['p_ramp_max']*dp
        self.y_min = -self.parameters['p_ramp_max']*dp

        self.Hp = (Pp*H*Pp.T).tocoo()
        self.gp = Pp*g
        self.Hq = self.Hp*self.parameters['cost_factor']
        self.gq = self.gp*self.parameters['cost_factor']

        self.G = A*Pp.T
        self.C = A*Pp.T
        self.R = A*Pr.T
        self.A = -A*Pw.T
        self.J = G*Pw.T
        self.D = -A*Pl.T
        self.b = b
        
        self.Pp = Pp
        self.Pw = Pw
        self.Pr = Pr

        self.r_cov = r_cov
        self.L_cov = L
        self.L_sca = [np.sqrt(t/(self.T-1.)) for t in range(self.T)] # variance grows linearly

        self.Ip = eye(self.num_p,format='coo')
        self.Iy = eye(self.num_y,format='coo')
        self.Iz = eye(self.num_z,format='coo')

        self.Ow = coo_matrix((self.num_w,self.num_w))
        self.Os = coo_matrix((self.num_s,self.num_s))
        self.Oy = coo_matrix((self.num_y,self.num_y))
        self.Oz = coo_matrix((self.num_z,self.num_z))

        self.oq = np.zeros(self.num_q)
        self.ow = np.zeros(self.num_w)
        self.os = np.zeros(self.num_s)
        self.oy = np.zeros(self.num_y)
        self.oz = np.zeros(self.num_z)

        self.x_prev = np.hstack((self.p_prev,self.oq,self.ow,self.os,self.oy,self.oz)) # stage vars

        self.d_forecast = []
        self.r_forecast = []
        for t in range(self.T):
            for load in net.loads:
                load.P = forecast['load'][load.index][t]
            for gen in net.var_generators:
                gen.P = forecast['vargen'][gen.index][t]
            x = net.get_var_values()
            self.d_forecast.append(Pl*x)
            self.r_forecast.append(Pr*x)

        # Check problem data
        assert(net.num_vars == num_w+num_p+num_r+num_l)
        assert(self.num_p == self.num_q == self.num_y)
        assert(self.num_z == self.num_br)
        assert(np.all(self.p_min == 0.))
        assert(np.all(self.p_min < self.p_max))
        assert(np.all(self.q_min < self.q_max))
        assert(np.all(self.p_min == self.q_min))
        assert(np.all(self.p_max == self.q_max))
        assert(np.all(self.w_min < self.w_max))
        assert(np.all(self.z_min < self.z_max))
        assert(np.all(self.y_min < self.y_max))
        assert(np.all(self.Hp.row == self.Hp.col))
        assert(np.all(self.Hp.data > 0))
        assert(np.all(self.Hq.row == self.Hq.col))
        assert(np.all(self.Hq.data > 0))
        assert(np.all(self.Hq.data == self.parameters['cost_factor']*self.Hp.data))
        assert(np.all(self.gp >= 0))
        assert(np.all(self.gq == self.gp*self.parameters['cost_factor']))
        assert(self.gp.shape == self.gq.shape)
        assert(self.D.shape == (self.num_bus,self.num_l))
        assert(self.G.shape == (self.num_bus,self.num_p))
        assert(self.C.shape == (self.num_bus,self.num_q))
        assert(self.R.shape == (self.num_bus,self.num_s))
        assert(self.A.shape == (self.num_bus,self.num_w))
        assert(self.J.shape == (self.num_br,self.num_w))
        assert(self.b.shape == (self.num_bus,))
        assert(all([d.shape == (self.num_l,) for d in self.d_forecast]))
        assert(all([r.shape == (self.num_r,) for r in self.r_forecast]))
        assert(all([np.all(r < self.r_max) for r in self.r_forecast]))
        assert(all([np.all(r >= 0) for r in self.r_forecast]))
        assert(np.all(D.row == D.col))
        assert(np.all(Dh.row == Dh.col))
        assert(np.all(D.data > 0))
        assert(np.all(Dh.data > 0))
        assert(self.r_cov.shape == (self.num_r,self.num_r))
        for i in range(10):
            z = np.random.randn(self.num_r)
            assert(norm(self.r_cov*z-self.L_cov*self.L_cov.T*z) < 1e-10)

        # Construct base problems
        self.base_problem = []
        for t in range(self.T):
            self.base_problem.append(self.construct_base_problem(t))

    def construct_base_problem(self,t,tf=None):
        """
        Constructs base problem for given time period.

        Parameters
        ----------
        t : {0,...,T} (initial stage)
        tf : {0,...,T} (end stage)

        Returns
        -------
        problem : QuadProblem
        """

        if tf is None:
            tf = self.T-1
        
        assert(t >= 0)
        assert(t < self.T)
        assert(tf >= 0)
        assert(tf < self.T)
        assert(t <= tf)

        H_list = []
        g_list = []
        A_list = []
        b_list = []
        l_list = []
        u_list = []

        for i in range(tf-t+1):

            H = bmat([[self.Hp,None,None,None,None,None],  # p
                      [None,self.Hq,None,None,None,None],  # q
                      [None,None,self.Ow,None,None,None],  # w
                      [None,None,None,self.Os,None,None],  # s
                      [None,None,None,None,self.Oy,None],  # y
                      [None,None,None,None,None,self.Oz]], # z
                     format='coo')

            g = np.hstack((self.gp,  # p (place to add correction)
                           self.gq,  # q
                           self.ow,  # w
                           self.os,  # s
                           self.oy,  # y
                           self.oz)) # z

            Arow1 = 6*(tf-t+1)*[None]
            Arow1[6*i:6*(i+1)] = [self.G,self.C,-self.A,self.R,None,None]
            
            Arow2 = 6*(tf-t+1)*[None]
            Arow2[6*i:6*(i+1)] = [self.Ip,None,None,None,-self.Iy,None]
            if i > 0:
                Arow2[6*(i-1)] = -self.Ip

            Arow3 = 6*(tf-t+1)*[None]
            Arow3[6*i:6*(i+1)] = [None,None,self.J,None,None,-self.Iz]

            H_list.append(H)
            g_list.append(g)

            A_list += [Arow1,Arow2,Arow3]
            b_list += [self.b+self.D*self.d_forecast[t+i],
                       self.oy, # (place to add p_prev for first stage)
                       self.oz]
            
            u_list += [self.p_max,
                       self.q_max,
                       self.w_max,
                       self.r_max, # (place to add available r)
                       self.y_max,
                       self.z_max]
            l_list += [self.p_min,
                       self.q_min,
                       self.w_min,
                       self.os,
                       self.y_min,
                       self.z_min]
            
        H = block_diag(H_list,format='coo')
        g = np.hstack(g_list)
 
        A = bmat(A_list,format='coo')
        b = np.hstack(b_list)

        # Stages after start stage
        if A_list[3:]:
            An = bmat([a[6:] for a in A_list[3:]],format='coo') # ignores first 6 blocks
            bn = np.hstack((b_list[3:]))
        else:
            An = None
            bn = None

        u = np.hstack((u_list))
        l = np.hstack((l_list))

        # Checks
        num_vars = self.num_x*(tf-t+1)
        assert(H.shape == (num_vars,num_vars))
        assert(g.shape == (num_vars,))
        assert(A.shape == ((self.num_bus+self.num_p+self.num_z)*(tf-t+1),num_vars))
        assert(b.shape == ((self.num_bus+self.num_p+self.num_z)*(tf-t+1),))
        assert(u.shape == (num_vars,))
        assert(l.shape == (num_vars,))
        assert(np.all(l < u))

        # Problem
        problem = QuadProblem(H,g,A,b,l,u)
        problem.An = An
        problem.bn = bn
        return problem

    def construct_x(self,p=None,q=None,w=None,s=None,y=None,z=None):
        """
        Constructs stage vector from components.
        
        Parameters
        ----------

        Returns
        -------
        """

        return np.hstack((p,q,w,s,y,z))

    def eval_F(self,t,x,w):
        """
        Evaluates current cost.
        
        Parameters
        ----------
        t : {0,...,T-1}
        x : vector
        w : vector
        
        Returns
        -------
        F : float
        """
        
        p,q,w,s,y,z = self.separate_x(x)

        return (np.dot(self.gp,p) +
                0.5*np.dot(p,self.Hp*p) + # slow gen cost
                np.dot(self.gq,q) +
                0.5*np.dot(q,self.Hq*q)) # fast gen cost

    def separate_x(self,x):
        """
        Separates stage vector into components.
        
        Parameters
        ----------

        Returns
        -------
        """
        
        offset = 0
        p = x[offset:offset+self.num_p]
        offset += self.num_p
        
        q = x[offset:offset+self.num_q]
        offset += self.num_q

        w = x[offset:offset+self.num_w]
        offset += self.num_w

        s = x[offset:offset+self.num_s]
        offset += self.num_s

        y = x[offset:offset+self.num_y]
        offset += self.num_y

        z = x[offset:offset+self.num_z]
        offset += self.num_z

        return p,q,w,s,y,z

    def get_num_stages(self):
        """
        Gets number of stages.
        
        Returns
        -------
        num : int
        """
        
        return self.T

    def get_size_x(self,t):
        """
        Gets size of stage vector x.

        Parameters
        ----------
        t : {0,...,T-1}        

        Returns
        -------
        size : int
        """

        assert(0 <= t < self.T)
        
        return self.num_x

    def get_x_prev(self):
        """
        Gets constant x for time before t=0.

        Returns
        -------
        x_prev : vector
        """
        
        return self.x_prev

    def solve_stage_with_cuts(self,t,w,x_prev,A,b,tol=1e-4,init_data=None,quiet=False):
        """
        Solves approximate stage problem for given realization of
        uncertainty and cuts that approximate cost-to-go function.

        Parameters
        ----------
        t : {0,...,T-1}
        w : random vector
        x_prev : vector of previous stage variables
        A : matrix for constructing cuts
        b : vector for contructing cuts
        tol : solver tolerance
        init_data : dict (for warm start)
        quiet : {True,False}        

        Results
        -------
        x : stage-t solution
        H : stage-t cost
        gH : stage-t cost subgradient wrt x_prev
        results : dict
        """
        
        assert(t >= 0)
        assert(t < self.T)
        assert(x_prev.shape == (self.num_x,))
        assert(A.shape[1] == self.num_x)
        assert(b.shape == (A.shape[0],))

        p_prev = x_prev[:self.num_p]
        inf = self.parameters['infinity']*1e2

        H = bmat([[self.Hp,None,None,None,None,None],  # p
                  [None,self.Hq,None,None,None,None],  # q
                  [None,None,self.Ow,None,None,None],  # w
                  [None,None,None,self.Os,None,None],  # s
                  [None,None,None,None,self.Oy,None],  # y
                  [None,None,None,None,None,self.Oz]], # z
                 format='coo')

        g = np.hstack([self.gp,  # p
                       self.gq,  # q
                       self.ow,  # w
                       self.os,  # s
                       self.oy,  # y
                       self.oz]) # z

        Aeq = bmat([[self.G,self.C,-self.A,self.R,None,None], # power balance
                    [self.Ip,None,None,None,-self.Iy,None],   # ramp eq
                    [None,None,self.J,None,None,-self.Iz]],   # thermal lim eq
                   format='coo')
        
        beq = np.hstack([self.b+self.D*self.d_forecast[t],
                         p_prev,
                         self.oz])
        
        u = np.hstack([self.p_max,
                       self.q_max,
                       self.w_max,
                       w,           # avail r
                       self.y_max,
                       self.z_max])
        l = np.hstack([self.p_min,
                       self.q_min,
                       self.w_min,
                       self.os,
                       self.y_min,
                       self.z_min])
        
        # Cuts (h are slack vectors, v is scalar)
        num_cuts = A.shape[0]
        Oh = coo_matrix((num_cuts,num_cuts))
        oh = np.zeros(num_cuts)
        Ih = eye(num_cuts,format='coo')
        Ev = np.ones((num_cuts,1))
        Ov = coo_matrix((1,1))
        if num_cuts > 0:
            
            H = bmat([[H ,None,None],
                      [None,Ov,None],
                      [None,None,Oh]],
                     format='coo')

            g = np.hstack((g,1.,oh))
            
            Aeq = bmat([[Aeq,None,None],
                        [A,Ev,-Ih]],
                       format='coo')
            
            beq = np.hstack((beq,
                             -b))

            u = np.hstack((u,
                           inf,
                           np.ones(num_cuts)*inf))
            
            l = np.hstack((l,    # x
                           -inf, # v
                           oh))  # h (slack)
        
        # Construct problem
        QPproblem = QuadProblem(H,g,Aeq,beq,l,u)

        # Warm start
        if init_data is not None:
            x0 = init_data['x']
            lam0 = init_data['lam']
            mu0 = init_data['mu']
            pi0 = init_data['pi']
            QPproblem.x = np.hstack((x0,np.zeros(g.size-x0.size)))
            QPproblem.lam = np.hstack((lam0,np.zeros(beq.size-lam0.size)))
            QPproblem.mu = np.hstack((mu0,np.zeros(g.size-x0.size)))
            QPproblem.pi = np.hstack((pi0,np.zeros(g.size-x0.size)))
        
        # Set up solver
        solver = OptSolverIQP()
        solver.set_parameters({'quiet': quiet, 
                               'tol': tol})
        
        # Solve
        solver.solve(QPproblem)

        # Results
        results = solver.get_results()

        # Stage optimal point
        x = solver.get_primal_variables()
        
        # Optimal duals
        lam,nu,mu,pi = solver.get_dual_variables()

        # Solutions
        xt = x[:self.num_x]
        y_offset = self.num_p+self.num_q+self.num_w+self.num_s
        H = np.dot(QPproblem.g,x)+0.5*np.dot(x,QPproblem.H*x)
        gH = np.hstack(((-mu+pi)[y_offset:y_offset+self.num_y],
                        self.oq,self.ow,self.os,self.oy,self.oz))

        # Return
        return xt,H,gH,results

    def solve_stages(self,t,w_list,x_prev,g_list=[],tf=None,tol=1e-4,quiet=False,next=False):
        """
        Solves stages using given realizations of uncertainty 
        and cost-to-go slope corrections.
        
        Parameters
        ----------
        t : {0,...,T-1}
        w_list : list of random vectors for stage t,...,tf
        x_prev : vector of previous stage variables
        g_list : list of slope corrections for stage t,...,tf
        tf : {0,...,T-1} (T-1 by default)
        tol : float
        quiet : {True,False}
        next : {True,False}
        
        Returns
        -------
        x : stage-t solution
        H : stage-t cost
        gH : stage-t cost subgradient wrt x_prev
        gHnext : stage-(t+1) cost subgradient wrt x
        """

        if tf is None:
            tf = self.T-1

        if not len(g_list):
            g_list = (tf-t+1)*[np.zeros(self.num_x)]
        
        assert(t >= 0)
        assert(t < self.T)
        assert(tf >= 0)
        assert(tf < self.T)
        assert(t <= tf)
        assert(len(w_list) == tf-t+1)
        assert(len(g_list) == tf-t+1)
        assert(x_prev.shape == (self.num_x,))
        
        p_prev = x_prev[:self.num_p]

        # Base problem
        if tf == self.T-1:
            QPproblem = self.base_problem[t]
        else:
            QPproblem = self.construct_base_problem(t,tf=tf)
        
        # Updates
        p_offset = 0
        s_offset = self.num_p+self.num_q+self.num_w
        QPproblem.b[self.num_bus:self.num_bus+self.num_y] = p_prev
        for i in range(tf-t+1):
            QPproblem.g[p_offset:p_offset+self.num_p] = self.gp+g_list[i][:self.num_p]
            QPproblem.u[s_offset:s_offset+self.num_s] = w_list[i]
            p_offset += self.num_x
            s_offset += self.num_x

        # Quiet
        if not quiet:
            QPproblem.show()

        # Set up solver
        solver = OptSolverIQP()
        solver.set_parameters({'quiet': quiet, 
                               'tol': tol})
        
        # Solve
        solver.solve(QPproblem)
        assert(solver.get_status() == 'solved')

        # Results
        results = solver.get_results()

        # Stage optimal point
        x = solver.get_primal_variables()
        
        # Optimal duals
        lam,nu,mu,pi = solver.get_dual_variables()

        # Solutions
        xt = x[:self.num_x]
        y_offset = self.num_p+self.num_q+self.num_w+self.num_s
        H = np.dot(QPproblem.g,x)+0.5*np.dot(x,QPproblem.H*x)
        gH = np.hstack(((-mu+pi)[y_offset:y_offset+self.num_y],
                        self.oq,self.ow,self.os,self.oy,self.oz))
        gHnext = None

        # Next stage
        if t < self.T-1 and next:
            Pn = eye(x.size-self.num_x,x.size,self.num_x,format='csr')
            xn = Pn*x
            un = Pn*QPproblem.u
            ln = Pn*QPproblem.l
            An = QPproblem.An
            bn = QPproblem.bn
            bn[self.num_bus:self.num_bus+self.num_y] = x[:self.num_p] 
            gn = Pn*QPproblem.g
            Hn = Pn*QPproblem.H*Pn.T
            lamn = lam[self.num_bus+self.num_p+self.num_z:]
            solver.solve(QuadProblem(Hn,gn,An,bn,ln,un,x=xn,lam=lamn,mu=Pn*mu,pi=Pn*pi))
            xn = solver.get_primal_variables()
            lamn,nun,mun,pin = solver.get_dual_variables() 
            gHnext = np.hstack(((-mun+pin)[y_offset:y_offset+self.num_y],
                                self.oq,self.ow,self.os,self.oy,self.oz))
            
        # Return
        return xt,H,gH,gHnext

    def is_point_feasible(self,t,x,x_prev,w):
        """
        Checks wether point is feasible for the given stage.

        Parameters
        ----------
        t : {0,...,T-1}
        x : vector
        x_prev : vector
        w : vector

        Returns
        -------
        flag : {True,False}
        """

        r = w
        p,q,w,s,y,z = self.separate_x(x)
        p_prev,q_prev,w_prev,s_prev,y_prev,z_prev = self.separate_x(x_prev)

        try: 
            eps = 1e-4
            assert 0 <= t < self.T, 'time'
            assert np.all(self.y_min <= y), 'ramp min'
            assert np.all(self.y_max >= y), 'ramp_max'
            assert np.all(self.z_min <= z), 'thermal_min'
            assert np.all(self.z_max >= z), 'thermal_max'
            assert np.all(self.q_min <= q), 'fast_min'
            assert np.all(self.q_max >= q), 'fast_max'
            assert np.all(self.p_min <= p), 'slow_min'
            assert np.all(self.p_max >= p), 'slow_max'
            assert np.all(self.w_min <= w), 'ang_min'
            assert np.all(self.w_max >= w), 'ang_max'
            assert np.all(0 <= s),          'ren_min'
            assert np.all(r >= s),          'ren_max'
            assert norm(self.G*p+self.C*q+self.R*s-self.A*w-self.b-self.D*self.d_forecast[t])/norm(self.A.data) < eps, 'power flow'
            assert norm(self.J*w-z)/norm(self.J.data) < eps, 'thermal eq'
            assert norm(p-p_prev-y)/(norm(p)+norm(p_prev)+norm(y)) < eps, 'ramp eq'
            return True
        except AssertionError as e:
            print(e)
        return False

    def sample_w(self,t,observations):
        """
        Samples realization of renewable powers for the given stage
        given the observations.

        Parameters
        ----------
        t : {0,...,T-1}
        observations : list (length t)

        Parameters
        ----------
        w : vector
        """

        assert(t >= 0)
        assert(t < self.T)
        assert(len(observations) == t)

        r_eps = self.parameters['r_eps']
        r_ramp_max = self.parameters['r_ramp_max']
        r_ramp_freq = self.parameters['r_ramp_freq']

        assert(0 <= r_ramp_freq <= 1.)

        r = self.r_forecast[t]+self.L_sca[t]*self.L_cov*np.random.randn(self.num_r) # perturbed
        r = np.maximum(np.minimum(r,self.r_max),r_eps)                              # cap bound
        if observations and np.random.rand() <= 1.-r_ramp_freq:
            dr = r_ramp_max*self.r_max
            rprev = observations[-1]
            return np.maximum(np.minimum(r,rprev+dr),rprev-dr)                      # ramp bound with prob 1-eps
        else:
            return r

    def sample_W(self,t,t_from=0,observations=[]):
        """
        Samples realization of renewable powers up
        to the given stage.
        
        Parameters
        ----------
        t : {0,...,T-1}
        t_from : {0,...,T-1}
        observations : list (length t_from)

        Parameters
        ----------
        W : list (length t+1)
        """

        assert(t >= 0)
        assert(t < self.T)
        assert(len(observations) == t_from)
        if t_from > t:
            return []

        samples = list(observations)
        for tau in range(t_from,t+1):
            samples.append(self.sample_w(tau,samples))
        assert(len(samples) == t+1)
        return samples[t_from:]

    def predict_w(self,t,observations):
        """
        Predicts renewable powers for the given stage
        given the observations.

        Parameters
        ----------
        t : {0,...,T-1}
        observations : list (length t)

        Returns
        -------
        w : vector
        """

        assert(t >= 0)
        assert(t < self.T)
        assert(len(observations) == t)

        r_pred = np.zeros(self.num_r)
        for i in range(self.parameters['num_samples']):
            r_pred *= float(i)/float(i+1)
            r_pred += self.sample_w(t,observations)/(i+1.)
        return r_pred

    def predict_W(self,t,t_from=0,observations=[]):
        """
        Predicts renewable powers up to the
        given stage.

        Parameters
        ----------
        t : {0,...,T-1}
        t_from : {0,...,T-1}
        observations : list (length t_from)

        Returns
        -------
        W : list
        """
        
        assert(t >= 0)
        assert(t < self.T)
        assert(len(observations) == t_from)
        if t_from > t:
            return []
        
        r_pred = np.zeros((t-t_from+1,self.num_r))
        for i in range(self.parameters['num_samples']):
            r_pred *= float(i)/float(i+1)
            r_pred += np.array(self.sample_W(t,t_from,observations))/(i+1.)
        assert(r_pred.shape == (t-t_from+1,self.num_r))
        predictions = [r_pred[tau,:] for tau in range(t-t_from+1)]
        assert(len(predictions) == t-t_from+1)
        return predictions

    def set_parameters(self,params):
        """
        Sets problem parameters.
        
        Parameters
        ----------
        params : dic
        """
        
        for key,value in list(params.items()):
            if key in self.parameters:
                self.parameters[key] = value
 
    def show(self,scenario_tree=None):
        """
        Shows problem information.

        Parameters
        ----------
        scenario_tree : StochProbleMS_Tree
        """

        vargen_cap = np.sum(self.r_max)
        vargen_for = [np.sum(r) for r in self.r_forecast]
        vargen_unc = [np.sum(np.sqrt(tril(triu((s**2.)*self.r_cov)).tocoo().data)) for s in self.L_sca]
        load_for = [np.sum(d) for d in self.d_forecast]
        load_max = max(load_for)
 
        print('\nStochastic Multi-Stage DCOPF')
        print('----------------------------')
        print('num buses          : %d' %self.num_bus)
        print('num branches       : %d' %self.num_br)
        print('num gens           : %d' %self.num_p)
        print('num vargens        : %d' %self.num_r)
        print('num loads          : %d' %self.num_l)
        print('num stages         : %d' %self.T)
        print('vargen cap         : %.2f (%% of max load)' %(100.*vargen_cap/load_max))
        print('vargen corr_rad    : %d (edges)' %(self.corr_radius))
        print('vargen corr_val    : %.2f (unitless)' %(self.corr_value))

        if scenario_tree is not None:
            scenario_tree.show()

        # Draw
        if self.parameters['draw']:
        
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
            import seaborn

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            rcParams.update({'figure.autolayout': True})
            seaborn.set_style("ticks")

            N = 20
            colors = seaborn.color_palette("muted",N)
 
            # Vargen forecast
            plt.subplot(2,2,1)
            plt.plot([100.*r/load_max for r in vargen_for])
            plt.xlabel(r'stage')
            plt.ylabel(r'vargen forecast (\% of max load)')
            plt.axis([0,self.T-1,0.,100.])
            plt.grid()

            # Vargen uncertainty
            plt.subplot(2,2,2)
            plt.plot([100.*u/vargen_cap for u in vargen_unc])
            plt.xlabel(r'stage')
            plt.ylabel(r'vargen uncertainty (\% of local cap)')
            plt.axis([0,self.T-1,0.,100.])
            plt.grid()
            
            # Vargen profile
            plt.subplot(2,2,3)
            plt.plot([r/max(vargen_for) for r in vargen_for])
            plt.xlabel(r'stage')
            plt.ylabel(r'vargen profile')
            plt.axis([0,self.T-1,0.,1.])
            plt.grid()
            
            # Load profile
            plt.subplot(2,2,4)
            plt.plot([l/max(load_for) for l in load_for])
            plt.xlabel(r'stage')
            plt.ylabel(r'load profile')
            plt.axis([0,self.T-1,0.,1.])
            plt.grid()
            
            # Vargen prediction
            fig = plt.figure(figsize=(6,5))
            plt.hold(True)
            for i in range(N):
                R = [np.sum(w) for w in self.sample_W(self.T-1)]
                plt.plot(range(1,self.T+1),[100.*r/load_max for r in R],color=colors[i])
            R = [np.sum(w) for w in self.predict_W(self.T-1)]
            plt.plot(range(1,self.T+1),[100.*r/load_max for r in R],color='black',linewidth=3.)
            plt.xlabel(r'stage',fontsize=22)
            plt.ylabel(r'power (\% of max load)',fontsize=22)
            plt.axis([1,self.T,0.,100.])
            plt.tick_params(axis='both',which='major',labelsize=18)
            plt.tick_params(axis='both',which='minor',labelsize=18)
            plt.title(r'%s: Renewables' %self.parameters['name'],fontsize=22,y=1.05)
            plt.grid()

            # Scenario tree
            if scenario_tree is not None:
                scenario_tree.draw()

            # Vargen prediction from scenario tree
            if scenario_tree is not None:
                fig = plt.figure()
                plt.hold(True)
                for i in range(N):
                    R = [np.sum(n.get_w()) for n in scenario_tree.sample_branch(self.T-1)]
                    plt.plot([100.*r/load_max for r in R],color=colors[i])
                R = [np.sum(w) for w in self.predict_W(self.T-1)]
                plt.plot([100.*r/load_max for r in R],color='black',linewidth=3.)
                plt.xlabel(r'stage',fontsize=22)
                plt.ylabel(r'\% of max load',fontsize=22)
                plt.axis([0,self.T-1,0.,100.])
                plt.tick_params(axis='both',which='major',labelsize=20)
                plt.tick_params(axis='both',which='minor',labelsize=20)
                plt.title(r'Renewables (Scenerio Tree)')
                plt.grid()

            # Vargen closests branch from scenario tree
            if scenario_tree is not None:
                fig = plt.figure()
                plt.hold(True)
                for i in range(3):
                    W = self.sample_W(self.T-1)
                    R1 = [np.sum(w) for w in W]
                    R2 = [np.sum(n.get_w()) for n in scenario_tree.get_closest_branch(W)]
                    plt.plot([100.*r/load_max for r in R1],color=colors[i],linestyle='-')
                    plt.plot([100.*r/load_max for r in R2],color=colors[i],linestyle='--')
                    plt.xlabel(r'stage',fontsize=22)
                    plt.ylabel(r'\% of max load',fontsize=22)
                    plt.axis([0,self.T-1,0.,100.])
                    plt.tick_params(axis='both',which='major',labelsize=20)
                    plt.tick_params(axis='both',which='minor',labelsize=20)
                    plt.title(r'Closest Branch from Scenerio Tree')
                    plt.grid()
            plt.show()

    def simulate_policies(self,sim_id):
        """
        Simulates policies for a given realization of uncertainty.

        Parameters
        ----------
        sim_id : int

        Returns
        -------
        dtot : vector
        rtot : vector
        cost : dict
        ptot : dict
        qtot : dict
        stot : dict
        """

        t0 = time.time()
    
        policies = self.policies
        R = self.samples[sim_id]

        assert(len(R) == self.T)

        print('simulation %d,' %sim_id, end=' ')

        num = len(policies)
        dtot = np.zeros(self.T)
        rtot = np.zeros(self.T)
        cost = dict([(i,np.zeros(self.T)) for i in range(num)])
        ptot = dict([(i,np.zeros(self.T)) for i in range(num)])
        qtot = dict([(i,np.zeros(self.T)) for i in range(num)])
        stot = dict([(i,np.zeros(self.T)) for i in range(num)])
        x_prev = dict([(i,self.x_prev) for i in range(num)])
        for t in range(self.T):
            r = R[t]
            dtot[t] = np.sum(self.d_forecast[t])
            rtot[t] = np.sum(r)
            for i in range(num):
                x = policies[i].apply(t,x_prev[i],R[:t+1])
                p,q,w,s,y,z = self.separate_x(x)
                F = self.eval_F(t,x,r)
                for tau in range(t+1):
                    cost[i][tau] += F
                ptot[i][t] = np.sum(p)
                qtot[i][t] = np.sum(q)
                stot[i][t] = np.sum(s)
                x_prev[i] = x.copy()

        print('time %.2f min' %((time.time()-t0)/60.))
                    
        return dtot,rtot,cost,ptot,qtot,stot

    def evaluate_policies(self,policies,num_sims,seed=1000,num_procs=0,outfile=''):
        """
        Simulates operation policies.

        Parameters
        ----------
        policies : list of StochProblemMS_Policy objects
        num_sims : int
        seed : int
        num_procs : int
        outfile : string (name of output file)
        ref_pol : string (name of refernece policy)
        """

        assert(len(policies) > 0)

        from multiprocess import Pool,cpu_count,Process
        
        if not num_procs:
            num_procs = cpu_count()
            
        if not outfile:
            outfile = 'evaluation.csv'

        csvfile = open(outfile,'wb')
        writer = csv.writer(csvfile)

        np.random.seed(seed)

        print('Evaluating policies with %d processes' %num_procs)
                   
        # Eval
        self.policies = policies
        self.samples = [self.sample_W(self.T-1) for i in range(num_sims)]
        if num_procs > 1:
            pool = Pool(num_procs)
            func = pool.map
        else:
            func = map
        t0 = time.time()
        results = func(lambda i: self.simulate_policies(i), range(num_sims))
        t1 = time.time()            
        print('Total time: %.2f min' %((t1-t0)/60.))

        # Process
        num_pol = len(policies)
        dtot,rtot,cost,ptot,qtot,stot = list(zip(*results))
        dtot = np.average(np.array(dtot),axis=0)
        rtot = np.average(np.array(rtot),axis=0)
        cost = dict([(i,np.average(np.array([cost[j][i] for j in range(num_sims)]),axis=0)) for i in range(num_pol)])
        ptot = dict([(i,np.average(np.array([ptot[j][i] for j in range(num_sims)]),axis=0)) for i in range(num_pol)])
        qtot = dict([(i,np.average(np.array([qtot[j][i] for j in range(num_sims)]),axis=0)) for i in range(num_pol)])
        stot = dict([(i,np.average(np.array([stot[j][i] for j in range(num_sims)]),axis=0)) for i in range(num_pol)])
        
        # Checks
        assert(dtot.shape == (self.T,))
        assert(rtot.shape == (self.T,))
        for i in range(num_pol):
            assert(cost[i].shape == (self.T,))
            assert(ptot[i].shape == (self.T,))
            assert(qtot[i].shape == (self.T,))
            assert(stot[i].shape == (self.T,))

        # Ref policy
        try:
            iref = [p.name for p in policies].index('CE')
        except ValueError:
            iref = 0
        
        # Write
        writer.writerow([self.num_bus,num_sims])
        writer.writerow([p.get_name() for p in policies])
        writer.writerow([p.get_construction_time() for p in policies])
        writer.writerow([p.get_param1() for p in policies])
        writer.writerow([p.get_param2() for p in policies])
        writer.writerow(['d','r']+num_pol*['cost','p','q','s'])
        for t in range(self.T):
            row = [dtot[t],rtot[t]]
            for i in range(num_pol):
                row += [cost[i][t]/cost[iref][t],
                        ptot[i][t],
                        qtot[i][t],
                        stot[i][t]]
            writer.writerow(row)
        csvfile.close()
