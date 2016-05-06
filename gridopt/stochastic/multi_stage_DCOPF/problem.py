#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.    #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

import pfnet as pf
import numpy as np
from numpy.linalg import norm
from optalg.opt_solver.opt_solver_error import *
from optalg.stoch_solver import StochObjMS_Problem
from optalg.opt_solver import OptSolverIQP,QuadProblem
from scipy.sparse import triu,bmat,coo_matrix,eye,spdiags
            
class MS_DCOPF(StochObjMS_Problem):
    
    # Parameters
    parameters = {'cost_factor' : 1e2,   # factor for determining fast gen cost
                  'infinity' : 1e3,      # infinity
                  'flow_factor' : 1.,    # factor for relaxing thermal limits
                  'max_ramping' : 0.1,   # factor for constructing ramping limits
                  'num_samples' : 2000}  # number of samples

    def __init__(self,net,horizon):
        """
        Class constructor.
        
        Parameters
        ----------
        net : PFNET Network
        """

        # Parameters
        self.parameters = MS_DCOPF.parameters.copy()

        # Save info
        self.net = net
        self.T = horizon

        # Generator limits
        for gen in net.generators:
            gen.P_min = 0.
            gen.P_max = np.maximum(gen.P_max,0.)
            assert(gen.P_min <= gen.P_max)

        # Branch flow limits
        for br in net.branches:
            if br.ratingA == 0.:
                br.ratingA = self.parameters['infinity']
            else:
                br.ratingA *= self.parameters['flow_factor']
                
        # Counters
        num_w = net.num_buses-net.get_num_slack_buses() # voltage angles
        num_p = net.get_num_P_adjust_gens()             # adjustable generators
        num_r = net.num_vargens                         # renewable generators
        num_l = net.num_loads                           # loads
        num_bus = net.num_buses                         # buses
        num_br = net.num_branches                       # branches
        
        # Variables
        net.clear_flags()
        net.set_flags(pf.OBJ_BUS,
                      pf.FLAG_VARS,
                      pf.BUS_PROP_NOT_SLACK,
                      pf.BUS_VAR_VANG)
        net.set_flags(pf.OBJ_GEN,
                      pf.FLAG_VARS,
                      pf.GEN_PROP_P_ADJUST,
                      pf.GEN_VAR_P)
        net.set_flags(pf.OBJ_LOAD,
                      pf.FLAG_VARS,
                      pf.LOAD_PROP_ANY,
                      pf.LOAD_VAR_P)
        net.set_flags(pf.OBJ_VARGEN,
                      pf.FLAG_VARS,
                      pf.VARGEN_PROP_ANY,
                      pf.VARGEN_VAR_P)

        # Current values
        x = net.get_var_values()

        # Projections
        Pw = net.get_var_projection(pf.OBJ_BUS,pf.BUS_VAR_VANG)
        Pp = net.get_var_projection(pf.OBJ_GEN,pf.GEN_VAR_P)
        Pl = net.get_var_projection(pf.OBJ_LOAD,pf.LOAD_VAR_P)
        Pr = net.get_var_projection(pf.OBJ_VARGEN,pf.VARGEN_VAR_P)
        assert(Pw.shape == (num_w,net.num_vars))
        assert(Pp.shape == (num_p,net.num_vars))
        assert(Pl.shape == (num_l,net.num_vars))
        assert(Pr.shape == (num_r,net.num_vars))

        # Power flow equations
        pf_eq = pf.Constraint(pf.CONSTR_TYPE_DCPF,net)
        pf_eq.analyze()
        pf_eq.eval(x)
        A = pf_eq.A.copy()
        b = pf_eq.b.copy()

        # Branch flow limits
        fl_lim = pf.Constraint(pf.CONSTR_TYPE_DC_FLOW_LIM,net)
        fl_lim.analyze()
        fl_lim.eval(x)
        G = fl_lim.G.copy()
        hl = fl_lim.l.copy()
        hu = fl_lim.u.copy()
        assert(np.all(hl < hu))
        
        # Generation cost
        cost = pf.Function(pf.FUNC_TYPE_GEN_COST,1.,net)
        cost.analyze()
        cost.eval(x)
        H = (cost.Hphi + cost.Hphi.T - triu(cost.Hphi))/net.base_power # symmetric, scaled
        g = cost.gphi/net.base_power - H*x
        
        # Bounds
        l = net.get_var_values(pf.LOWER_LIMITS)
        u = net.get_var_values(pf.UPPER_LIMITS)
        assert(np.all(Pw*l < Pw*u))
        assert(np.all(Pp*l < Pp*u))
        assert(np.all(Pl*l <= Pl*u))
        assert(np.all(Pr*l < Pr*u))

        # Renewable covariance
        from scikits.sparse.cholmod import cholesky
        r_cov = Pr*net.create_vargen_P_sigma(net.vargen_corr_radius,net.vargen_corr_value)*Pr.T
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

        self.p_max = Pp*u
        self.p_min = Pp*l
        
        self.q_max = Pp*u
        self.q_min = Pp*l
        
        self.w_max = self.parameters['infinity']*np.ones(self.num_w)
        self.w_min = -self.parameters['infinity']*np.ones(self.num_w)

        self.r_max = Pr*u
        self.r_base = Pr*x

        self.z_max = hu
        self.z_min = hl

        self.y_max = self.parameters['max_ramping']*(self.p_max-self.p_min)
        self.y_min = -self.parameters['max_ramping']*(self.p_max-self.p_min)
      
        self.Hp = (Pp*H*Pp.T).tocoo()
        self.gp = Pp*g
        self.Hq = self.Hp*self.parameters['cost_factor']
        self.gq = np.zeros(self.num_q)

        self.G = A*Pp.T
        self.C = A*Pp.T
        self.R = A*Pr.T
        self.A = -A*Pw.T
        self.J = G*Pw.T
        self.b = b
        
        self.Pp = Pp
        self.Pw = Pw
        self.Pr = Pr

        self.r_cov = r_cov
        self.L_cov = L

        self.Ow = coo_matrix((self.num_w,self.num_w))
        self.Os = coo_matrix((self.num_s,self.num_s))
        self.Oy = coo_matrix((self.num_y,self.num_y))
        self.Oz = coo_matrix((self.num_z,self.num_z))

        self.ow = np.zeros(self.num_w)
        self.os = np.zeros(self.num_s)
        self.oy = np.zeros(self.num_y)
        self.oz = np.zeros(self.num_z)

        # Check problem data
        assert(net.num_vars == num_w+num_p+num_r+num_l)
        assert(self.num_p == self.num_q == self.num_y)
        assert(self.num_z == self.num_br)
        assert(np.all(self.p_min < self.p_max))
        assert(np.all(self.q_min < self.q_max))
        assert(np.all(self.p_min == self.q_min))
        assert(np.all(self.p_max == self.q_max))
        assert(np.all(self.w_min < self.w_max))
        assert(np.all(self.z_min < self.z_max))
        assert(np.all(self.y_min < self.y_max))
        assert(np.all(self.r_base < self.r_max))
        assert(np.all(self.r_base >= 0))
        assert(np.all(self.Hp.row == self.Hp.col))
        assert(np.all(self.Hp.data > 0))
        assert(np.all(self.Hq.row == self.Hq.col))
        assert(np.all(self.Hq.data > 0))
        assert(np.all(self.Hq.data == 100*self.Hp.data))
        assert(np.all(self.gp >= 0))
        assert(np.all(self.gq == 0))
        assert(self.gp.shape == self.gq.shape)
        assert(self.G.shape == (self.num_bus,self.num_p))
        assert(self.C.shape == (self.num_bus,self.num_q))
        assert(self.R.shape == (self.num_bus,self.num_s))
        assert(self.A.shape == (self.num_bus,self.num_w))
        assert(self.J.shape == (self.num_br,self.num_w))
        assert(self.b.shape == (self.num_bus,))
        assert(np.all(D.row == D.col))
        assert(np.all(Dh.row == Dh.col))
        assert(np.all(D.data > 0))
        assert(np.all(Dh.data > 0))
        assert(self.r_cov.shape == (self.num_r,self.num_r))
        for i in range(10):
            z = np.random.randn(self.num_r)
            assert(norm(self.r_cov*z-self.L_cov*self.L_cov.T*z) < 1e-10)

    def eval_stage_approx(self,t,p_prev,r_list,g_corr=[]):
        """
        Evaluates approximate optimal stage cost.

        Parameters
        ----------
        t : int (stage)
        p_prev : vector (prev slow gen powers)
        r_list : list of renewable injections for stage t,...,T
        g_corr : list of slope corrections for stage t,...,T

        Returns
        -------
        x : stage solution
        Q : stage cost
        gQ : stage cost subgradient wrt p_prev
        """
        
        assert(t >= 0)
        assert(t <= self.T)
        assert(p_prev.shape == (self.num_p,))
        assert(len(r_list) == self.T-t+1)
        assert(len(g_corr) == self.T-t+1 or len(g_corr) == 0)

        if len(g_corr) == 0:
            g_corr = (self.T-t+1)*[0]
        
        H_list = []
        g_list = []

        for i in range(self.T-t+1):

            H = bmat([[self.Hp,None,None,None,None,None],  # p
                      [None,self.Hq,None,None,None,None],  # q
                      [None,None,self.Ow,None,None,None],  # w
                      [None,None,None,self.Os,None,None],  # s
                      [None,None,None,None,self.Oy,None],  # y
                      [None,None,None,None,None,self.Oz]], # z
                     format='coo')

            g = hstack((self.gp + g_corr[i],
                        self.gq,
                        self.ow,
                        self.os,
                        self.oy,
                        self.oz))

            H_list.append(H)
            g_list.append(g)
            
    def show(self):
        """
        Shows problem information.
        """

        tot_cap = np.sum(self.r_max)
        tot_base = np.sum(self.r_base)
        tot_unc = sum([g.P_std for g in self.net.var_generators])
        tot_load = sum([l.P for l in self.net.loads])
        
        print 'Stochastic Multi-Stage DCOPF'
        print '----------------------------'
        print 'buses            : %d' %self.num_bus
        print 'gens             : %d' %self.num_p
        print 'vargens          : %d' %self.num_r
        print 'time horizon     : %d' %self.T
        print 'penetration cap  : %.2f (%% of load)' %(100.*tot_cap/tot_load)
        print 'penetration base : %.2f (%% of load)' %(100.*tot_base/tot_load)
        print 'penetration std  : %.2f (%% of local cap)' %(100.*tot_unc/tot_load)
        print 'correlation rad  : %d (edges)' %(self.net.vargen_corr_radius)
        print 'correlation val  : %.2f (unitless)' %(self.net.vargen_corr_value)        
