#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.    #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

from __future__ import print_function
import pfnet
import numpy as np
from .method_error import *
from .method import PFmethod
from optalg.opt_solver import OptSolverError, OptCallback, OptTermination, OptSolverAugL

class AugLOPF(PFmethod):
    """
    Augmented Lagrangian-based optimal power flow method.
    """
    name = 'AugLOPF'

    parameters = {'weight_cost': 1e0,  # for generation cost
                  'weight_limit': 0,   # for soft limits
                  'weight_ang_reg': 0, # for voltage angle regularization
                  'weight_gen_reg': 0, # for generators regularization
                  'feastol' : 1e-4,    # see AugL
                  'optol' : 1e-4,      # see AugL
                  'kappa' : 1e-2,      # see AugL
                  'vmin_thresh': 0.1}  # threshold for vmin
                   
    def __init__(self):

        PFmethod.__init__(self)
        parameters = OptSolverAugL.parameters.copy()
        parameters.update(AugLOPF.parameters)
        self.parameters = parameters

    def create_problem(self,net):

        # Parameters
        params = self.parameters
        wc  = params['weight_cost']
        wl  = params['weight_limit']
        war = params['weight_ang_reg']
        wgr = params['weight_gen_reg']
        
        # Clear flags
        net.clear_flags()
        
        # Voltage magnitudes
        #Add the term bounded 
        net.set_flags('bus',
                      ['variable','bounded'], 
                      'any',
                      'voltage magnitude')
        
        # Voltage angles
        net.set_flags('bus',
                      'variable',
                      'not slack',
                      'voltage angle')

        # Generator active power
        net.set_flags('generator',
                      ['variable','bounded'],
                      'not on outage',
                      'active power')

        # Generator reactive power
        net.set_flags('generator',
                      ['variable','bounded'],
                      'regulator',
                      'reactive power')

        try:
            assert(net.num_vars == (2*net.num_buses-net.get_num_slack_buses() +
                                    net.get_num_gens_not_on_outage() + 
                                    net.get_num_reg_gens())*net.num_periods)
            assert(net.num_bounded == (net.get_num_gens_not_on_outage() + 
                                       net.get_num_reg_gens())*net.num_periods + net.num_buses)
        except AssertionError:
            raise PFmethodError_BadProblem(self)
                                    
        # Set up problem
        problem = pfnet.Problem()
        problem.set_network(net)
        problem.add_constraint('AC power balance')
        problem.add_constraint('variable bounds') 
        problem.add_function('generation cost',wc/max([net.num_generators,1.]))
        
        #Set of options depending on the parameters
        if wl!=0:
            problem.add_function('soft voltage magnitude limits',wl/max([net.num_buses,1.]))
        if war!=0:
            problem.add_function('voltage angle regularization',war/max([net.num_buses,1.]))
        if wgr!=0:
            problem.add_function('generator powers regularization',wgr/max([net.num_generators,1.]))

        problem.analyze()
        
        # Return
        return problem

    def get_info_printer(self):

        def info_printer(solver,header):
            net = solver.problem.network
            if header:
                print('{0:^5}'.format('vmax'), end=' ')
                print('{0:^5}'.format('vmin'), end=' ')
                print('{0:^6}'.format('bvvio'), end=' ')
                print('{0:^6}'.format('gQvio'), end=' ')
                print('{0:^6}'.format('gPvio'))
            else:
                print('{0:^5.2f}'.format(np.average(net.bus_v_max)), end=' ')
                print('{0:^5.2f}'.format(np.average(net.bus_v_min)), end=' ')
                print('{0:^6.0e}'.format(np.average(net.bus_v_vio)), end=' ')
                print('{0:^6.0e}'.format(np.average(net.gen_Q_vio)), end=' ')
                print('{0:^6.0e}'.format(np.average(net.gen_P_vio)))
        return info_printer
            
    def solve(self,net):
        
        # Parameters
        params = self.parameters
        vmin_thresh = params['vmin_thresh']

        # Problem
        problem = self.create_problem(net)

        # G identity, otherwise use transform
        assert(np.all(problem.G.row == problem.G.col))
        assert(np.all(problem.G.data == 1.))

        # Termination
        def t1(s):
            if np.min(s.problem.network.bus_v_min) < vmin_thresh:
                return True
            else:
                return False
        
        # Info printer
        info_printer = self.get_info_printer()
        
        # Set up solver
        solver = OptSolverAugL()
        solver.set_parameters(params)
        solver.add_termination(OptTermination(t1,'low voltage'))
        solver.set_info_printer(info_printer)
        
        # Solve
        try:
            solver.solve(problem)
        except OptSolverError as e:
            raise PFmethodError_SolverError(self,e)
        finally:
            
            # Get results
            self.set_status(solver.get_status())
            self.set_error_msg(solver.get_error_msg())
            self.set_iterations(solver.get_iterations())
            self.set_primal_variables(solver.get_primal_variables())
            self.set_dual_variables(solver.get_dual_variables())
            self.set_net_properties(net.get_properties())
            self.set_problem(problem)

            # Restors net properties
            net.update_properties()

    def update_network(self,net):
        
        # Get data
        problem = self.results['problem']
        x = self.results['primal variables']
        lam,nu,mu,pi = self.results['dual variables']
       
        # No problem
        if problem is None:
            raise PFmethodError_NoProblem(self)
 
        # Checks
        assert(problem.x.shape == x.shape)
        assert(net.num_vars == x.size)
        assert(problem.A.shape[0] == lam.size)
        assert(problem.f.shape[0] == nu.size)
        assert(problem.G.shape[0] == mu.size)
        assert(problem.G.shape[0] == pi.size)

        # Network quantities
        net.set_var_values(x)

        # Network properties
        net.update_properties()
        
        # Network sensitivities
        net.clear_sensitivities()
        problem.store_sensitivities(lam,nu,mu,pi)
        
