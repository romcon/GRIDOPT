#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015, Tomas Tinoco De Rubira.         #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

from __future__ import print_function
import time
import numpy as np
from .method_error import *
from .method import PFmethod
from numpy.linalg import norm
from collections.abc import Iterable


class ACPF(PFmethod):
    """
    AC power flow method.
    """

    CONTROL_MODE_LOCKED = 'locked'
    CONTROL_MODE_FREE = 'free'
    CONTROL_MODE_REG = 'regulating'
    CONTROL_MODE_DISC = 'discrete'

    name = 'ACPF'

    _parameters = {'weight_vmag': 1e0,       # weight for voltage magnitude regularization
                   'weight_vang': 1e-3,      # weight for angle difference regularization
                   'weight_powers': 1e-3,    # weight for gen powers regularization
                   'weight_controls': 1e0,   # weight for control deviation penalty
                   'weight_var': 1e-5,       # weight for general variable regularization
                   'weight_redispatch': 1e0, # weight for gen real power redispatch deviation penalty
                   'v_min_clip': 0.5,        # lower v threshold for clipping
                   'v_max_clip': 1.5,        # upper v threshold for clipping
                   'v_limits': False,        # voltage magnitude limits (OPT only)
                   'Q_limits': True,         # flag for enforcing generator, VSC and FACTS reactive power limits
                   'Q_mode': 'regulating',   # reactive power mode: free, regulating
                   'shunt_limits': True,     # flag for enforcing switched shunt susceptance limits
                   'shunt_mode': 'locked',   # switched shunts mode: locked, free, regulating
                   'tap_limits': True,       # flag for enforcing transformer tap ratio limits
                   'tap_mode': 'locked',     # transformer tap ratio mode: locked, free, regulating
                   'phase_shift_limits': True,      # flag for enforcing transformer tap ratio limits
                   'phase_shift_mode': 'locked',    # transformer phase-shift mode: locked, free, regulating
                   'lock_vsc_P_dc': True,    # flag for locking vsc P dc
                   'lock_csc_P_dc': True,    # flag for locking csc P dc
                   'lock_csc_i_dc': True,    # flag for locking csc i dc
                   'vdep_loads': False,      # flag for modeling voltage dependent loads
                   'pvpq_start_k': 0,        # start iteration number for PVPQ switching heuristics
                   'vmin_thresh': 0.1,       # minimum voltage magnitude threshold
                   'gens_redispatch': False, # flag for allowing active power redispatch
                #    'shunts_round': False,     # flag for rounding discrete switched shunt susceptances (not supported yet)
                   'taps_round': False,      # flag for rounding discrete transformer tap ratios (NR only)
                   'v_mag_warm_ref': False,  # flag for using current v_mag as reference in v_mag regularization
                   'solver': 'nr',           # OPTALG optimization solver: augl, ipopt, nr, inlp
                   'tap_step': 0.5,          # tap ratio acceleration factor (NR only)
                   'shunt_step': 0.5,        # susceptance acceleration factor (NR only)
                   'dtap': 1e-4,             # tap ratio perturbation (NR only)
                   'dsus': 1e-4,             # susceptance perturbation (NR only)
                   'y_correction': True,     # admittance correction
                   'load_q_curtail': False,  # flag for allowing load Q to change (OPT only)
                   'P_transfer': False,      # flag for enabling inter-area transfer or not
                   'dsus': 1e-4,             # susceptance perturbation (NR only)
                   'v_min_2_ZIP': 0.85,      # minimum voltage threshold to convert loads to constant impedance
                   'loads_2_ZIP': True}      # Flag to convert loads to constant impedance if the voltage drops below
                                             # v_min_2_ZIP

    _parameters_augl = {'feastol': 1e-4,
                        'optol': 1e0,
                        'kappa': 1e-5,
                        'theta_max': 1e-6,
                        'sigma_init_max': 1e9}

    _parameters_ipopt = {}

    _parameters_inlp = {'feastol': 1e-4,
                        'optol': 1e0}

    _parameters_nr = {}

    def __init__(self):

        from optalg.opt_solver import OptSolverAugL, OptSolverIpopt, OptSolverNR, OptSolverINLP

        # Parent init
        PFmethod.__init__(self)

        # Solver params
        augl_params = OptSolverAugL.parameters.copy()
        augl_params.update(self._parameters_augl)   # overwrite defaults

        ipopt_params = OptSolverIpopt.parameters.copy()
        ipopt_params.update(self._parameters_ipopt) # overwrite defaults

        inlp_params = OptSolverINLP.parameters.copy()
        inlp_params.update(self._parameters_inlp)   # overwrite defaults

        nr_params = OptSolverNR.parameters.copy()
        nr_params.update(self._parameters_nr)       # overwrite defaults

        self._parameters = ACPF._parameters.copy()
        self._parameters['solver_parameters'] = {'augl': augl_params,
                                                 'ipopt': ipopt_params,
                                                 'nr': nr_params,
                                                 'inlp': inlp_params}

    def create_problem(self, net):

        solver_name = self._parameters['solver']

        if solver_name == 'nr':
            return self.create_problem_nr(net)
        else:
            return self.create_problem_opt(net)

    def create_problem_nr(self, net):

        import pfnet

        # Parameters
        params = self._parameters
        Q_mode = params['Q_mode']
        Q_limits = params['Q_limits']
        shunt_mode = params['shunt_mode']
        shunt_limits = params['shunt_limits']
        tap_mode = params['tap_mode']
        tap_limits = params['tap_limits']
        phase_shift_mode = params['phase_shift_mode']
        phase_shift_limits = params['phase_shift_limits']
        lock_vsc_P_dc = params['lock_vsc_P_dc']
        lock_csc_P_dc = params['lock_csc_P_dc']
        lock_csc_i_dc = params['lock_csc_i_dc']
        vdep_loads = params['vdep_loads']
        gens_redispatch = params['gens_redispatch']
        P_transfer = params['P_transfer']
        convert_loads_2_zip = params['loads_2_ZIP']
        y_correction = params['y_correction']

        # Check shunt options
        if shunt_mode not in [self.CONTROL_MODE_LOCKED,
                              self.CONTROL_MODE_REG]:
            raise ValueError('invalid shunts mode')
        if shunt_mode == self.CONTROL_MODE_REG and not shunt_limits:
            raise ValueError('unsupported shunts configuration')

        # Check tap options
        if tap_mode not in [self.CONTROL_MODE_LOCKED,
                            self.CONTROL_MODE_REG]:
            raise ValueError('invalid taps mode')
        if tap_mode == self.CONTROL_MODE_REG and not tap_limits:
            raise ValueError('unsupported taps configuration')

        # Check phase-shift options
        if phase_shift_mode not in [self.CONTROL_MODE_LOCKED,
                                    self.CONTROL_MODE_REG]:
            raise ValueError('invalid phase-shifter mode')
        if phase_shift_mode == self.CONTROL_MODE_REG and not phase_shift_limits:
            raise ValueError('unsupported phase-shifter configuration')

        # Check Q options
        if Q_mode != self.CONTROL_MODE_REG:
            raise ValueError('invalid reactive power mode')

        # Check other options
        if gens_redispatch:
            raise ValueError('generation redispatch not supported')
        if not lock_vsc_P_dc:
            raise ValueError('VSC P DC must be locked')
        if not lock_csc_P_dc:
            raise ValueError('CSC P DC must be locked')
        if not lock_csc_i_dc:
            raise ValueError('CSC i DC must be locked')

        # Clear flags
        net.clear_flags()

        # Buses
        net.set_flags('bus',
                      'variable',
                      'not slack',
                      'voltage angle')
        net.set_flags('bus',
                      'variable',
                      'any',
                      'voltage magnitude')

        # Generators
        net.set_flags('generator',
                      'variable',
                      'slack',
                      'active power')

        net.set_flags('generator',
                      'variable',
                      'regulator',
                      'reactive power')

        net.set_flags('generator',
                      'variable',
                      'fixed power factor',
                      'reactive power')

        # VSC HVDC
        net.set_flags('vsc converter',
                      'variable',
                      'any',
                      ['dc power', 'active power', 'reactive power'])

        # CSC HVDC
        net.set_flags('csc converter',
                      'variable',
                      'any',
                      ['dc power', 'active power', 'reactive power'])

        # DC buses
        net.set_flags('dc bus',
                      'variable',
                      'any',
                      'voltage')

        # FACTS
        net.set_flags('facts',
                      'variable',
                      'any',
                      'all')

        # Load
        if vdep_loads or convert_loads_2_zip:
            for load in net.loads:
                if load.is_in_service():
                    if convert_loads_2_zip:      # Consider all loads
                        net.set_flags_of_component(load,
                                                   'variable',
                                                   ['active power', 'reactive power'])
                    if load.is_voltage_dependent():  # Consider only voltage dependent loads
                        net.set_flags_of_component(load,
                                                   'variable',
                                                   ['active power', 'reactive power'])

        # Tap changer
        if tap_mode != self.CONTROL_MODE_LOCKED:
            net.set_flags('branch',
                          ['variable', 'fixed'],
                          'tap changer - v',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'tap changer - Q',
                          'tap ratio')
            if y_correction:
                net.set_flags('branch', 
                              'variable',
                              'y correction - ratio', 
                              'y scale') 
        
        # Phase-shifters
        if phase_shift_mode != self.CONTROL_MODE_LOCKED:
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            if y_correction:
                net.set_flags('branch', 
                              'variable',
                              'y correction - phase',
                              'y scale')

        # Switched shunts
        if shunt_mode != self.CONTROL_MODE_LOCKED:
            net.set_flags('shunt',
                          ['variable', 'fixed'],
                          'switching - v',
                          'susceptance')

        # Inter-area transfer
        if P_transfer:
            net.set_flags('transfer',
                          'variable',
                          'any',
                          'all')

        # Set up problem
        problem = pfnet.Problem(net)

        problem.add_constraint(pfnet.Constraint('AC power balance', net))
        problem.add_constraint(pfnet.Constraint('HVDC power balance', net))
        problem.add_constraint(pfnet.Constraint('generator active power participation', net))
        problem.add_constraint(pfnet.Constraint('variable fixing', net))
        problem.add_constraint(pfnet.Constraint('VSC converter equations', net))
        problem.add_constraint(pfnet.Constraint('CSC converter equations', net))
        problem.add_constraint(pfnet.Constraint('FACTS equations', net))
        problem.add_constraint(pfnet.Constraint('VSC DC voltage control', net))
        problem.add_constraint(pfnet.Constraint('CSC DC voltage control', net))
        problem.add_constraint(pfnet.Constraint('VSC DC power control', net))
        problem.add_constraint(pfnet.Constraint('CSC DC power control', net))
        problem.add_constraint(pfnet.Constraint('CSC DC current control', net))
        problem.add_constraint(pfnet.Constraint('fixed gen constant power factor', net))

        problem.add_constraint(pfnet.Constraint('PVPQ switching', net))
        problem.add_constraint(pfnet.Constraint('switching power factor regulation', net))
        problem.add_constraint(pfnet.Constraint('switching FACTS active power control', net))
        problem.add_constraint(pfnet.Constraint('switching FACTS reactive power control', net))
        problem.add_constraint(pfnet.Constraint('switching FACTS series voltage control', net))
        problem.add_constraint(pfnet.Constraint('switching FACTS series impedance control', net))

        if vdep_loads or convert_loads_2_zip:
            problem.add_constraint(pfnet.Constraint('load voltage dependence', net))

        if Q_limits:
            problem.add_heuristic(pfnet.Heuristic('PVPQ switching', net))
            problem.add_heuristic(pfnet.Heuristic('switching power factor regulation', net))

        if P_transfer:
            problem.add_constraint(pfnet.Constraint('switching area transfer regulation', net))
            problem.add_heuristic(pfnet.Heuristic('switching area transfer regulation', net))
        
        if tap_mode != self.CONTROL_MODE_LOCKED:
            problem.add_constraint(pfnet.Constraint('switching transformer q regulation', net))
            problem.add_heuristic(pfnet.Heuristic('switching transformer q regulation', net))

        if phase_shift_mode != self.CONTROL_MODE_LOCKED:
            problem.add_constraint(pfnet.Constraint('switching transformer p regulation', net))
            problem.add_heuristic(pfnet.Heuristic('switching transformer p regulation', net))

        if y_correction and (tap_mode != self.CONTROL_MODE_LOCKED or phase_shift_mode != self.CONTROL_MODE_LOCKED):
            problem.add_constraint(pfnet.Constraint('y correction table', net))

        problem.analyze()

        # Check
        if (problem.J.shape[0] + problem.A.shape[0] != problem.get_num_primal_variables()):
            raise PFmethodError_BadProblem()

        # Return
        return problem

    def create_problem_opt(self, net):

        import pfnet

        # Parameters
        params = self._parameters
        wm = params['weight_vmag']
        wa = params['weight_vang']
        wp = params['weight_powers']
        wc = params['weight_controls']
        wv = params['weight_var']
        wr = params['weight_redispatch']
        v_limits = params['v_limits']
        Q_mode = params['Q_mode']
        Q_limits = params['Q_limits']
        shunt_mode = params['shunt_mode']
        shunt_limits = params['shunt_limits']
        tap_mode = params['tap_mode']
        tap_limits = params['tap_limits']
        phase_shift_mode = params['phase_shift_mode']
        phase_shift_limits = params['phase_shift_limits']
        lock_vsc_P_dc = params['lock_vsc_P_dc']
        lock_csc_P_dc = params['lock_csc_P_dc']
        lock_csc_i_dc = params['lock_csc_i_dc']
        vdep_loads = params['vdep_loads']
        v_mag_warm_ref = params['v_mag_warm_ref']
        gens_redispatch = params['gens_redispatch']
        curtail_load_q = params['load_q_curtail']
        P_transfer = params['P_transfer']
        y_correction = params['y_correction']

        # Check shunt options
        if shunt_mode not in [self.CONTROL_MODE_LOCKED,
                              self.CONTROL_MODE_FREE,
                              self.CONTROL_MODE_REG]:
            raise ValueError('invalid shunts mode')
        if shunt_mode == self.CONTROL_MODE_REG and not shunt_limits:
            raise ValueError('unsupported shunts configuration')

        # Check tap options
        if tap_mode not in [self.CONTROL_MODE_LOCKED,
                            self.CONTROL_MODE_FREE,
                            self.CONTROL_MODE_REG]:
            raise ValueError('invalid taps mode')
        if tap_mode == self.CONTROL_MODE_REG and not tap_limits:
            raise ValueError('unsupported taps configuration')

        # Check tap options
        if phase_shift_mode not in [self.CONTROL_MODE_LOCKED,
                                    self.CONTROL_MODE_FREE,
                                    self.CONTROL_MODE_REG]:
            raise ValueError('invalid phase-shifter mode')
        if phase_shift_mode == self.CONTROL_MODE_REG and not phase_shift_limits:
            raise ValueError('unsupported phase-shifter configuration')

        # Check Q options
        if Q_mode not in [self.CONTROL_MODE_REG,
                          self.CONTROL_MODE_FREE]:
            raise ValueError('invalid reactive power mode')

        # Clear flags
        net.clear_flags()

        # Buses
        net.set_flags('bus',
                      'variable',
                      'not slack',
                      'voltage angle')
        net.set_flags('bus',
                      'variable',
                      'any',
                      'voltage magnitude')
        if Q_mode == self.CONTROL_MODE_REG and not Q_limits:
            net.set_flags('bus',
                          'fixed',
                          'v set regulated',
                          'voltage magnitude')
        if v_limits:
            net.set_flags('bus',
                          'bounded',
                          'any',
                          'voltage magnitude')

        # Generators
        if gens_redispatch:
            # Assume slack gens (excep renewables) are redispatchable
            net.set_flags('generator',
                          ['variable', 'bounded'],
                          'redispatchable',
                          'active power')
        else:
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
        net.set_flags('generator',
                      'variable',
                      'regulator',
                      'reactive power')
        net.set_flags('generator',
                      'variable',
                      'fixed power factor',
                      'reactive power')
        if Q_mode == self.CONTROL_MODE_FREE and Q_limits:
            net.set_flags('generator',
                          'bounded',
                          'regulator',
                          'reactive power')

        # Loads
        if vdep_loads:
            for load in net.loads:
                if load.is_voltage_dependent() and load.is_in_service():
                    net.set_flags_of_component(load,
                                               'variable',
                                               ['active power', 'reactive power'])

        if curtail_load_q:
            for load in net.loads:
                load.Q_min = np.minimum(load.Q, 0)
                load.Q_max = np.maximum(load.Q, 0)
                net.set_flags_of_component(load,
                                            ['variable','bounded'],
                                            'reactive power')
        # VSC HVDC
        net.set_flags('vsc converter',
                      'variable',
                      'any',
                      ['dc power', 'active power', 'reactive power'])
        if Q_mode == self.CONTROL_MODE_FREE and Q_limits:
            net.set_flags('vsc converter',
                          'bounded',
                          'any',
                          'reactive power')

        # CSC HVDC
        net.set_flags('csc converter',
                      'variable',
                      'any',
                      ['dc power', 'active power', 'reactive power'])

        # DC buses
        net.set_flags('dc bus',
                      'variable',
                      'any',
                      'voltage')

        # FACTS
        net.set_flags('facts',
                      'variable',
                      'any',
                      'all')
        if Q_mode == self.CONTROL_MODE_FREE and Q_limits:
            net.set_flags('facts',
                          'bounded',
                          'any',
                          'reactive power')

        # Trans tap mode
        if tap_mode != self.CONTROL_MODE_LOCKED:
            net.set_flags('branch',
                          'variable',
                          'tap changer - v',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'tap changer - Q',
                          'tap ratio')
            # Trans reg limit
            if tap_limits:
                net.set_flags('branch',
                            'bounded',
                            'tap changer - v',
                            'tap ratio')
                net.set_flags('branch',
                            'bounded',
                            'tap changer - Q',
                            'tap ratio')
            # Y correction
            if y_correction:
                net.set_flags('branch',
                              'variable',
                              'y correction - ratio',
                              ['tap ratio', 'y scale'])

        # Trans phase-shifter mode
        if phase_shift_mode != self.CONTROL_MODE_LOCKED:
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            # Reg limit
            if phase_shift_limits:
                net.set_flags('branch',
                              'bounded',
                              'phase shifter',
                              'phase shift')
                if net.get_num_asymmetric_phase_shifters() > 0:
                    net.set_flags('branch',
                                  ['variable', 'bounded'],
                                  'asymmetric phase shifter',
                                  'ratio')
            # Y correction
            if y_correction:
                net.set_flags('branch',
                              'variable',
                              'y correction - phase',
                              ['phase shift', 'y scale'])

        # Switched shunts
        if shunt_mode != self.CONTROL_MODE_LOCKED:
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
        if shunt_mode == self.CONTROL_MODE_FREE and shunt_limits:
            net.set_flags('shunt',
                          'bounded',
                          'switching - v',
                          'susceptance')

        # Inter-area transfer
        if P_transfer:
            buses = net.get_area_slack_buses()
            for bus in buses:
                for gen in bus.generators:
                    net.set_flags_of_component(gen,
                                               'variable',
                                               'active power')
            net.set_flags('transfer',
                          'variable',
                          'any',
                          'all')

        # Set up problem
        problem = pfnet.Problem(net)

        problem.add_constraint(pfnet.Constraint('AC power balance', net))
        problem.add_constraint(pfnet.Constraint('HVDC power balance', net))
        problem.add_constraint(pfnet.Constraint('generator active power participation', net))
        problem.add_constraint(pfnet.Constraint('VSC converter equations', net))
        problem.add_constraint(pfnet.Constraint('CSC converter equations', net))
        problem.add_constraint(pfnet.Constraint('FACTS equations', net))
        problem.add_constraint(pfnet.Constraint('VSC DC voltage control', net))
        problem.add_constraint(pfnet.Constraint('CSC DC voltage control', net))
        problem.add_constraint(pfnet.Constraint('power factor regulation', net))
        problem.add_constraint(pfnet.Constraint('switching FACTS series voltage control', net))
        problem.add_constraint(pfnet.Constraint('switching FACTS series impedance control', net))
        problem.add_constraint(pfnet.Constraint('fixed gen constant power factor', net))

        if lock_vsc_P_dc:
            problem.add_constraint(pfnet.Constraint('VSC DC power control', net))
        if lock_csc_P_dc:
            problem.add_constraint(pfnet.Constraint('CSC DC power control', net))
        if lock_csc_i_dc:
            problem.add_constraint(pfnet.Constraint('CSC DC current control', net))

        func = pfnet.Function('voltage magnitude regularization', wm/(net.get_num_buses(True)+1.), net)
        func.set_parameter('v_set_reference', not v_mag_warm_ref)
        problem.add_function(func)

        problem.add_function(pfnet.Function('variable regularization', wv/(net.num_vars+1.), net))
        problem.add_function(pfnet.Function('voltage angle regularization', wa/(net.get_num_buses(True)+1.), net))
        problem.add_function(pfnet.Function('generator powers regularization', wp/(net.get_num_generators(True)+1.), net))
        problem.add_function(pfnet.Function('VSC DC power control', wc/(net.get_num_vsc_converters(True)+1.), net))
        problem.add_function(pfnet.Function('CSC DC power control', wc/(net.get_num_csc_converters(True)+1.), net))
        problem.add_function(pfnet.Function('CSC DC current control', wc/(net.get_num_csc_converters(True)+1.), net))
        problem.add_function(pfnet.Function('FACTS active power control', wc/(net.get_num_facts(True)+1.), net))
        problem.add_function(pfnet.Function('FACTS reactive power control', wc/(net.get_num_facts(True)+1.), net))

        if gens_redispatch:
            problem.add_function(pfnet.Function('generation redispatch penalty', wr/(net.get_num_generators(True)+1.), net))

        if Q_mode == self.CONTROL_MODE_REG and Q_limits:
            problem.add_constraint(pfnet.Constraint('voltage set point regulation', net))

        if net.num_fixed > 0:
            problem.add_constraint(pfnet.Constraint('variable fixing', net))

        if tap_mode != self.CONTROL_MODE_LOCKED:
            problem.add_function(pfnet.Function('tap ratio regularization', wc/(net.get_num_tap_changers(True)+1.), net))
            if tap_mode == self.CONTROL_MODE_REG and tap_limits:
                problem.add_function(pfnet.Function('transformer Q regularization', wc/(net.get_num_tap_changers_Q(True)+1.), net))
                problem.add_constraint(pfnet.Constraint('voltage regulation by transformers', net))

        if phase_shift_mode != self.CONTROL_MODE_LOCKED:
            problem.add_function(pfnet.Function('phase shift regularization', wc/(net.get_num_phase_shifters(True)+1.), net))
            if phase_shift_mode == self.CONTROL_MODE_REG and phase_shift_limits:
                problem.add_function(pfnet.Function('transformer P regularization', wc/(net.get_num_phase_shifters(True)+1.), net))
                if net.get_num_asymmetric_phase_shifters() > 0:
                    problem.add_constraint(pfnet.Constraint('asymmetric transformer equations', net))

        if y_correction and (tap_mode != self.CONTROL_MODE_LOCKED or phase_shift_mode != self.CONTROL_MODE_LOCKED):
            problem.add_constraint(pfnet.Constraint("y correction table", net))

        if shunt_mode != self.CONTROL_MODE_LOCKED:
            problem.add_function(pfnet.Function('susceptance regularization', wc/(net.get_num_switched_v_shunts(True)+1.), net))
            if shunt_mode == self.CONTROL_MODE_REG and shunt_limits:
                problem.add_constraint(pfnet.Constraint('voltage regulation by shunts', net))

        if vdep_loads:
            problem.add_constraint(pfnet.Constraint('load voltage dependence', net))

        if P_transfer:
            problem.add_constraint(pfnet.Constraint('area transfer regulation', net))

        if net.num_bounded > 0:
            problem.add_constraint(pfnet.Constraint('variable bounds', net))

        # Analyze
        problem.analyze()

        # Return
        return problem

    def initialize_problem(self, net, update_net=False):
        # Parameters
        params = self._parameters
        v_min_clip = params['v_min_clip']
        v_max_clip = params['v_max_clip']

        if not update_net:
            # Copy network
            net = net.get_copy(merge_buses=True)

        # Clipping
        for bus in net.buses:
            bus.v_mag = np.minimum(np.maximum(bus.v_mag, v_min_clip), v_max_clip)

        # Problem
        t0 = time.time()
        problem = self.create_problem(net)
        problem_time = time.time()-t0
        self.set_problem_time(problem_time)

        return net, problem

    def solve_problem(self, net, problem, save_problem=False, update_net=False):
        """
        solve the Optimization problem
        """

        from optalg.opt_solver import OptSolverError, OptTermination, OptCallback
        from optalg.opt_solver import OptSolverAugL, OptSolverIpopt, OptSolverNR, OptSolverINLP

        # Parameters
        params = self._parameters
        Q_mode = params['Q_mode']
        shunt_mode = params['shunt_mode']
        tap_mode = params['tap_mode']
        vmin_thresh = params['vmin_thresh']
        solver_name = params['solver']
        solver_params = params['solver_parameters']
        quiet = solver_params[solver_name]['quiet']
        convert_loads_2_zip = params['loads_2_ZIP']
        vmin_to_zip = params['v_min_2_ZIP']

        # Opt solver
        if solver_name == 'augl':
            solver = OptSolverAugL()
        elif solver_name == 'ipopt':
            solver = OptSolverIpopt()
        elif solver_name == 'inlp':
            solver = OptSolverINLP()
        elif solver_name == 'nr':
            solver = OptSolverNR()
        else:
            raise PFmethodError_BadOptSolver()
        solver.set_parameters(solver_params[solver_name])

        # Callbacks
        def c1(s):
            """Apply Transformer Voltage Regulation"""
            if (s.k != 0 and params['tap_limits'] and tap_mode == self.CONTROL_MODE_REG and
                norm(s.problem.f, np.inf) < 100.*solver_params['nr']['feastol']):
                try:
                    self.apply_tran_v_regulation(s)
                    if params['taps_round']:
                        self.apply_tran_tap_rounding(s)
                except Exception as e:
                    raise PFmethodError_TranVReg(e)
                
        def c2(s):
            """Apply Shunt Voltage Regulation"""
            if (s.k != 0 and params['shunt_limits'] and shunt_mode == self.CONTROL_MODE_REG and
                norm(s.problem.f, np.inf) < 100.*solver_params['nr']['feastol']):
                try:
                    self.apply_shunt_v_regulation(s)
                except Exception as e:
                    raise PFmethodError_ShuntVReg(e)

        def c3(s):
            """PV-PQ start at given iteration"""
            if (s.k >= params['pvpq_start_k'] and params['Q_limits'] and Q_mode == self.CONTROL_MODE_REG):
                prob = s.problem.wrapped_problem
                prob.apply_heuristics(s.x)
                s.problem.A = prob.A
                s.problem.b = prob.b

        def c4(s):
            net = s.problem.wrapped_problem.network
            if np.min(s.problem.wrapped_problem.network.bus_v_min) <= vmin_to_zip:
                for lod in net.loads:
                    if lod.is_in_service():
                        bus_v = s.get_primal_variables()[lod.bus.index_v_mag]
                        if isinstance(bus_v, Iterable):
                            i = np.argmin(bus_v)
                            if bus_v[i] <= vmin_to_zip:
                                # Move cp&ci to cg
                                lod.comp_cg = lod.comp_cp[i] + lod.comp_ci[i] + lod.comp_cg
                                lod.comp_cp = [0.0]*len(bus_v)
                                lod.comp_ci = [0.0]*len(bus_v)
                                # Move cq&cj to cb
                                lod.comp_cb = lod.comp_cb - lod.comp_cq[i] - lod.comp_cj[i]
                                lod.comp_cq = [0.0]*len(bus_v)
                                lod.comp_cj = [0.0]*len(bus_v)
                        else:
                            if bus_v <= vmin_to_zip:
                                # Move cp&ci to cg
                                lod.comp_cg = lod.comp_cp + lod.comp_ci + lod.comp_cg
                                lod.comp_cp = 0.0
                                lod.comp_ci = 0.0
                                # Move cq&cj to cb
                                lod.comp_cb = lod.comp_cb - lod.comp_cq - lod.comp_cj
                                lod.comp_cq = 0.0
                                lod.comp_cj = 0.0

        if solver_name == 'nr':
            solver.add_callback(OptCallback(c1))
            solver.add_callback(OptCallback(c2))
            solver.add_callback(OptCallback(c3))
            if convert_loads_2_zip:
                solver.add_callback(OptCallback(c4))

        # Termination
        def t1(s):
            if np.min(s.problem.wrapped_problem.network.bus_v_min) < vmin_thresh:
                return True
            else:
                return False
        solver.add_termination(OptTermination(t1, 'low voltage'))

        # Info printer
        info_printer = self.get_info_printer()
        solver.set_info_printer(info_printer)

        # Solve
        update = True
        t0 = time.time()
        try:
            solver.solve(problem)
        except OptSolverError as e:
            raise PFmethodError_SolverError(e)
        except Exception as e:
            update = False
            raise e
        finally:

            # Update network
            if update:
                if update_net and solver.get_status() == 'error':
                    pass
                else:
                    net.set_var_values(solver.get_primal_variables()[:net.num_vars])
                    net.update_properties()
                    net.clear_sensitivities()
                    if solver_name != 'nr':
                        problem.store_sensitivities(*solver.get_dual_variables())

            if not quiet and convert_loads_2_zip and solver.problem.wrapped_problem.network.bus_v_min <= vmin_to_zip:
                for lod in net.loads:
                    if lod.is_in_service():
                        bus_v = solver.get_primal_variables()[lod.bus.index_v_mag]
                        if bus_v <= vmin_to_zip:
                            print(f"  Converted load {lod.name}, at bus {lod.bus.number} with v_mag={bus_v:>1.4f} to ZIP load")

            # Save results
            self.set_solver_name(solver_name)
            self.set_solver_status(solver.get_status())
            self.set_solver_message(solver.get_error_msg())
            self.set_solver_iterations(solver.get_iterations())
            self.set_solver_time(time.time()-t0)
            self.set_solver_primal_variables(solver.get_primal_variables())
            self.set_solver_dual_variables(solver.get_dual_variables())
            self.set_problem(problem if save_problem else None)
            self.set_network_snapshot(net)

    def solve(self, net, save_problem=False, update_net=False):
        """
        Solve the network

        Parameters
        -----------
        net: |Network|
        save_problem: bool
        update_net: bool (update network with the results if solved successfully)
        """

        # Step 1: initialize optimization Problem
        net, problem = self.initialize_problem(net, update_net)

        # step 2: Solve the initialized optimization Problem
        self.solve_problem(net, problem, save_problem, update_net)

    def get_info_printer(self):

        # Parameters
        solver_name = self._parameters['solver']

        # Define
        def info_printer(solver,header):
            net = solver.problem.wrapped_problem.network
            if header:
                print('{0:^5}'.format('vmax'), end=' ')
                print('{0:^5}'.format('vmin'), end=' ')
                print('{0:^8}'.format('gvdev'), end=' ')
                print('{0:^8}'.format('gQvio'))
            else:
                print('{0:^5.2f}'.format(np.average(net.bus_v_max)), end=' ')
                print('{0:^5.2f}'.format(np.average(net.bus_v_min)), end=' ')
                print('{0:^8.1e}'.format(np.average(net.gen_v_dev)), end=' ')
                print('{0:^8.1e}'.format(np.average(net.gen_Q_vio)))

        # Return
        return info_printer

    def apply_shunt_v_regulation(self,solver):

        # Local variables
        dsus = self._parameters['dsus']
        step = self._parameters['shunt_step']
        p = solver.problem.wrapped_problem
        net = p.network
        x = solver.x
        eps = 1e-8

        # Fix constraints
        c = p.find_constraint('variable fixing')
        A = c.A
        b = c.b

        # Rhs
        rhs = np.hstack((np.zeros(p.f.size),np.zeros(p.b.size)))

        # Offset
        offset = 0
        for c in p.constraints:
            if c.name == 'variable fixing':
                break
            else:
                offset += c.A.shape[0]

        # Violation check
        for i in range(net.num_buses):

            bus = net.get_bus(i)

            if bus.is_regulated_by_shunt(True) and not bus.is_slack():

                assert(bus.has_flags('variable','voltage magnitude'))

                for t in range(net.num_periods):

                    v = x[bus.index_v_mag[t]]
                    vmax = bus.v_max_reg
                    vmin = bus.v_min_reg

                    assert(len(bus.reg_shunts) > 0)
                    assert(vmax >= vmin)

                    # Violation
                    if v > vmax or v < vmin:

                        for reg_shunt in bus.reg_shunts:

                            if not reg_shunt.is_in_service():
                                continue

                            assert(reg_shunt.has_flags('variable','susceptance'))

                            s = x[reg_shunt.index_b[t]]
                            smax = reg_shunt.b_max
                            smin = reg_shunt.b_min
                            assert(smin <= smax)

                            # Fix constr index
                            k = int(np.where(A.col == reg_shunt.index_b[t])[0])
                            i = A.row[k]

                            assert(np.abs(b[i]-x[reg_shunt.index_b[t]]) < eps)
                            assert(A.data[k] == 1.)

                            # Sensitivity
                            assert(rhs[p.f.size+offset+i] == 0.)
                            rhs[p.f.size+offset+i] = dsus
                            dx = solver.linsolver.solve(rhs)
                            dv = dx[bus.index_v_mag[t]]
                            dvds = dv/dsus
                            rhs[p.f.size+offset+i] = 0.

                            # Adjustment
                            dv = (vmax+vmin)/2.-v
                            ds = step*dv/dvds if dvds != 0. else 0.
                            snew = np.maximum(np.minimum(s+ds,smax),smin)
                            x[reg_shunt.index_b[t]] = snew
                            b[i] = snew
                            if np.abs(snew-s) > eps:
                                break

        # Update
        solver.func(x)
        p.update_lin()
        solver.problem.A = p.A
        solver.problem.b = p.b

    def apply_tran_v_regulation(self,solver):

        # Local variables
        dtap = self._parameters['dtap']
        step = self._parameters['tap_step']
        p = solver.problem.wrapped_problem
        net = p.network
        x = solver.x
        eps = 1e-8

        # Fix constraints
        c = p.find_constraint('variable fixing')
        A = c.A
        b = c.b

        # Rhs
        rhs = np.hstack((np.zeros(p.f.size),np.zeros(p.b.size)))

        # Offset
        offset = 0
        for c in p.constraints:
            if c.name == 'variable fixing':
                break
            else:
                offset += c.A.shape[0]

        # Violation check
        for i in range(net.num_buses):

            bus = net.get_bus(i)

            if bus.is_regulated_by_tran(True) and not bus.is_slack():

                assert(bus.has_flags('variable','voltage magnitude'))
                for tau in range(net.num_periods):

                    v = x[bus.index_v_mag[tau]]
                    vmax = bus.v_max_reg
                    vmin = bus.v_min_reg

                    assert(len(bus.reg_trans) > 0)
                    assert(vmax > vmin)

                    # Violation
                    if v > vmax or v < vmin:

                        for reg_tran in bus.reg_trans:

                            if not reg_tran.is_in_service():
                                continue

                            assert(reg_tran.has_flags('variable','tap ratio'))

                            t = x[reg_tran.index_ratio[tau]]
                            tmax = reg_tran.ratio_max
                            tmin = reg_tran.ratio_min
                            assert(tmax >= tmin)

                            # Fix constr index
                            k = int(np.where(A.col == reg_tran.index_ratio[tau])[0])
                            i = A.row[k]
                            assert(np.abs(b[i]-x[reg_tran.index_ratio[tau]]) < eps)
                            assert(A.data[k] == 1.)

                            # Sensitivity
                            assert(rhs[p.f.size+offset+i] == 0.)
                            rhs[p.f.size+offset+i] = dtap
                            dx = solver.linsolver.solve(rhs)
                            dv = dx[bus.index_v_mag[tau]]
                            dvdt = dv/dtap
                            rhs[p.f.size+offset+i] = 0.

                            # Adjustment
                            dv = (vmax+vmin)/2.-v
                            dt = step*dv/dvdt if dvdt != 0. else 0.
                            tnew = np.maximum(np.minimum(t+dt,tmax),tmin)
                            x[reg_tran.index_ratio[tau]] = tnew
                            b[i] = tnew
                            if np.abs(tnew-t) > eps:
                                break

        # Update
        solver.func(x)
        p.update_lin()
        solver.problem.A = p.A
        solver.problem.b = p.b

    def apply_tran_tap_rounding(self, solver):
        """Apply transformer voltage regualtion using descrete mode."""

        # Local variables
        p = solver.problem.wrapped_problem
        net = p.network
        x = solver.x
        eps = 1e-8

        # Fix constraints
        c = p.find_constraint('variable fixing')
        A = c.A
        b = c.b

        for i in range(net.num_branches):

            br = net.get_branch(i)

            if br.is_in_service() and br.is_tap_changer():

                assert(br.reg_bus.has_flags('variable', 'voltage magnitude'))
                for tau in range(net.num_periods):

                    if(br.has_flags('variable', 'tap ratio')):
                        # Ratio is variable
                        cur_ratio = x[br.index_ratio[tau]]
                        if cur_ratio > br.ratio_max:
                            x[br.index_ratio[tau]] = br.ratio_max
                        elif cur_ratio < br.ratio_min:
                            x[br.index_ratio[tau]] = br.ratio_min
                        else:
                            # Find closest ratio
                            dratio = (br.ratio_max - br.ratio_min) / (br.num_ratios-1)
                            ptn_ratios = np.arange(0, br.num_ratios)*dratio + br.ratio_min
                            i_close = np.argmin(np.abs(ptn_ratios - cur_ratio))
                            new_ratio = ptn_ratios[i_close]

                            # Fix constr index
                            k = int(np.where(A.col == br.index_ratio[tau])[0])
                            ik = A.row[k]
                            assert(np.abs(b[ik]-x[br.index_ratio[tau]]) < eps)
                            assert(A.data[k] == 1.)

                            # Update var and constraint
                            x[br.index_ratio[tau]] = new_ratio
                            b[ik] = new_ratio
                    else:
                        # Ratio not variable
                        if br.ratio > br.ratio_max:
                            br.ratio = br.ratio_max
                        elif br.ratio < br.ratio_min:
                            br.ratio = br.ratio_min
                        else:
                            # Find closest ratio
                            dratio = (br.ratio_max - br.ratio_min) / (br.num_ratios-1)
                            ptn_ratios = np.arange(0, br.num_ratios)*dratio + br.ratio_min
                            i_close = np.argmin(np.abs(ptn_ratios - br.ratio))
                            br.ratio = ptn_ratios[i_close]
                        
        # Update
        solver.func(x)
        p.update_lin()
        solver.problem.A = p.A
        solver.problem.b = p.b
