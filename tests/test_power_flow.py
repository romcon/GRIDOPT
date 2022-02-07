#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015, Tomas Tinoco De Rubira.         #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

from __future__ import print_function
import os
import copy
import unittest
import numpy as np
from . import utils
from numpy.linalg import norm


import optalg
import pfnet as pf
import gridopt as gopt


class TestPowerFlow(unittest.TestCase):

    def setUp(self):

        pass

    def test_ACPF_opt_controls_support(self):

        case = os.path.join('tests', 'resources', 'cases', 'aesoSL2014.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        raise unittest.SkipTest('test incomplete')

    def test_ACPF_nr_controls_support(self):

        case = os.path.join('tests', 'resources', 'cases', 'aesoSL2014.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)

        # Default
        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr'})
        self.assertTrue(method._parameters['Q_limits'])
        self.assertTrue(method._parameters['tap_limits'])
        self.assertTrue(method._parameters['shunt_limits'])
        self.assertEqual(method._parameters['Q_mode'], 'regulating')
        self.assertEqual(method._parameters['tap_mode'], 'locked')
        self.assertEqual(method._parameters['shunt_mode'], 'locked')
        method.solve(net)
        self.assertEqual(method.get_results()['solver status'], 'solved')

        # Q
        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'Q_mode': 'regulating',
                               'Q_limits': False})
        method.solve(net)
        self.assertEqual(method.get_results()['solver status'], 'solved')

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'Q_mode': 'free',
                               'Q_limits': False})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'Q_mode': 'free',
                               'Q_limits': True})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'Q_mode': 'locked',
                               'Q_limits': True})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'Q_mode': 'locked',
                               'Q_limits': False})
        self.assertRaises(ValueError, method.solve, net)

        # Shunts
        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'shunt_mode': 'locked',
                               'shunt_limits': False})
        method.solve(net)
        self.assertEqual(method.get_results()['solver status'], 'solved')

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'shunt_mode': 'regulating',
                               'shunt_limits': False})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'shunt_mode': 'regulating',
                               'shunt_limits': True})
        method.solve(net)
        self.assertEqual(method.get_results()['solver status'], 'solved')

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'shunt_mode': 'free',
                               'shunt_limits': False})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'shunt_mode': 'free',
                               'shunt_limits': True})
        self.assertRaises(ValueError, method.solve, net)

        # Taps
        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'tap_mode': 'locked',
                               'tap_limits': False})
        method.solve(net)
        self.assertEqual(method.get_results()['solver status'], 'solved')

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'tap_mode': 'regulating',
                               'tap_limits': False})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'tap_mode': 'regulating',
                               'tap_limits': True})
        method.solve(net)
        self.assertEqual(method.get_results()['solver status'], 'solved')

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'tap_mode': 'free',
                               'tap_limits': False})
        self.assertRaises(ValueError, method.solve, net)

        method = gopt.power_flow.ACPF()
        method.set_parameters({'quiet': True,
                               'solver': 'nr',
                               'tap_mode': 'free',
                               'tap_limits': True})
        self.assertRaises(ValueError, method.solve, net)

    def test_ACPF_keep_all(self):
        
        for case in utils.test_cases:

            if os.path.splitext(case)[1] == '.raw':

                parser = pf.ParserRAW()
                parser.set('output_level', 0)
                net = parser.parse(case)

                parser.set('keep_all_out_of_service', True)
                net_oos = parser.parse(case)

                if (net.num_buses == net_oos.num_buses and
                    net.num_generators == net_oos.num_generators and
                    net.num_loads == net_oos.num_loads and
                    net.num_shunts == net_oos.num_shunts and
                    net.num_branches == net_oos.num_branches):
                    continue

                method = gopt.power_flow.ACPF()
                method.set_parameters({'solver': 'augl', 'quiet': True, 'v_mag_warm_ref': True})

                method.solve(net)
                r = method.get_results()

                method.solve(net_oos)
                r_oos = method.get_results()

                self.assertEqual(r['solver status'], r_oos['solver status'])
                self.assertEqual(r['solver iterations'], r_oos['solver iterations'])
                self.assertEqual(r['network snapshot'].bus_v_max, r_oos['network snapshot'].bus_v_max)
                self.assertEqual(r['network snapshot'].bus_v_min, r_oos['network snapshot'].bus_v_min)
                self.assertEqual(r['network snapshot'].bus_P_mis, r_oos['network snapshot'].bus_P_mis)
                self.assertEqual(r['network snapshot'].bus_Q_mis, r_oos['network snapshot'].bus_Q_mis)

                self.assertRaises(AssertionError, pf.tests.utils.compare_networks, self, r['network snapshot'], r_oos['network snapshot'])

    def test_ACPF_parameters(self):

        acpf = gopt.power_flow.ACPF()

        # Set parameters
        solvers = ['augl', 'nr', 'inlp', 'ipopt']
        acpf.set_parameters({
                            'weight_vmag': 1.1,
                            'weight_vang': 1.2,
                            'weight_powers': 1.3,
                            'weight_controls': 1.4,
                            'weight_var': 1.5,
                            'weight_redispatch': 1.6,
                            'v_min_clip': 0.2,
                            'v_max_clip': 1.8,
                            'v_limits': True,
                            'Q_mode': 'regulating',
                            'Q_limits': True,
                            'shunt_mode': 'regulating',
                            'shunt_limits': True,
                            'tap_mode': 'regulating',
                            'tap_limits': True,
                            'lock_vsc_P_dc': False,
                            'lock_csc_P_dc': False,
                            'lock_csc_i_dc': False,
                            'vdep_loads': True,
                            'pvpq_start_k': 1,
                            'vmin_thresh': 0.25,
                            'v_mag_warm_ref': True,
                            'gens_redispatch': True,
                            'load_q_curtail': True,
                            'tap_step': 0.1,
                            'shunt_step': 0.1,
                            'dtap': 1e-6,
                            'dsus': 1e-6,
                            'feastol': 1e-2,
                            'optol': 2e-2,
                            'maxiter': 2,
                            'kappa': 1e-2,
                            'theta_max': 1e-4,
                            'sigma_init_max': 1e8})

        # Check exception for invalid param
        self.assertRaises(gopt.power_flow.method_error.PFmethodError_BadParams,
                          acpf.set_parameters,{'foo' : 'bar'})

        # Get parameters
        params = acpf.get_parameters()

        # Check that set_parameters worked
        self.assertEqual(params['weight_vmag'], 1.1)
        self.assertEqual(params['weight_vang'], 1.2)
        self.assertEqual(params['weight_powers'], 1.3)
        self.assertEqual(params['weight_controls'], 1.4)
        self.assertEqual(params['weight_var'], 1.5)
        self.assertEqual(params['weight_redispatch'], 1.6)
        self.assertEqual(params['v_min_clip'], 0.2)
        self.assertEqual(params['v_max_clip'], 1.8)    
        self.assertEqual(params['v_limits'], True)
        self.assertEqual(params['Q_mode'], 'regulating')
        self.assertEqual(params['Q_limits'], True)
        self.assertEqual(params['shunt_mode'], 'regulating')
        self.assertEqual(params['shunt_limits'], True)
        self.assertEqual(params['tap_mode'], 'regulating')
        self.assertEqual(params['tap_limits'], True)
        self.assertEqual(params['lock_vsc_P_dc'], False)
        self.assertEqual(params['lock_csc_P_dc'], False)
        self.assertEqual(params['lock_csc_i_dc'], False)
        self.assertEqual(params['vdep_loads'], True)
        self.assertEqual(params['pvpq_start_k'], 1)
        self.assertEqual(params['vmin_thresh'], 0.25)
        self.assertEqual(params['v_mag_warm_ref'], True)
        self.assertEqual(params['gens_redispatch'], True)
        self.assertEqual(params['load_q_curtail'], True)
        self.assertEqual(params['tap_step'], 0.1)
        self.assertEqual(params['shunt_step'], 0.1)
        self.assertEqual(params['dtap'], 1e-6)
        self.assertEqual(params['dsus'], 1e-6)

        self.assertEqual(params['solver_parameters']['nr']['maxiter'], 2)
        self.assertEqual(params['solver_parameters']['nr']['feastol'], 1e-2)

        self.assertEqual(params['solver_parameters']['augl']['maxiter'], 2)
        self.assertEqual(params['solver_parameters']['augl']['feastol'], 1e-2)
        self.assertEqual(params['solver_parameters']['augl']['optol'], 2e-2)
        self.assertEqual(params['solver_parameters']['augl']['kappa'], 1e-2)
        self.assertEqual(params['solver_parameters']['augl']['theta_max'], 1e-4)
        self.assertEqual(params['solver_parameters']['augl']['sigma_init_max'], 1e8)

        # self.assertEqual(params['solver_parameters']['ipopt']['maxiter'], 3)  # uses max_iter

        self.assertEqual(params['solver_parameters']['inlp']['maxiter'], 2)
        self.assertEqual(params['solver_parameters']['inlp']['optol'], 2e-2)


        # Run short test with each solver to verify it runs
        case = os.path.join('tests', 'resources', 'cases', 'ieee300.raw')
        if not os.path.isfile(case):
            return

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)
        for sol in solvers:
            if sol == 'nr':
                acpf.set_parameters({'gens_redispatch': False, 
                                     'load_q_curtail': False,
                                     'lock_vsc_P_dc': True, 
                                     'lock_csc_P_dc': True, 
                                     'lock_csc_i_dc': True})
            acpf.set_parameters({'solver': sol, 'quiet': True})
            self.assertEqual(acpf.get_parameters()['solver'], sol)
            self.assertRaises(gopt.power_flow.method_error.PFmethodError_SolverError, acpf.solve, net, {'quiet': True})
            
    def test_ACOPF_parameters(self):

        acopf = gopt.power_flow.ACOPF()

        # Set parameters
        acopf.set_parameters({'weight_cost': 1.1,
                              'weight_vmag' : 1.2,
                              'weight_vang' : 1.3,
                              'weight_pq' : 1.4,
                              'weight_t' : 1.5,
                              'weight_b' : 1.6,
                              'thermal_limits' : True,
                              'vmin_thresh' : 0.123,
                              'solver' : 'inlp',
                              'beta_small' : 8.9,
                              'hessian_approximation' : 'test',
                              'maxiter': 432})

        # Check exception for invalid param
        self.assertRaises(gopt.power_flow.method_error.PFmethodError_BadParams,
                          acopf.set_parameters,{'foo' : 'bar'})

        # Get parameters
        params = acopf.get_parameters()

        # Check that set_parameters worked
        self.assertEqual(params['weight_cost'],1.1)
        self.assertEqual(params['weight_vmag'],1.2)
        self.assertEqual(params['weight_vang'],1.3)
        self.assertEqual(params['weight_pq'],1.4)
        self.assertEqual(params['weight_t'],1.5)
        self.assertEqual(params['weight_b'],1.6)
        self.assertEqual(params['thermal_limits'],True)
        self.assertEqual(params['vmin_thresh'],0.123)
        self.assertEqual(params['solver'],'inlp')
        self.assertEqual(params['solver_parameters']['augl']['beta_small'],8.9)
        self.assertEqual(params['solver_parameters']['augl']['maxiter'],432)
        self.assertEqual(params['solver_parameters']['inlp']['maxiter'],432)
        self.assertEqual(params['solver_parameters']['ipopt']['hessian_approximation'],'test')

        # Make a deep copy
        new_params = copy.deepcopy(params)

        # Set manually solver parametres
        new_params['solver_parameters']['augl']['beta_small'] = 1.333e-5
        new_params['solver_parameters']['inlp']['maxiter'] = 555
        new_params['solver_parameters']['ipopt']['hessian_approximation'] = 'new test'

        # Check that this is a separate params dictionary
        self.assertNotEqual(new_params['solver_parameters']['augl']['beta_small'],
                            params['solver_parameters']['augl']['beta_small'])
        self.assertNotEqual(new_params['solver_parameters']['inlp']['maxiter'],
                            params['solver_parameters']['inlp']['maxiter'])
        self.assertNotEqual(new_params['solver_parameters']['ipopt']['hessian_approximation'],
                            params['solver_parameters']['ipopt']['hessian_approximation'])

        # Test setting parameters that specify solver parameters
        acopf.set_parameters(new_params)

        # Check that setting solver parameters worked
        self.assertEqual(new_params['solver_parameters']['augl']['beta_small'],
                         params['solver_parameters']['augl']['beta_small'])
        self.assertEqual(new_params['solver_parameters']['inlp']['maxiter'],
                         params['solver_parameters']['inlp']['maxiter'])
        self.assertEqual(new_params['solver_parameters']['ipopt']['hessian_approximation'],
                         params['solver_parameters']['ipopt']['hessian_approximation'])

    def test_DCPF(self):

        for case in utils.test_cases:

            method = gopt.power_flow.new_method('DCPF')
            self.assertTrue(isinstance(method, gopt.power_flow.DCPF))

            method.set_parameters({'solver' : 'superlu'})
            self.assertTrue('solver_parameters' in method.get_parameters().keys())

            parser = pf.Parser(case)
            parser.set('output_level', 0)
            net = parser.parse(case)

            method.solve(net)

            results = method.get_results()

            self.assertEqual(results['solver status'], 'solved')
            self.assertTrue(results['solver name'] in ['mumps','superlu'])
            self.assertTrue(isinstance(results['network snapshot'], pf.Network))

    def test_ACPF_with_gen_outages(self):

        case = os.path.join('tests', 'resources', 'cases', 'ieee25.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)

        method = gopt.power_flow.new_method('ACPF')

        for solver in ['nr', 'augl']:
            method.set_parameters(params={'solver': solver,
                                          'quiet': True})
            method.solve(net)

            self.assertEqual(method.get_results()['solver status'], 'solved')
            net1 = method.get_results()['network snapshot']

            for bus in net.buses:
                if bus.is_v_set_regulated():
                    reg_gens = [g.index for g in bus.reg_generators]
                    gen = bus.reg_generators[-1]
                    gen.in_service = False
                    method.solve(net)
                    self.assertEqual(method.get_results()['solver status'], 'solved')
                    net2 = method.get_results()['network snapshot']
                    self.assertEqual(net.get_generator(gen.index).Q,
                                     net2.get_generator(gen.index).Q)
                    self.assertNotEqual(net.get_generator(gen.index).Q,
                                        net1.get_generator(gen.index).Q)
                    gen.in_service = True

    def test_ACPF_with_branch_outages(self):

        case = os.path.join('tests', 'resources', 'cases', 'ieee300.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)

        method = gopt.power_flow.new_method('ACPF')

        for solver in ['nr','augl']:
            method.set_parameters(params={'solver': solver,
                                          'tap_mode': 'regulating',
                                          'quiet': True})
            method.solve(net)

            self.assertEqual(method.get_results()['solver status'], 'solved')
            net1 = method.get_results()['network snapshot']

            tested = False
            for bus in net.buses:
                if bus.is_regulated_by_tran() and bus.number == 1:
                    reg_trans = [br.index for br in bus.reg_trans]
                    tran = bus.reg_trans[-1]
                    tran.in_service = False
                    method.solve(net)
                    self.assertEqual(method.get_results()['solver status'], 'solved')
                    net2 = method.get_results()['network snapshot']
                    self.assertFalse(net.get_branch(tran.index).has_flags('variable', 'tap ratio'))
                    self.assertTrue(net1.get_branch(tran.index).has_flags('variable', 'tap ratio'))
                    self.assertFalse(net2.get_branch(tran.index).has_flags('variable', 'tap ratio'))
                    self.assertNotEqual(net1.num_vars, net2.num_vars)
                    self.assertGreater(net1.num_vars, net2.num_vars)
                    tran.in_service = True
                    tested = True
            self.assertTrue(tested)

    def test_ACPF_nr_heuristics(self):

        skipcases = ['aesoSL2014.raw', 'case2869.mat', 'case9241.mat', 'case32.art',
                     'ieee59_convert_to_zil_to_solve.raw']

        for case in utils.test_cases:

            if case.split(os.sep)[-1] in skipcases:
                continue

            parser = pf.Parser(case)
            parser.set('output_level', 0)
            net = parser.parse(case)

            method = gopt.power_flow.new_method('ACPF')
            method.set_parameters(params={'solver': 'nr',
                                          'quiet': True})
            method.solve(net)

            results = method.get_results()

            net_snap = results['network snapshot']
            self.assertLess(np.abs(net_snap.bus_P_mis), 1e-2) # MW
            self.assertLess(np.abs(net_snap.bus_Q_mis), 1e-2) # MVAr

            eps = 1e-4
            for bus in net_snap.buses:
                if bus.is_regulated_by_gen() and not bus.is_slack():
                    for gen in bus.reg_generators:
                        self.assertLessEqual(gen.Q, gen.Q_max+eps)
                        self.assertGreaterEqual(gen.Q, gen.Q_min-eps)
                    if np.abs(bus.v_mag-bus.v_set) < eps: # v at set
                        Qtotal = 0.
                        norm = 0.
                        for gen in bus.reg_generators:
                            if np.abs(gen.Q-gen.Q_max) > eps and np.abs(gen.Q-gen.Q_min) > eps:
                                Qtotal += gen.Q
                                norm += gen.Q_par
                        for gen in bus.reg_generators:
                            if np.abs(gen.Q-gen.Q_max) > eps and np.abs(gen.Q-gen.Q_min) > eps:
                                self.assertLess(np.abs(gen.Q-gen.Q_par*Qtotal/norm), eps)
                    else: # v not at set
                        num = 0
                        for gen in bus.reg_generators:
                            if np.abs(gen.Q-gen.Q_max) <= eps or np.abs(gen.Q-gen.Q_min) <= eps:
                                num += 1
                        self.assertGreaterEqual(num, 1)

            method.update_network(net)
            self.assertEqual(net.bus_P_mis, net_snap.bus_P_mis)
            self.assertEqual(net.bus_Q_mis, net_snap.bus_Q_mis)
            self.assertLess(np.abs(net.bus_P_mis), 1e-2) # MW
            self.assertLess(np.abs(net.bus_Q_mis), 1e-2) # MVAr

    def test_ACPF_solutions(self):

        print('')

        T = 2

        sol_types = {'sol1': 'no--controls',
                     'sol2': 'gen-controls',
                     'sol3': 'all-controls'}

        skipcases = ['aesoSL2014.raw', 'case2869.mat', 'case9241.mat', 'case32.art',
                     'ieee59_convert_to_zil_to_solve.raw']

        for case in utils.test_cases:

            if case.split(os.sep)[-1] in skipcases:
                continue

            for sol in list(sol_types.keys()):
                for solver in ['nr', 'augl', 'ipopt', 'inlp']:

                    method = gopt.power_flow.new_method('ACPF')
                    method.set_parameters(params={'solver': solver})

                    parser = pf.Parser(case)
                    parser.set('output_level', 0)
                    net = parser.parse(case)
                    parser = pf.Parser(case)
                    parser.set('output_level', 0)
                    netMP = parser.parse(case, T)

                    self.assertEqual(net.num_periods,1)
                    self.assertEqual(netMP.num_periods,T)

                    # Only small
                    if net.num_buses > 4000:
                        continue

                    sol_file = utils.get_pf_solution_file(case, utils.DIR_PFSOL, sol)
                    sol_data = utils.read_pf_solution_file(sol_file)

                    # Set parameters
                    if sol == 'sol1':
                        method.set_parameters({'Q_limits': False})
                    elif sol == 'sol2':
                        pass # defaults
                    elif sol == 'sol3':
                        method.set_parameters({'tap_mode': 'regulating'})
                    else:
                        raise ValueError('invalid solution type')
                    method.set_parameters({'quiet': True})

                    bus_P_mis = net.bus_P_mis
                    try:
                        method.solve(net)
                    except ImportError:
                        continue # no ipopt
                    results = method.get_results()
                    self.assertEqual(results['solver name'], solver)
                    self.assertEqual(results['solver status'],'solved')
                    self.assertEqual(net.bus_P_mis,bus_P_mis)
                    self.assertLessEqual(results['network snapshot'].bus_P_mis,bus_P_mis)
                    method.update_network(net)

                    net_snap = results['network snapshot']
                    for bus in net_snap.buses:
                        obus = net.get_bus_from_number(bus.number)
                        self.assertEqual(bus.v_ang, obus.v_ang)
                        self.assertEqual(bus.v_mag, obus.v_mag)
                        self.assertAlmostEqual(bus.P_mismatch, net.get_bus_from_number(bus.number).P_mismatch, places=10)
                    for gen in net_snap.generators:
                        self.assertEqual(gen.P, net.get_generator(gen.index).P)
                    for load in net_snap.loads:
                        self.assertEqual(load.P, net.get_load(load.index).P)
                    for shunt in net_snap.shunts:
                        self.assertEqual(shunt.g, net.get_shunt(shunt.index).g)

                    self.assertLess(abs(net.bus_P_mis), 1e-2) # MW
                    self.assertLess(abs(net.bus_Q_mis), 1e-2) # MVAr

                    self.assertAlmostEqual(results['network snapshot'].bus_P_mis, net.bus_P_mis, places=10)
                    self.assertAlmostEqual(results['network snapshot'].bus_Q_mis, net.bus_Q_mis, places=10)

                    method.solve(netMP)
                    resultsMP = method.get_results()
                    self.assertEqual(resultsMP['solver status'],'solved')
                    method.update_network(netMP)

                    self.assertLess(np.max(np.abs(netMP.bus_P_mis)), 1e-2) # MW
                    self.assertLess(np.max(np.abs(netMP.bus_Q_mis)), 1e-2) # MVAr

                    self.assertLess(norm(resultsMP['network snapshot'].bus_P_mis-netMP.bus_P_mis,np.inf),1e-10)
                    self.assertLess(norm(resultsMP['network snapshot'].bus_Q_mis-netMP.bus_Q_mis,np.inf),1e-10)
                    self.assertLess(norm(resultsMP['network snapshot'].gen_P_cost-netMP.gen_P_cost,np.inf),1e-10)

                    # Sol validation
                    validated = ''
                    if sol_data is not None:

                        v_mag_tol = sol_data['v_mag_tol']
                        v_ang_tol = sol_data['v_ang_tol']
                        bus_data = sol_data['bus_data']

                        counter = 0
                        v_mag_error = []
                        v_ang_error = []
                        for bus_num,val in list(bus_data.items()):

                            v_mag = val['v_mag']
                            v_ang = val['v_ang']

                            try:
                                busMP = netMP.get_bus_from_number(bus_num)
                                bus = net.get_bus_from_number(bus_num)
                            except pf.NetworkError:
                                continue

                            for t in range(T):
                                v_mag_error.append(np.abs(busMP.v_mag[t]-v_mag))
                                v_ang_error.append(np.abs(busMP.v_ang[t]*180./np.pi-v_ang))
                            v_mag_error.append(np.abs(bus.v_mag-v_mag))
                            v_ang_error.append(np.abs(bus.v_ang*180./np.pi-v_ang))

                            counter += 1

                        self.assertEqual(len(v_mag_error),len(v_ang_error))
                        if len(v_mag_error) > 0:
                            validated = 'validated'
                            self.assertLessEqual(np.max(v_mag_error),v_mag_tol)
                            self.assertLessEqual(np.max(v_ang_error),v_ang_tol)
                        self.assertEqual(len(v_mag_error),counter*(T+1))
                        self.assertEqual(len(v_ang_error),counter*(T+1))

                    # Show
                    print("\t%s\t%s\t%s\t%d\t%s" %(case.split(os.sep)[-1],
                                                   sol_types[sol],
                                                   solver,
                                                   results['solver iterations'],
                                                   validated))

    def test_ACOPF_solutions(self):

        print('')

        eps = 0.5 # %

        method_ipopt = gopt.power_flow.new_method('ACOPF')
        method_ipopt.set_parameters(params={'solver':'ipopt','quiet': True})
        method_augl = gopt.power_flow.new_method('ACOPF')
        method_augl.set_parameters(params={'solver':'augl','quiet': True, 'kappa': 1e-2, 'lam_reg': 1e-2})
        method_inlp = gopt.power_flow.new_method('ACOPF')
        method_inlp.set_parameters(params={'solver':'inlp','quiet': True})

        skipcases = ['aesoSL2014.raw','case2869.mat','case9241.mat','case32.art',
                     'ieee59_convert_to_zil_to_solve.raw']

        for case in utils.test_cases:

            if case.split(os.sep)[-1] in skipcases:
                continue

            parser = pf.Parser(case)
            parser.set('output_level', 0)
            net = parser.parse(case)

            # Only small
            if net.num_buses > 3300:
                continue

            self.assertEqual(net.num_periods,1)

            # IPOPT
            try:
                net.update_properties()
                gen_P_cost = net.gen_P_cost
                method_ipopt.solve(net)
                has_ipopt = True
                self.assertEqual(method_ipopt.results['solver status'],'solved')
                self.assertEqual(net.gen_P_cost,gen_P_cost)
                self.assertNotEqual(method_ipopt.results['network snapshot'].gen_P_cost,gen_P_cost)
                self.assertEqual(method_ipopt.results['solver name'], 'ipopt')
                x1 = method_ipopt.get_results()['solver primal variables']
                i1 = method_ipopt.get_results()['solver iterations']
                p1 = method_ipopt.get_results()['network snapshot'].gen_P_cost
                print("\t%s\t%s\t%d" %(case.split(os.sep)[-1],'ipopt',i1))
            except ImportError:
                has_ipopt = False

            # INLP
            net.update_properties()
            gen_P_cost = net.gen_P_cost
            method_inlp.solve(net)
            self.assertEqual(method_inlp.results['solver status'],'solved')
            self.assertEqual(net.gen_P_cost,gen_P_cost)
            self.assertNotEqual(method_inlp.results['network snapshot'].gen_P_cost,gen_P_cost)
            self.assertEqual(method_inlp.results['solver name'], 'inlp')
            x2 = method_inlp.get_results()['solver primal variables']
            i2 = method_inlp.get_results()['solver iterations']
            p2 = method_inlp.get_results()['network snapshot'].gen_P_cost
            print("\t%s\t%s\t%d" %(case.split(os.sep)[-1],'inlp',i2))

            # AUGL
            net.update_properties()
            gen_P_cost = net.gen_P_cost
            method_augl.solve(net)
            self.assertEqual(method_augl.results['solver status'],'solved')
            self.assertEqual(net.gen_P_cost,gen_P_cost)
            self.assertNotEqual(method_augl.results['network snapshot'].gen_P_cost,gen_P_cost)
            self.assertEqual(method_augl.results['solver name'], 'augl')
            x3 = method_augl.get_results()['solver primal variables']
            i3 = method_augl.get_results()['solver iterations']
            p3 = method_augl.get_results()['network snapshot'].gen_P_cost
            print("\t%s\t%s\t%d" %(case.split(os.sep)[-1],'augl',i3))

            # Checks
            if has_ipopt:
                error = 100*(p1-p3)/abs(p3)
                self.assertLess(np.abs(error),eps)
                self.assertNotEqual(p1,p3)
            error = 100*(p2-p3)/abs(p3)
            self.assertLess(np.abs(error),eps)
            self.assertNotEqual(p2,p3)

            # Feasibility
            method_augl.update_network(net)
            self.assertLess(abs(net.bus_P_mis), 1e-2) # MW
            self.assertLess(abs(net.bus_Q_mis), 1e-2) # MVAR

    def test_DCOPF_solutions(self):

        T = 2

        infcases = ['ieee25.raw', 'ieee25.m']

        skipcases = ['case1354.mat','case2869.mat',
                     'case3375wp.mat','case9241.mat',
                     'ieee59_convert_to_zil_to_solve.raw']

        method = gopt.power_flow.new_method('DCOPF')

        for case in utils.test_cases:

            if case.split(os.sep)[-1] in skipcases:
                continue

            parser = pf.Parser(case)
            parser.set('output_level', 0)
            net = parser.parse(case, T)

            for branch in net.branches:
                if branch.ratingA == 0:
                    branch.ratingA = 100

            self.assertEqual(net.num_periods,T)

            method.set_parameters({'quiet':True,
                                   'tol': 1e-6,
                                   'thermal_limits': True})

            try:
                net.update_properties()
                gen_P_cost = net.gen_P_cost
                method.solve(net)
                self.assertEqual(method.results['solver status'],'solved')
                self.assertEqual(method.results['solver name'], 'iqp')
                self.assertTrue(np.all(net.gen_P_cost == gen_P_cost))
                self.assertTrue(np.all(method.results['network snapshot'].gen_P_cost != gen_P_cost))
            except gopt.power_flow.PFmethodError:
                self.assertTrue(case.split(os.sep)[-1] in infcases)
                self.assertEqual(method.results['solver status'],'error')

            results = method.get_results()

            net = results['network snapshot']

            self.assertLess(norm(results['network snapshot'].bus_P_mis-net.bus_P_mis,np.inf),1e-10)
            self.assertLess(norm(results['network snapshot'].bus_Q_mis-net.bus_Q_mis,np.inf),1e-10)
            self.assertLess(norm(results['network snapshot'].gen_P_cost-net.gen_P_cost,np.inf),1e-10)

            gen_P_cost0 = net.gen_P_cost
            load_P_util0 = net.load_P_util
            self.assertTupleEqual(gen_P_cost0.shape,(T,))
            self.assertTupleEqual(load_P_util0.shape,(T,))

            x = results['solver primal variables']
            lam0,nu0,mu0,pi0 = results['solver dual variables']

            self.assertTupleEqual(x.shape,((net.num_buses-
                                            net.get_num_slack_buses()+
                                            net.get_num_generators())*T,))
            self.assertTupleEqual(x.shape,(net.num_vars,))
            self.assertTupleEqual(lam0.shape,(net.num_buses*T,))
            self.assertTrue(nu0.size == 0)
            self.assertTupleEqual(mu0.shape,(net.num_vars+net.num_branches*T,))
            self.assertTupleEqual(pi0.shape,(net.num_vars+net.num_branches*T,))

            # Network update (vars and sensitivities)
            xx = x[:net.num_vars]
            row = 0
            for t in range(T):
                for bus in net.buses:
                    if not bus.is_slack():
                        self.assertEqual(bus.v_ang[t],xx[bus.index_v_ang[t]])
                        self.assertEqual(bus.sens_v_ang_u_bound[t],mu0[bus.index_v_ang[t]])
                        self.assertEqual(bus.sens_v_ang_l_bound[t],pi0[bus.index_v_ang[t]])
                    for branch in bus.branches_k:
                        self.assertEqual(branch.sens_P_u_bound[t],mu0[net.num_vars+row])
                        self.assertEqual(branch.sens_P_l_bound[t],pi0[net.num_vars+row])
                        row += 1
                for gen in net.generators:
                    self.assertEqual(gen.P[t],xx[gen.index_P[t]])
                    self.assertEqual(gen.sens_P_u_bound[t],mu0[gen.index_P[t]])
                    self.assertEqual(gen.sens_P_l_bound[t],pi0[gen.index_P[t]])

            # No thermal limits
            method.set_parameters({'thermal_limits':False})
            method.solve(net)
            self.assertEqual(method.results['solver status'],'solved')
            results = method.get_results()
            net = results['network snapshot']
            gen_P_cost1 = net.gen_P_cost
            load_P_util1 = net.load_P_util
            lam1,nu1,mu1,pi1 = results['solver dual variables']
            self.assertTupleEqual(mu1.shape,(net.num_vars,))
            self.assertTupleEqual(pi1.shape,(net.num_vars,))

            # Elastic loads
            for load in net.loads:
                load.P_max = load.P+1.
                load.P_min = load.P-1.
            self.assertEqual(net.get_num_P_adjust_loads(),net.num_loads)
            for load in net.loads:
                self.assertFalse(load.has_flags('variable','active power'))
                self.assertFalse(load.has_flags('bounded','active power'))
                self.assertTrue(np.all(load.P_min < load.P_max))
            method.solve(net)
            for load in net.loads:
                self.assertFalse(load.has_flags('variable','active power'))
                self.assertFalse(load.has_flags('bounded','active power'))
            results = method.get_results()
            net = results['network snapshot']
            for load in net.loads:
                self.assertTrue(load.has_flags('variable','active power'))
                self.assertTrue(load.has_flags('bounded','active power'))
            self.assertEqual(method.results['solver status'],'solved')
            self.assertTrue(np.all(net.gen_P_cost-net.load_P_util < gen_P_cost1-load_P_util1))

            x = results['solver primal variables']
            lam2,nu2,mu2,pi2 = results['solver dual variables']

            self.assertTupleEqual(x.shape,((net.get_num_P_adjust_loads()+
                                            net.num_buses-
                                            net.get_num_slack_buses()+
                                            net.get_num_generators())*net.num_periods,))
            self.assertTupleEqual(x.shape,(net.num_vars,))
            self.assertTupleEqual(lam2.shape,(net.num_buses*net.num_periods,))
            self.assertTrue(nu2.size == 0)
            self.assertTupleEqual(mu2.shape,x.shape)
            self.assertTupleEqual(pi2.shape,x.shape)

            xx = x[:net.num_vars]
            for t in range(T):
                for bus in net.buses:
                    if not bus.is_slack():
                        self.assertEqual(bus.v_ang[t],xx[bus.index_v_ang[t]])
                        self.assertEqual(bus.sens_v_ang_u_bound[t],mu2[bus.index_v_ang[t]])
                        self.assertEqual(bus.sens_v_ang_l_bound[t],pi2[bus.index_v_ang[t]])
                for gen in net.generators:
                    self.assertEqual(gen.P[t],xx[gen.index_P[t]])
                    self.assertEqual(gen.sens_P_u_bound[t],mu2[gen.index_P[t]])
                    self.assertEqual(gen.sens_P_l_bound[t],pi2[gen.index_P[t]])
                for load in net.loads:
                    if load.is_P_adjustable():
                        self.assertEqual(load.P[t],xx[load.index_P[t]])
                        self.assertEqual(load.sens_P_u_bound[t],mu2[load.index_P[t]])
                        self.assertEqual(load.sens_P_l_bound[t],pi2[load.index_P[t]])

    def test_ACPF_initialize_problem(self):

        import time 

        case = os.path.join('tests', 'resources', 'cases', 'ieee25.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)

        # Define some contingencies
        contingencies = []
        for gen in net.generators:
            if not gen.is_slack():
                gc = pf.Contingency()
                gc.name = gen.name
                gc.add_generator_outage(gen)
                contingencies.append(gc)

        acpf = gopt.power_flow.ACPF()
        acpf.set_parameters({'solver': 'nr', 'quiet': True})

        # Check consistent for base case solves
        unet, problem = acpf.initialize_problem(net)
        self.assertTrue(isinstance(unet, pf.Network))
        self.assertTrue(isinstance(problem, pf.Problem))
        acpf.solve_problem(unet, problem, save_problem=True)
        result_steps = acpf.get_results()
        self.assertEqual(problem, result_steps['problem'])
        self.assertLess(np.linalg.norm(
            result_steps['network snapshot'].get_var_values() - problem.x), 1e-6)
        self.assertLess(np.linalg.norm(
            result_steps['network snapshot'].get_var_values() - unet.get_var_values()), 1e-6)
        self.assertLess(np.linalg.norm(
            result_steps['network snapshot'].get_var_values() - result_steps['solver primal variables']), 1e-6)

        acpf.solve(net, save_problem=True)
        result_orig = acpf.get_results()
        acpf.update_network(net)
        self.assertTrue(isinstance(result_orig['problem'], pf.Problem))
        self.assertNotEqual(result_orig['problem'], result_steps['problem'])
        self.assertLess(np.linalg.norm(
            result_orig['network snapshot'].get_var_values() - result_orig['solver primal variables']), 1e-6)
        self.assertLess(np.linalg.norm(
            result_orig['solver primal variables'] - result_steps['solver primal variables']), 1e-6)

        # Check solve for contingencies are consistent
        for cont in contingencies:
            cont.apply(net)
            unet, problem = acpf.initialize_problem(net)
            acpf.solve_problem(unet, problem, save_problem=True)
            result_steps = acpf.get_results()
            xsteps = result_steps['solver primal variables']
            self.assertTrue(isinstance(result_steps['problem'], pf.Problem))
            self.assertLess(np.linalg.norm(xsteps - problem.x), 1e-6)

            acpf.solve(net, save_problem=True)
            result_orig = acpf.get_results()
            xorig = result_orig['solver primal variables']
            self.assertTrue(isinstance(result_orig['problem'], pf.Problem))
            self.assertLess(np.linalg.norm(xorig - result_orig['problem'].x), 1e-6)

            self.assertLess(np.linalg.norm(xorig - xsteps), 1e-6)
            cont.clear(net)

    def test_ACPF_with_redispatch(self):

        case = os.path.join('tests', 'resources', 'cases', 'ieee25.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        # Define non-redispatchable gens
        gen_no_rdisp = range(10,20)

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)

        method = gopt.power_flow.new_method('ACPF')
        method.set_parameters(params={'solver': 'augl',
                                      'gens_redispatch': True,
                                      'weight_redispatch': 1e0,
                                      'Q_limits': False,
                                      'quiet': True})
        method.solve(net)

        # Network with no redispatchable generator other than slack
        net_no_rdisp = method.get_results()['network snapshot']

        self.assertEqual(method.get_results()['solver status'], 'solved')

        for gen in net.generators:
            gen.redispatchable = True
        method.solve(net)

        # Network with all generator redispatchable
        net_all_rdisp = method.get_results()['network snapshot']

        for i in gen_no_rdisp:
            net.generators[i].redispatchable = False
        method.solve(net)

        # Network with partial generator redispatchable
        net_part_rdisp = method.get_results()['network snapshot']

        for i in gen_no_rdisp:
            gen = net_part_rdisp.generators[i]
            self.assertFalse(gen.is_redispatchable())
            self.assertEqual(gen.P, net_no_rdisp.get_generator(gen.index).P)

        for i in range(22,24):
            gen = net_part_rdisp.generators[i]
            self.assertTrue(gen.is_redispatchable())
            self.assertFalse(gen.P == net_no_rdisp.get_generator(gen.index).P)
            self.assertFalse(gen.P == net_all_rdisp.get_generator(gen.index).P)

        bus = net.buses[2]
        self.assertFalse(bus.is_slack())
        for gen in bus.generators:
            if gen.is_slack():
                self.assertFalse(gen.is_redispatchable())

        bus.set_slack_flag(True)
        self.assertTrue(bus.is_slack())
        for gen in bus.generators:
            if gen.is_slack():
                self.assertTrue(gen.is_redispatchable())
    
    def test_acpf_with_zip_loads(self):

        case = os.path.join('tests', 'resources', 'cases',
                            'ieee59_convert_to_zil_to_solve.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.Parser(case)
        parser.set('output_level', 0)
        net = parser.parse(case)
        load_orig = net.get_load_from_name_and_bus_number('1', 49)

        acpf = gopt.power_flow.ACPF()
        acpf.set_parameters(params={'solver': 'nr',
                                    'loads_2_ZIP': False,
                                    'quiet': False})
        try:
            acpf.solve(net)
        except gopt.power_flow.method_error.PFmethodError_SolverError:
            pass
        no_load_2_zip = acpf.get_results()
        net_no_zip = no_load_2_zip['network snapshot']
        load_no_zip = net_no_zip.get_load_from_name_and_bus_number('1', 49)
        self.assertTrue(load_no_zip.bus.v_mag < 0.8)
        self.assertEqual(load_orig.P, load_no_zip.P)
        self.assertEqual(load_orig.Q, load_no_zip.Q)
        self.assertEqual(no_load_2_zip['solver status'], 'error')

        acpf.set_parameters(params={'loads_2_ZIP': True})
        acpf.solve(net)
        with_load_2_zip = acpf.get_results()       
        net_with_zip = with_load_2_zip['network snapshot']
        load_zip = net_with_zip.get_load_from_name_and_bus_number('1', 49)
        vm = load_zip.bus.v_mag
        self.assertTrue(vm >= 0.8)
        self.assertTrue(load_orig.P > load_zip.P)
        self.assertTrue(load_orig.Q > load_zip.Q)
        self.assertAlmostEqual(
            load_zip.P, 
            load_zip.comp_cp * vm * vm + load_zip.comp_ci * vm + load_zip.comp_cg, 4)
        self.assertAlmostEqual(
            load_zip.Q, 
            load_zip.comp_cb - + load_zip.comp_cq * vm * vm - load_zip.comp_cj * vm, 4)
        self.assertEqual(with_load_2_zip['solver status'], 'solved')

    def tearDown(self):

        pass
