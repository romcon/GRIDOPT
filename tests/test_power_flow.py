#*****************************************************#
# This file is part of GRIDOPT.                       #
#                                                     #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.    #
#                                                     #
# GRIDOPT is released under the BSD 2-clause license. #
#*****************************************************#

import utils
import unittest
import numpy as np
import pfnet as pf
import gridopt as gopt

INFCASE = './tests/resources/ieee25.raw'

class TestPowerFlow(unittest.TestCase):
    
    def setUp(self):
        
        # Network
        self.net = pf.Network()

    def test_method_solutions(self):

        print ''
        net = self.net
        sol_types = {'sol1': 'no controls',
                     'sol2': 'gen controls',
                     'sol3': 'all controls'}
        
        for method_name in ['NRPF','AugLPF']:
            for case in utils.test_cases:
                for sol in sol_types.keys():

                    method = gopt.power_flow.new_method(method_name)
                    
                    net.load(case)
                    
                    sol_data = utils.read_solution_data(case+'.'+sol)
                
                    # Skip
                    if (sol == 'sol1' and         # no controls
                        method_name == 'AugLPF'):

                        continue

                    if sol == 'sol1':   # no controls
                        method.set_parameters({'limit_gens': False})
                    elif sol == 'sol2': # generator voltage control
                        pass
                    else:               # generator and tap controls
                        method.set_parameters({'lock_taps': False})
                    method.set_parameters({'quiet': True})

                    method.solve(net)

                    results = method.get_results()

                    self.assertEqual(results['status'],'solved')
                    
                    method.update_network(net)

                    self.assertLess(np.abs(results['net_properties']['bus_P_mis']-net.bus_P_mis),1e-10)
                    self.assertLess(np.abs(results['net_properties']['bus_Q_mis']-net.bus_Q_mis),1e-10)
                    self.assertLess(np.abs(results['net_properties']['gen_P_cost']-net.gen_P_cost),1e-10)

                    v_mag_tol = sol_data['v_mag_tol']
                    v_ang_tol = sol_data['v_ang_tol']
                    bus_data = sol_data['bus_data']

                    v_mag_error = [0]
                    v_ang_error = [0]
                    for bus_num,val in bus_data.items():
                        
                        v_mag = val['v_mag']
                        v_ang = val['v_ang']
                        
                        try:
                            bus = net.get_bus_by_number(bus_num)
                        except pf.NetworkError:
                            continue

                        v_mag_error.append(np.abs(bus.v_mag-v_mag))
                        v_ang_error.append(np.abs(bus.v_ang*180./np.pi-v_ang))
                    
                    print method_name,case,sol_types[sol],len(v_mag_error),len(v_ang_error)

                    self.assertLessEqual(np.max(v_mag_error),v_mag_tol)
                    self.assertLessEqual(np.max(v_ang_error),v_ang_tol)

    def test_AugLOPF(self):
        
        net = self.net
        method = gopt.power_flow.new_method('AugLOPF')

        for case in utils.test_cases:
        
            net.load(case)
            
            method.set_parameters({'quiet':True})

            method.solve(net)
            self.assertEqual(method.results['status'],'solved')
            
            # gen outage
            cont = pf.Contingency([net.get_gen(0)])
            cont.apply()
            problem = method.create_problem(net)
            cont.clear()

    def test_DCOPF(self):
        
        net = self.net
        method = gopt.power_flow.new_method('DCOPF')

        for case in utils.test_cases:
        
            net.load(case)
            
            method.set_parameters({'quiet':True, 'thermal_factor': 0.93})

            try:
                method.solve(net)
                self.assertEqual(method.results['status'],'solved')
            except gopt.power_flow.PFmethodError:
                self.assertEqual(case,INFCASE)
                self.assertEqual(method.results['status'],'error')
                
            results = method.get_results()
                
            method.update_network(net)
           
            self.assertLess(np.abs(results['net_properties']['bus_P_mis']-net.bus_P_mis),1e-10)
            self.assertLess(np.abs(results['net_properties']['bus_Q_mis']-net.bus_Q_mis),1e-10)
            self.assertLess(np.abs(results['net_properties']['gen_P_cost']-net.gen_P_cost),1e-10)

            gen_P_cost0 = net.gen_P_cost
            load_P_util0 = net.load_P_util
            
            x = results['primal_variables']
            lam0,nu0,mu0,pi0 = results['dual_variables']

            self.assertTupleEqual(x.shape,(net.num_branches+
                                           net.num_buses-
                                           net.get_num_slack_buses()+
                                           net.get_num_P_adjust_gens(),))
            self.assertTupleEqual(x.shape,(net.num_vars+net.num_branches,))
            self.assertTupleEqual(lam0.shape,(net.num_buses+net.num_branches,))
            self.assertTrue(nu0 is None)
            self.assertTupleEqual(mu0.shape,x.shape)
            self.assertTupleEqual(pi0.shape,x.shape)

            xx = x[:net.num_vars]
            for bus in net.buses:
                if not bus.is_slack():
                    self.assertEqual(bus.v_ang,xx[bus.index_v_ang])
                    self.assertEqual(bus.sens_v_ang_u_bound,mu0[bus.index_v_ang]*net.base_power)
                    self.assertEqual(bus.sens_v_ang_l_bound,pi0[bus.index_v_ang]*net.base_power)
            for gen in net.generators:
                if gen.is_P_adjustable():
                    self.assertEqual(gen.P,xx[gen.index_P])
                    self.assertEqual(gen.sens_P_u_bound,mu0[gen.index_P]*net.base_power)
                    self.assertEqual(gen.sens_P_l_bound,pi0[gen.index_P]*net.base_power)
            for branch in net.branches:
                self.assertEqual(branch.sens_P_u_bound,mu0[net.num_vars+branch.index]*net.base_power)
                self.assertEqual(branch.sens_P_l_bound,pi0[net.num_vars+branch.index]*net.base_power)

            # gen outage 
            if net.get_num_P_adjust_gens() > 1:
                cont = pf.Contingency(gens=[net.get_gen(0)])
                cont.apply()
                try:
                    method.solve(net)
                    self.assertEqual(method.results['status'],'solved')
                except gopt.power_flow.PFmethodError:
                    self.assertEqual(case,INFCASE)
                    self.assertEqual(method.results['status'],'error')
                cont.clear()

            # no thermal limits
            method.set_parameters({'thermal_limits':False})
            method.solve(net)
            self.assertEqual(method.results['status'],'solved')
            results = method.get_results()
            method.update_network(net)
            gen_P_cost1 = net.gen_P_cost
            load_P_util1 = net.load_P_util
            lam1,nu1,mu1,pi1 = results['dual_variables']
            if ((np.linalg.norm(mu0[net.num_vars:],np.inf) > 1e-3 or 
                np.linalg.norm(pi0[net.num_vars:],np.inf) > 1e-3) and case != INFCASE):
                self.assertLessEqual(gen_P_cost1,gen_P_cost0)
            self.assertLess(np.linalg.norm(mu1[net.num_vars:],np.inf),1e-8)
            self.assertLess(np.linalg.norm(pi1[net.num_vars:],np.inf),1e-8)
           
            # elastic loads
            for load in net.loads:
                load.P_max = load.P+1.
                load.P_min = load.P-1.
            for load in net.loads:
                self.assertFalse(load.has_flags(pf.FLAG_VARS,pf.LOAD_VAR_P))
                self.assertFalse(load.has_flags(pf.FLAG_BOUNDED,pf.LOAD_VAR_P))
            method.solve(net)
            for load in net.loads:
                self.assertTrue(load.has_flags(pf.FLAG_VARS,pf.LOAD_VAR_P))
                self.assertTrue(load.has_flags(pf.FLAG_BOUNDED,pf.LOAD_VAR_P))
                self.assertEqual(method.results['status'],'solved')
            results = method.get_results()
            method.update_network(net)
            self.assertLess(net.gen_P_cost-net.load_P_util,
                            gen_P_cost1-load_P_util1)

            x = results['primal_variables']
            lam2,nu2,mu2,pi2 = results['dual_variables']

            self.assertTupleEqual(x.shape,(net.num_branches+
                                           net.get_num_P_adjust_loads()+
                                           net.num_buses-
                                           net.get_num_slack_buses()+
                                           net.get_num_P_adjust_gens(),))
            self.assertTupleEqual(x.shape,(net.num_vars+net.num_branches,))
            self.assertTupleEqual(lam2.shape,(net.num_buses+net.num_branches,))
            self.assertTrue(nu2 is None)
            self.assertTupleEqual(mu2.shape,x.shape)
            self.assertTupleEqual(pi2.shape,x.shape)

            xx = x[:net.num_vars]
            for bus in net.buses:
                if not bus.is_slack():
                    self.assertEqual(bus.v_ang,xx[bus.index_v_ang])
                    self.assertEqual(bus.sens_v_ang_u_bound,mu2[bus.index_v_ang]*net.base_power)
                    self.assertEqual(bus.sens_v_ang_l_bound,pi2[bus.index_v_ang]*net.base_power)
            for gen in net.generators:
                if gen.is_P_adjustable():
                    self.assertEqual(gen.P,xx[gen.index_P])
                    self.assertEqual(gen.sens_P_u_bound,mu2[gen.index_P]*net.base_power)
                    self.assertEqual(gen.sens_P_l_bound,pi2[gen.index_P]*net.base_power)
            for load in net.loads:
                if load.is_P_adjustable():
                    self.assertEqual(load.P,xx[load.index_P])
                    self.assertEqual(load.sens_P_u_bound,mu2[load.index_P]*net.base_power)
                    self.assertEqual(load.sens_P_l_bound,pi2[load.index_P]*net.base_power)
            for branch in net.branches:
                self.assertEqual(branch.sens_P_u_bound,mu2[net.num_vars+branch.index]*net.base_power)
                self.assertEqual(branch.sens_P_l_bound,pi2[net.num_vars+branch.index]*net.base_power)
 
    @unittest.skip("")
    def test_DCOPF_prev(self):
        
        net = self.net
        method = gopt.power_flow.new_method('DCOPF_Prev')
        method_ref = gopt.power_flow.new_method('DCOPF')

        for case in utils.test_cases:
        
            net.load(case)

            method.set_parameters({'quiet':True, 'tol':1e-5})
            method_ref.set_parameters({'quiet':True,'tol':1e-5})

            # No contingencies (compare with DCOPF)
            try:
                method.solve(net,[])
                self.assertEqual(method.results['status'],'solved')
            except gopt.power_flow.PFmethodError:
                self.assertEqual(case,INFCASE)
                self.assertEqual(method.results['status'],'error')
            try:
                method_ref.solve(net)
                self.assertEqual(method_ref.results['status'],'solved')
            except gopt.power_flow.PFmethodError:
                self.assertEqual(case,INFCASE)
                self.assertEqual(method_ref.results['status'],'error')

            if case != INFCASE:
                
                results = method.get_results()
                results_ref = method_ref.get_results()
            
                self.assertEqual(results['status'],results_ref['status'])
                self.assertEqual(results['error_msg'],results_ref['error_msg'])
                self.assertEqual(results['iterations'],results_ref['iterations'])
                nprop = results['net_properties']
                nprop_ref = results_ref['net_properties']
                self.assertTrue(set(nprop.keys()) == set(nprop_ref.keys()))
                for k in nprop.keys():
                    self.assertLess(np.abs(nprop[k]-nprop_ref[k]),1e-5)
                x = results['primal_variables']
                x_ref = results_ref['primal_variables']
                self.assertLess(np.linalg.norm(x-x_ref,np.inf),1e-5)
                lam,nu,mu,pi = results['dual_variables']
                lam_ref,nu_ref,mu_ref,pi_ref = results_ref['dual_variables']
                self.assertLess(np.linalg.norm(lam-lam_ref,np.inf),1e-5)
                self.assertLess(np.linalg.norm(mu-mu_ref,np.inf),1e-5)
                self.assertLess(np.linalg.norm(pi-pi_ref,np.inf),1e-5)
                self.assertTrue(nu is None)
                self.assertTrue(nu_ref is None)

            # Multiple base cases
            try:
                method.solve(net,[pf.Contingency(),pf.Contingency()])
                self.assertEqual(method.results['status'],'solved')
            except gopt.power_flow.PFmethodError:
                if net.get_num_P_adjust_gens() > 1:
                    self.assertEqual(case,INFCASE)
                    self.assertEqual(method.results['status'],'error')
                else:
                    pass

            if case != INFCASE and net.get_num_P_adjust_gens() > 1:
                
                results = method.get_results()
            
                self.assertEqual(results['status'],results_ref['status'])
                self.assertEqual(results['error_msg'],results_ref['error_msg'])
                nprop = results['net_properties']
                nprop_ref = results_ref['net_properties']
                self.assertTrue(set(nprop.keys()) == set(nprop_ref.keys()))
                for k in nprop.keys():
                    self.assertLess(100*np.abs(nprop[k]-nprop_ref[k])/np.maximum(np.abs(nprop_ref[k]),1e-5),0.1)
                x = results['primal_variables']
                x_ref = results_ref['primal_variables']
                self.assertLess(np.linalg.norm(x-x_ref,np.inf),1e-3)

            # Single gen contingency
            if net.get_num_P_adjust_gens() > 1:
                gen = net.get_gen(np.argmin([g.P_max-g.P_min for g in net.generators]))
                contG = pf.Contingency(gens=[gen])
                method.set_parameters({'quiet':True, 'tol':1e-4})
                try:
                    method.solve(net,[contG])
                    self.assertEqual(method_ref.results['status'],'solved')
                except gopt.power_flow.PFmethodError:
                    self.assertEqual(case,INFCASE)
                    self.assertEqual(method_ref.results['status'],'error')

            # Single branch contingency
            branch = net.get_branch(np.argmax([np.minimum(br.bus_from.degree,br.bus_to.degree) for br in net.branches]))
            if branch.bus_from.degree > 3 and branch.bus_to.degree > 3:
                contB = pf.Contingency(branches=[branch])
                method.set_parameters({'quiet':True, 'tol':1e-4})
                try:
                    method.solve(net,[contB])
                    self.assertEqual(method_ref.results['status'],'solved')
                except gopt.power_flow.PFmethodError:
                    self.assertEqual(case,INFCASE)
                    self.assertEqual(method_ref.results['status'],'error')

    #@unittest.skip("")
    def test_DCOPF_corr(self):
        
        net = self.net
        method = gopt.power_flow.new_method('DCOPF_Corr')
        method_ref = gopt.power_flow.new_method('DCOPF')

        for case in utils.test_cases:
        
            net.load(case)
            
            method.set_parameters({'quiet':True, 'tol':1e-5})
            method_ref.set_parameters({'quiet':True,'tol':1e-5})

            # No contingencies (compare with DCOPF)
            try:
                method.solve(net,[])
                self.assertEqual(method.results['status'],'solved')
            except gopt.power_flow.PFmethodError:
                self.assertEqual(case,INFCASE)
                self.assertEqual(method.results['status'],'error')
            try:
                method_ref.solve(net)
                self.assertEqual(method_ref.results['status'],'solved')
            except gopt.power_flow.PFmethodError:
                self.assertEqual(case,INFCASE)
                self.assertEqual(method_ref.results['status'],'error')

            if case != INFCASE:
                
                results = method.get_results()
                results_ref = method_ref.get_results()
            
                self.assertEqual(results['status'],results_ref['status'])
                self.assertEqual(results['error_msg'],results_ref['error_msg'])
                self.assertEqual(results['iterations'],results_ref['iterations'])
                nprop = results['net_properties']
                nprop_ref = results_ref['net_properties']
                self.assertTrue(set(nprop.keys()) == set(nprop_ref.keys()))
                for k in nprop.keys():
                    self.assertLess(np.abs(nprop[k]-nprop_ref[k]),1e-5)
                x = results['primal_variables']
                x_ref = results_ref['primal_variables']
                self.assertLess(np.linalg.norm(x-x_ref,np.inf),1e-5)
                lam,nu,mu,pi = results['dual_variables']
                lam_ref,nu_ref,mu_ref,pi_ref = results_ref['dual_variables']
                self.assertLess(np.linalg.norm(lam-lam_ref,np.inf),1e-5)
                self.assertLess(np.linalg.norm(mu-mu_ref,np.inf),1e-5)
                self.assertLess(np.linalg.norm(pi-pi_ref,np.inf),1e-5)
                self.assertTrue(nu is None)
                self.assertTrue(nu_ref is None)
                    
    def tearDown(self):
        
        pass

