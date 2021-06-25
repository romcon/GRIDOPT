.. include:: defs.hrst

.. _power_flow:

**********
Power Flow
**********

The Power Flow (PF) problem consists of determining steady-state voltage magnitudes and angles at every bus of the network as well as any unknown generator powers. On the other hand, the Optimal Power Flow (OPF) problem consists of determining generator powers and network control settings that result in the optimal operation of the network according to some measure, *e.g.*, generation cost. In GRIDOPT, methods for solving PF and OPF problems are represented by objects derived from a :class:`method <gridopt.power_flow.method.PFmethod>` base class.

To solve a PF or OPF problem, one first needs to create an instance of a specific method subclass. This is done using the function :func:`new_method() <gridopt.power_flow.new_method>`, which takes as argument the name of an available PF or OPF method. Available methods are the following:

* :ref:`dc_pf`
* :ref:`dc_opf`
* :ref:`ac_pf`
* :ref:`ac_opf`

The following code sample creates an instance of the :ref:`ac_pf` method::

  >>> import gridopt

  >>> method = gridopt.power_flow.new_method('ACPF')

Once a method has been instantiated, its parameters can be set using the function :func:`set_parameters() <gridopt.power_flow.method.PFmethod.set_parameters>`. This function takes as argument a dictionary with parameter name-value pairs. Valid parameters include parameters of the method, which are described in the sections below, and parameters of the numerical :class:`solver <optalg.opt_solver.opt_solver.OptSolver>` used by the method. The numerical solvers used by the methods of GRIDOPT belong to the Python package |OPTALG|. The following code sample sets a few parameters of the method created above::

  >>> method.set_parameters({'solver': 'nr', 'quiet': True, 'feastol': 1e-4})

After configuring parameters, a method can be used to solve a problem using the function :func:`solve() <gridopt.power_flow.method.PFmethod.solve>`. This function takes as argument a |PFNET| |network| object, as follows::

  >>> import pfnet

  >>> net = pfnet.PyParserMAT().parse('ieee14.m')

  >>> method.solve(net)

Information about the execution of the method can be obtained from the :data:`results <gridopt.power_flow.method.PFmethod.results>` attribute of the :class:`method <gridopt.power_flow.method.PFmethod>` object. This dictionary of results includes information such as ``'solver status'``, *e.g.*, ``'solved'`` or ``'error'``, any ``'solver message'``, ``'solver iterations'``, a ``'network snapshot'`` reflecting the solution, and others. The following code sample shows how to extract some results::

  >>> results = method.get_results()

  >>> print(results['solver status'])
  solved

  >>> print(results['solver iterations'])
  2

  >>> print(results['network snapshot'].bus_v_max)
  1.09

If desired, one can update the input |Network| object with the solution found by the method. This can be done with the function :func:`update_network() <gridopt.power_flow.method.PFmethod.update_network>`. This routine not only updates the network quantities treated as variables by the method, but also information about the sensitivity of the optimal objective value with respect to perturbations of the constraints. The following code sample updates the power network with the results obtained by the method and shows the resulting maximum active and reactive bus power mismatches in units of MW and MVAr::

  >>> method.update_network(net)

  >>> print('%.2e %.2e' %(net.bus_P_mis, net.bus_Q_mis))
  5.09e-06 9.70e-06

.. _ref_optalg_params:

Typical OptSolver Parameters
============================

Parameters for the |OPTALG| solver can be set through the GRIDOPT power flow method parameters. The tables below list some of the more typical parameters. For more information see |OPTALG| documentation. When using external solvers, typically default values are used.

Newton-Raphson (|NR|) OPTALG Solver:

=========================== ====================================================== =============
Name                        Description                                            Default
=========================== ====================================================== =============
``'feastol'``               Feasibility tolerance in per units                     ``1e-4``
``'acc_factor'``            Acceleration factor at each step                       ``1.0``
``'maxiter'``               Maximum number of iterations                           ``100``
``'linsolver'``             Linear solver ``{'superlu', 'mumps', 'umfpack'}``      ``'superlu'``
=========================== ====================================================== =============

Augmented Lagrangian (|AUGL|) OPTALG Solver:

=========================== ====================================================== =============
Name                        Description                                            Default
=========================== ====================================================== =============
``'feastol'``               Feasibility tolerance in per units                     ``1e-4``
``'optol'``                 Optimality tolerance                                   ``1e-4``
``'maxiter'``               Maximum number of iterations                           ``1000``
``'linsolver'``             Linear solver ``{'superlu', 'mumps', 'umfpack'}``      ``'superlu'``
=========================== ====================================================== =============

Interior-point Nonlinear Programming (|INLP|) OPTALG Solver:

=========================== ====================================================== =============
Name                        Description                                            Default
=========================== ====================================================== =============
``'feastol'``               Feasibility tolerance in per units                     ``1e-4``
``'optol'``                 Optimality tolerance                                   ``1e-4``
``'sigma'``                 Factor for increasing subproblem solution accuracy     ``0.1``
``'maxiter'``               Maximum number of iterations                           ``300``
``'linsolver'``             Linear solver ``{'superlu', 'mumps', 'umfpack'}``      ``'superlu'``
=========================== ====================================================== =============

Interior-point Quadratic Programming (|IQP|) OPTALG Solver:

=========================== ====================================================== =============
Name                        Description                                            Default
=========================== ====================================================== =============
``'tol'``                   Optimality tolerance                                   ``1e-4``
``'sigma'``                 Factor for increasing subproblem solution accuracy     ``0.1``
``'maxiter'``               Maximum number of iterations                           ``1000``
``'linsolver'``             Linear solver ``{'superlu', 'mumps', 'umfpack'}``      ``'superlu'``
=========================== ====================================================== =============

|IPOPT| wrapper OPTALG Solver:

=========================== ====================================================== =============
Name                        Description                                            Default
=========================== ====================================================== =============
``'tol'``                   Optimality tolerance                                   ``1e-4``
``'sigma'``                 Factor for increasing subproblem solution accuracy     ``0.1``
``'max_iter'``              Maximum number of iterations                           ``1000``
``'linsolver'``             Linear solver ``{'superlu', 'mumps', 'umfpack'}``      ``'mumps'``
``'print_level'``           Ouptut print level {0-9}                               ``5``
``'max_cpu_time'``          Maximum cpu time (seconds) stopping critera            ``1e6``
=========================== ====================================================== =============

.. _dc_pf:

DCPF
====

This method is represented by an object of type :class:`DCPF <gridopt.power_flow.dc_pf.DCPF>` and solves a DC power flow problem, which is just a linear system of equations representing |ConstraintDCPF| constraints.

The parameters of this method are the following:

=========================== ====================================================== =============
Name                        Description                                            Default
=========================== ====================================================== =============
``'solver'``                OPTALG linear solver ``{'superlu','mumps','umfpack'}`` ``'superlu'``
``'quiet'``                 Flag for disabling output                              ``False``
=========================== ====================================================== =============

.. _dc_opf:

DCOPF
=====

This method is represented by an object of type :class:`DCOPF <gridopt.power_flow.dc_opf.DCOPF>` and solves a DC optimal power flow problem, which is just a quadratic program that considers |FunctionGEN_COST|, |FunctionLOAD_UTIL|, |ConstraintDCPF|, |ConstraintBOUND|, *e.g.*, generator and load limits, and |ConstraintDC_FLOW_LIM|.

The parameters of this method are the following:

=========================== ============================================================ =========
Name                        Description                                                  Default
=========================== ============================================================ =========
``'thermal_limits'``        Flag for considering branch flow limits                      ``False``
``'renewable_curtailment'`` Flag for allowing curtailment of renewable generators        ``False``
``'solver'``                OPTALG optimization solver ``{'iqp','inlp','augl','ipopt'}`` ``'iqp'``
``'quiet'``                 Flag for disabling output                                    ``False``
=========================== ============================================================ =========

.. _ac_pf:

ACPF
====

This method is represented by an object of type :class:`ACPF <gridopt.power_flow.ac_pf.ACPF>` and solves an AC power flow problem. For doing this, it can use the |NR| solver from |OPTALG| together with "switching" heuristics for modeling local controls. Alternatively, it can formulate the problem as an optimization problem with a convex objective function and *complementarity constraints*, *e.g.*, |ConstraintREG_GEN|, for modeling local controls, and solve it using the |AUGL|, |INLP|, or |IPOPT| solver available through |OPTALG|.

The parameters of this power flow method are the following:

======================= ================================================================= =================
Name                    Description                                                       Default
======================= ================================================================= =================
``'weight_vmag'``       Weight for bus voltage magnitude regularization                   ``1e0``
``'weight_vang'``       Weight for bus voltage angle regularization                       ``1e0``
``'weight_powers'``     Weight for generator power regularization                         ``1e-3``
``'weight_var'``        Weight for generic regularization                                 ``1e-5``
``'weight_redispatch'`` Weight for generate real power redispatch deviation penalty       ``1e0``
``'v_min_clip'``        Per unit voltage threshold for clipping lower bound               ``0.5``
``'v_max_clip'``        Per unit voltage threshold for clipping upper bounds              ``1.5``
``'v_limits'``          Flag for applying voltage magnitude limits                        ``False``
``'Q_limits'``          Flag for enforcing generator, VSC, FACTS reactive limits          ``True``
``'Q_mpode'``           Reactive power mode ``{'regulating', 'free'}``                    ``'regulating'``
``'shunt_limits'``      Flag for enforcing switched shunt susceptance Q_limits            ``True``
``'shunt_mode'``        Switched shunts mode ``{'locked', 'free', 'regulating'}``         ``locked``
``'tap_limits'``        Flag for enforcing transformer tap ratio limits                   ``True``
``'tap_mode'``          Transformer tap ratio mode ``{'locked', 'free', 'regulating'}``   ``locked``
``'lock_vsc_P_dc'``     Flag for locking VSC :math:`P_{dc}`                               ``True``
``'lock_csc_P_dc'``     Flag for locking CSC :math:`P_{dc}`                               ``True``
``'lock_csc_i_dc'``     Flag for locking CSC :math:`i_{dc}`                               ``True``
``'vdep_loads'``        Flag for modeling voltage dependent loads                         ``False``
``'pvpq_start_k'``      Start iteration number for PVPQ switching heuristics              ``0``
``'vmin_thresh'``       Low-voltage threshold stopping criteria                           ``1e-1``
``'gens_redispatch'``   Flag for allowing generator active power redispatch               ``False``
``'v_mag_warm_ref'``    Flag for using current voltage magnitude for regularization       ``False``
``'tap_step'``          Tap ratio acceleration factor (NR only)                           ``0.5``
``'shunt_step'``        Shunt susceptance acceleration factor (NR only)                   ``0.5``
``'dtap'``              Tap ratio perturbation step (NR only)                             ``1e-4``
``'dsus'``              Susceptance perturbation step (NR only)                           ``1e-4``
``'solver'``            OPTALG optimization solver ``{'nr','inlp','augl','ipopt'}``       ``'nr'``
``'quiet'``             Flag for disabling output                                         ``False``
======================= ================================================================= =================

.. _ac_opf:

ACOPF
=====

This method is represented by an object of type :class:`ACOPF <gridopt.power_flow.ac_opf.ACOPF>` and solves an AC optimal power flow problem. For doing this, it uses the |AUGL|, |INLP|, or |IPOPT| solver from |OPTALG|. By default, it minimizes |FunctionGEN_COST| subject to voltage magnitude limits, generator power limits, *e.g.*, |ConstraintBOUND|, and |ConstraintACPF|.

The parameters of this optimal power flow method are the following:

==================== ============================================================ ===========
Name                 Description                                                  Default
==================== ============================================================ ===========
``'weight_cost'``    Weight for active power generation cost                      ``1e0``
``'weight_vmag'``    Weight for bus voltage magnitude regularization              ``0.``
``'weight_vang'``    Weight for bus voltage angle regularization                  ``0.``
``'weight_pq'``      Weight for generator power regularization                    ``0``
``'weight_t'``       Weight for tap ratios regularization                         ``0``
``'weight_b'``       Weight for shunt susceptance regularization                  ``0``
``'thermal_limits'`` Branch thermal limits ``{'none','linear','nonlinear'}``      ``'none'``
``'vmin_thresh'``    Low-voltage threshold                                        ``1e-1``
``'solver'``         OPTALG optimization solver ``{'augl','inlp','ipopt'}``       ``'inlp'``
``'quiet'``          Flag for disabling output                                    ``False``
==================== ============================================================ ===========
