Version 1.5.0
-------------
* Added variables on Inter-Area Transfer.
* Added constraints and heuristics on regulating inter-area transfer MW.
* Added transformer P, Q control constraints and heuristics in NR.
* Added transformer P, Q control functions in Opt-based ACPF.
* Added phase shift variables for phase shifters.
* Added asymmetric transformer equation on ratio and phase angle in Opt-based.
* Added support for fixed power factor generator controls (e.g., wind machines).
* Added support for Q limit regulation from generator power factors (e.g., wind machines).
* Add FACTS constraints with for series voltage control and series impedance control.
* Added `loads_2_ZIP` parameter for ACPF to convert loads to constant impedance if voltage drops below threshold (default is False).
* Adding admittance correction parameter `y_correction` and callback to ACPF Newton-Raphson only.
* Added tap-changer rounding parameter `taps_round` that will round taps each step if tap-mode is regulating.
* Moved targeted support to Python 3.10
* Move from nosetest to pytest
* Add tox support for py37, py38, py39, py310
* Added KLU for DCPF
* Changed the switched shunt ACPF from the band regulation using PFNET to a callback function that mimics PSSE that includes switched shunt operation limit check to control for oscillations.

Version 1.4.1
-------------
* Updated setup.py to use compiled files with `bdist_wheel_compiled` and `bdist_egg_compiled`.
* Added `load_q_curtail` parameter setting to ACPF that allows for load.Q to deviate.
* Change `gens_redispatch` as True that forced changing slack to redispatchable.
* Separate ACPF.solve into steps for `initialize_problem` and `solve_problem` in order to reduce computation time with contructing the problem for contingencies or other calculations where the problem construction does not change.

Version 1.4.0
-------------
* Updated ACPF to treat slack bus voltage mag as variable.
* ACPF includes CSC constant current control.
* Added voltage clipping to acpf.
* Added gen P participation constr to opt-base acpf (for consistency).
* All methods automatically merge buses when creating network copy.
* Overwrote barrier parameter bound for ACPF with augl.
* Shunts mode in ACPF.
* Support for network components being in or out of service.
* v_mag_warm_ref parameter to ACPF method.
* Improved ACPF parameters and new options (e.g. gens_redispatch)

Version 1.3.6
-------------
* Added slack P participation to non-nr acpf for consistency.
* Updated to modular heuristic.
* Made unittests handling of test cases and solution files cross-platform.
* Made test utils more flexible to be used from outside unittests.
* Added HVDC, FACTS and vdep-loads to ACPF.
* Added option for using linearized AC power flow equations in ACPF.
* Specified gridopt command-line utility using console entry point.
* Added command-line utility option for writing network to file.

Version 1.3.5
-------------
* Changed ACPF based on NR to use new PVPQ switching constraint from PFNET.
* Added pvpq_start_k parameter to ACPF to control start iteration of PVPQ switching heuristics.
* Updated ACPF method class to work as a parent method class.
* Made thermal_limits param for ACOPF take on values 'none', 'nonlinear', and 'linear' (lineflow).

Version 1.3.4
-------------
* Reduced width of ACOPF output per iteration.
* Updated ACOPF regularization param names to be more consistent with those of ACPF.
* Hid internal param data structure from user in power flow methods, and allowed saving/restoring params including optsolver parameters.
* Changed method results and mechanism for storing solution and updating network.
* Moved location of command-line unitlity.
* Fixed bug in NR ACPF related to tap_changers vs tap_changers_v.
* Changed travis to get pfnet and optalg from pypi.
* Added "solver name" to method results.
* Changed "optsolver" to "solver" and added "solver" to DCPF.

Version 1.3.3
-------------
* Unified methods (ACPF, ACOPF, DCPF, DCOPF) with param "optsolver" for choosing optimization solver.
* Added documentation for "gridopt" command-line utility.
* Updated and re-enabled DCOPF unittests.
* Added new OPTALG solver "inlp" to ACPF and ACOPF and updated documentation.
* limit_gens flag for non-NR ACPF with unittest.

Version 1.3.2
-------------
* Compatibility with PFNET v1.3.0.
* Compatibility with OPTALG v1.1.3.
* Thermal limits option for AugL and Ipopt OPFs.

Version 1.3.1
-------------
* Compatibility with PFNET 1.2.9.
* Tool for transforming PFNET to OPTALG problems and recovering primal and dual variables.
* Standard OPF problem formulation.
* OPF method based on AugL uses linear variable bounds and handles them with barrier.
* OPF method based on IPOPT solver.
* Flat start option in gridopt script.
* Travis continuous integration.
* Readthedocs.

Version 1.3
-----------
* Compatibility with multiperiod pfnet (dc_opf, nr_pf, augL_pf and augL_opf)

Version 1.2
-----------
* Multi-stage stochastic DCOPF.
* Variable generator power curtailment in DCOPF.
* Python 2 and 3 compatibility.

Version 1.1
-----------
* Multi-period DCOPF.
* Corrective DCOPF.
* Python 3 and Jupyter compatibility.
* Updated Dockerfile.
* Elastic loads in DCOPF.
* Thermal limits flag in DCOPF.

Version 1.0
-----------
* Initial version.
