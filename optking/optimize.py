import numpy as np
import copy
from pprint import PrettyPrinter

from psi4.driver import json_wrapper
import hessian
import stepAlgorithms
import caseInsensitiveDict
import optparams as op
import optExceptions
import addIntcos
import history
import intcosMisc
import convCheck
import testB
import IRCFollowing
import atomData
import psi4methods
from qcdbjson import jsonSchema
from printTools import printGeomGrad, \
                       printMat, \
                       printArray, \
                       welcome, \
                       print_opt, \
                       generate_file_output

pp = PrettyPrinter(indent=4)

def optimize(oMolsys, options_in, json_in=None):
    """Logical 'driver' for optking's optimization procedure takes in optking's moolecular system
    a list of options for both the optimizer and the QM program with the options of passing a 
    json_file in directly.  
    """

    try:
        userOptions = caseInsensitiveDict.CaseInsensitiveDict(options_in)
        origOptions = userOptions.copy()  # Save copy of original user options.

        # Create full list of parameters from user options plus defaults.
        welcome()  # print header
        print_opt("\tProcessing user input options...\n")
        op.Params = op.optParams(userOptions)
        print_opt("\tParameters from optking.optimize\n")
        print_opt(str(op.Params))

        #op.Params.dynamic_level_max = 3

        converged = False

        #generates a json dictionary if optking is not being called directly by json.
        #other option would be to have the wrapper make a JSON object and make json
        #be a requirement of calling optking. If we're adding additional ways to call optking
        #this if json_in is None will be refactored into a method within optimize
        o_json = 0
        if json_in is None:
            QM_method, basis, keywords = psi4methods.collect_psi4_options(options_in)
            atom_list = oMolsys.get_atom_list()
            o_json = jsonSchema.make_qcschema("", atom_list, QM_method, basis, keywords)        
        else:
            o_json = json_in
            op.Params.output_type = 'JSON'

        # For IRC computations:
        ircNumber = 0
        qPivot = None  # Dummy argument for non-IRC

        while not converged:  # may contain multiple algorithms

            try:
                print_opt("Starting optimization algorithm.\n")
                print_opt(str(oMolsys))

                # Set internal or cartesian coordinates.
                if not oMolsys.intcos:
                    C = addIntcos.connectivityFromDistances(oMolsys.geom, oMolsys.Z)

                    if op.Params.frag_mode == 'SINGLE':
                        # Splits existing fragments if they are not connected.
                        oMolsys.splitFragmentsByConnectivity()
                        # Add to connectivity to make sure all fragments connected.
                        oMolsys.augmentConnectivityToSingleFragment(C)
                        #print_opt("Connectivity\n")
                        #printMat(C)
                        # Bring fragments together into one.
                        oMolsys.consolidateFragments()
                    elif op.Params.frag_mode == 'MULTI':
                        # should do nothing if fragments are already split by calling program/constructor.
                        oMolsys.splitFragmentsByConnectivity()

                    if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
                        oMolsys.addIntcosFromConnectivity(C)

                    if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
                        oMolsys.addCartesianIntcos()

                    addIntcos.addFrozenAndFixedIntcos(oMolsys)
                    oMolsys.printIntcos()

                # Special code for first step of IRC.  Compute Hessian and take step along eigenvector.
                #if op.Params.opt_type == 'IRC' and ircNumber == 0:
                #    ircStepList = []  #Holds data points for IRC steps
                #    xyz = oMolsys.geom.copy()
                #    print_opt("Initial Cartesian Geom")
                #    printMat(xyz)
                #    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)
                #    print_opt("Initial internal coordinates\n")
                #    printArray(qZero)
                #    Hcart = get_hessian(xyz, o_json, printResults=False)
                #    e, gX = get_gradient(xyz)
                #    fq = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX)
                #    Hq = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, xyz)
                #    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                #    dqPrime, qPivot, qPrime = IRCFollowing.takeHessianHalfStep(
                #        oMolsys, Hq, B, fq, op.Params.irc_step_size)

                # Loop over geometry steps.
                energies = []  # should be moved into history
                for stepNumber in range(op.Params.geom_maxiter):
                    # compute energy and gradient
                    xyz = oMolsys.geom.copy()
                    E, g_x, nuc = get_gradient(xyz, o_json, printResults=False, nuc=True)
                    oMolsys.geom = xyz  # use setter function to save data in fragments
                    printGeomGrad(oMolsys.geom, g_x)
                    energies.append(E)

                    if op.Params.test_B:
                        testB.testB(oMolsys.intcos, oMolsys.geom)
                    if op.Params.test_derivative_B:
                        testB.testDerivativeB(oMolsys.intcos, oMolsys.geom)

                    if op.Params.print_lvl > 3:
                        B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
                        printMat(B, title="B matrix")

                    fq = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, g_x)
                    if op.Params.print_lvl > 1:
                        printArray(fq, title="Internal forces in au")

                    history.oHistory.append(oMolsys.geom, E, fq)  # Save initial step info.
                    history.oHistory.nuclear_repulsion_energy = nuc
                    # Analyze previous step performance; adjust trust radius accordingly.
                    # Returns true on first step (no history)
                    lastStepOK = history.oHistory.currentStepReport()

                    # If step was bad, take backstep here or raise exception.
                    if lastStepOK:
                        history.History.consecutiveBacksteps = 0
                    else:
                        # Don't go backwards until we've gone a few iterations.
                        if len(history.oHistory.steps) < 5:
                            print_opt("\tNear start of optimization, so ignoring bad step.\n")
                        elif history.History.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                            history.History.consecutiveBacksteps += 1
                            print_opt("\tCalling for consecutive backstep number %d.\n" %
                                       history.History.consecutiveBacksteps)
                            Dq = stepAlgorithms.Dq(oMolsys, E, fq, H, stepType="BACKSTEP")
                            print_opt("\tStructure for next step (au):\n")
                            oMolsys.printGeom()
                            continue
                        elif op.Params.dynamic_level == 0: # not using dynamic level, so ignore.
                            print_opt("\tNo more backsteps allowed.  Dynamic level is off.\n")
                            pass
                        else:
                            raise optExceptions.AlgFail(
                                "Bad step, and no more backsteps allowed.")

                    # Produce guess Hessian or update existing Hessian..
                    if stepNumber == 0:
                        if op.Params.full_hess_every > -1:
                            xyz = oMolsys.geom.copy()
                            Hcart = get_hessian(
                                xyz, o_json, printResults=True)  # don't let function move geometry
                            H = intcosMisc.convertHessianToInternals(
                                Hcart, oMolsys.intcos, xyz, masses = None)  
                        else:
                            H = hessian.guess(oMolsys.intcos, oMolsys.geom, oMolsys.Z, C,
                                              op.Params.intrafrag_hess)
                    else:
                        if op.Params.full_hess_every < 1:  # that is, compute hessian never or only once.
                            history.oHistory.hessianUpdate(H, oMolsys.intcos)
                        elif stepNumber % op.Params.full_hess_every == 0:
                            xyz = oMolsys.geom.copy()
                            Hcart = get_hessian(
                                xyz, o_json,
                                printResults=False)  # it's possible function moves geometry
                            H = intcosMisc.convertHessianToInternals(
                                Hcart, oMolsys.intcos, xyz, masses=None)
                        else:
                            history.oHistory.hessianUpdate(H, oMolsys.intcos)
                        #print_opt("Hessian (in au) is:\n")
                        #printMat(H)
                        #print_opt("Hessian in aJ/Ang^2 or aJ/deg^2\n")
                        #hessian.show(H, oMolsys.intcos)

                        # handle user defined forces, redundances and constraints
                    intcosMisc.applyFixedForces(oMolsys, fq, H, stepNumber)
                    intcosMisc.projectRedundanciesAndConstraints(oMolsys.intcos, oMolsys.geom,
                                                                 fq, H)
                    intcosMisc.qShowValues(oMolsys.intcos, oMolsys.geom)

                    #if op.Params.opt_type == 'IRC':
                    #    xyz = Molsys.geom.copy()
                    #    E, g = get_gradient(xyz, False)
                    #    Dq = IRCFollowing.Dq(oMolsys, g, E, Hq, B, op.Params.irc_step_size,
                    #                         qPrime, dqPrime)
                    #else:  # Displaces and adds step to history.
                    Dq = stepAlgorithms.Dq(oMolsys, E, fq, H, op.Params.step_type, get_energy, o_json) #else statement paried with above IRC if was removed

                    converged = convCheck.convCheck(stepNumber, oMolsys, Dq, fq, energies,
                                                    qPivot)

                    #if converged and (op.Params.opt_type == 'IRC'):
                    #    converged = False
                    #    #add check for minimum
                    #    if atMinimum:
                    #        converged = True
                    #        break
                    if converged: #changed from elif when above if statement active
                        print_opt("\tConverged in %d steps!\n" % (stepNumber + 1))
                        print_opt("\tFinal energy is %20.13f\n" % E)
                        print_opt("\tFinal structure (Angstroms):\n")
                        oMolsys.showGeom()
                        break

                    print_opt("\tStructure for next step (au):\n")
                    oMolsys.printGeom()

                else:  # executes if step limit is reached
                    print_opt("Number of steps (%d) exceeds maximum allowed (%d).\n" %
                              (stepNumber + 1, op.Params.geom_maxiter))
                    history.oHistory.summary()
                    raise optExceptions.AlgFail("Maximum number of steps exceeded.")

                #This should be called at the end of each iteration of the for loop,
                #if (op.Params.opt_type == 'IRC') and (not atMinimum):
                #    ircNumber += 1
                #    xyz = oMolsys.geom.copy()
                #    ircStepsList.append(ircStep.IRCStep(qPivot, xyz, ircNumber))
                #    history.oHistory.hessianUpdate(H, oMolsys.intcos)
                #    Hq = H
                #    E, gX = get_gradient(xyz, printResults=False)
                #    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                #    qPivot, qPrime, Dq = IRCFollowing.takeGradientHalfStep(
                #        oMolsys, E, Hq, B, op.Params.irc_step_size, gX)

            except optExceptions.AlgFail as AF:
                print_opt("\tCaught AlgFail exception\n")
                eraseHistory = False
                eraseIntcos = False

                if AF.linearBends:  # New linear bends detected; Add them, and continue at current level.
                    from . import bend
                    for l in AF.linearBends:
                        if l.bendType == "LINEAR":  # no need to repeat this code for "COMPLEMENT"
                            F = addIntcos.checkFragment(l.atoms, oMolsys)
                            intcosMisc.removeOldNowLinearBend(l.atoms,
                                                              oMolsys._fragments[F].intcos)
                    oMolsys.addIntcosFromConnectivity()
                    eraseHistory = True
                elif op.Params.dynamic_level >= (op.Params.dynamic_level_max - 1):
                    print_opt("\t Current approach/dynamic_level is %d.\n" %
                              op.Params.dynamic_level)
                    print_opt("\t Alternative approaches are not available or turned on.\n")
                    raise optExceptions.OptFail("Maximum dynamic_level exceeded.")
                else:
                    op.Params.dynamic_level += 1
                    print_opt("\t Increasing dynamic_level algorithm to %d.\n" %
                              op.Params.dynamic_level)
                    print_opt("\t Erasing old history, hessian, intcos.\n")
                    eraseIntcos = True
                    eraseHistory = True
                    op.Params.updateDynamicLevelParameters(op.Params.dynamic_level)

                if eraseIntcos:
                    print_opt("\t Erasing coordinates.\n")
                    for f in oMolsys._fragments:
                        del f._intcos[:]

                if eraseHistory:
                    print_opt("\t Erasing history.\n")
                    stepNumber = 0
                    del H
                    del history.oHistory[:]  # delete steps in history
                    history.oHistory.stepsSinceLastHessian = 0
                    history.oHistory.consecutiveBacksteps = 0

                

        output_dict = {}
        # print summary
        if op.Params.output_type == 'FILE':
            generate_file_output()
        else:
           output_dict = o_json.generate_json_output(history.oHistory[-1].geom) 

        if op.Params.trajectory:
            # history doesn't contain atomic numbers so pass them in
            returnVal = history.oHistory.trajectory(oMolsys.Z)
        else:
            returnVal = history.oHistory[-1].E
   
        returnNuc = history.oHistory.nuclear_repulsion_energy
 
        # clean up
        del H
        for f in oMolsys._fragments:
            del f._intcos[:]
            del f
        del history.oHistory[:]
        del oMolsys
        #this is where'd i'd like to add an if statement to potentilly generate an json_file_outputi

        if op.Params.output_type == 'FILE':
            del op.Params
            del o_json
            return returnVal, returnNuc
        else:
            del op.Params
            del o_json
            return output_dict

    except Exception as error:
        print_opt("\tA non-optimization error has occured\n")
        print_opt("""\tResetting all optimzation options to prevent queued optimizations 
                  from failing\n""")
        print(type(error))
        print(error)
        del history.oHistory[:]
        del o_json
        for f in oMolsys._fragments:
            del f._intcos[:]
            del f
        del oMolsys
        del op.Params

def get_gradient(new_geom, o_json, printResults=True, nuc=True):
    """get_gradient gets a gradient from a QM program (only psi4 is currently implemented
    calls psi4.driver.json_wrapper.run_json_qc_schema. Similar method calls can be added later.
    Input:
        new_geom - geometry optimizer is stepping to - type Natom*3 numpy array
        o_json - optking's json object
    returns a energy and gradient with the option to return the nueclear repulsion enrgy as well
    """
    from .printTools import print_opt
    #i may need to add in a line to convert the geometry into json form
    geom = o_json.to_JSON_geom(new_geom)
    json_input = o_json.update_geom_and_driver(geom, 'gradient')
    json_output = json_wrapper.run_json_qc_schema(json_input, True)
    E, g_x, nuclear_rep = o_json.get_JSON_result(json_output, 'gradient', nuc)
    g_x = np.asarray(g_x)
    if nuc:
        return E, g_x, nuclear_rep
    return E, g_x

def get_hessian(new_geom, o_json, printResults=False): 
    """get_hessian gets a hessian from a QM program (currently only psi4 is implemented)
    calls psi4.driver.json_wrapper.run_json_qc_schema. Similar method calls can be added later.
    Input:
        new_geom - geometry optimizer is stepping to - type Natom*3 numpy array
        o_json - optking's json object
    returns the hessian as a 1D numpy array
    """

    geom = o_json.to_JSON_geom(new_geom)
    json_input = o_json.update_geom_and_driver(geom, 'hessian')
    json_output = json_wrapper.run_json_qc_schema(json_input, True)     
    H = np.array(o_json.get_JSON_result(json_output, 'hessian'))
    return hessian.convert_json_hess_to_matrix(H, len(json_output['molecule']['symbols'])) 

def get_energy(new_geom, o_json, printResults=False, nuc=True):
    """get_energy gets a energy from a QM program corresponding to the input method.
    calls psi4.driver.json_wrapper.run_json_qc_schema. Similar method calls can be added later.    
    This is a specialized method only used for linesearching.
    Input:
        new_geom - geometry optimizer is stepping to - type Natom*3 numpy array
        o_json - optking's json object
    returns the energy with the options of returning the nuclear repulsion energy as well
    Get_gradient returns both Energy and Gradient (which is what we usually want)
    """
    
    geom = o_json.to_JSON_geom(new_geom)
    json_input = o_json.update_geom_and_driver(geom, 'energy')
    json_output = json_wrapper.run_json_qc_schema(json_input, True)
    return o_json.get_JSON_result(json_output, 'energy', nuc) 
