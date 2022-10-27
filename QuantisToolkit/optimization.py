# =================================================================================================================
# MAIN REFERENCE: 
# "MODEL-BASED THREE-MATERIAL DECOMPOSITINO IN DUAL-ENERGY CT USING THE VOLUME CONSERVATION CONSTRAINT"
# PHYSICS IN MEDICINE AND BIOLOGY (2022)
# STEPHEN Z. LIU, MATTHEW TIVNAN, GREG M. OSGOOD, JEFFREY H. SIEWERDSEN, J. WEBSTER STAYMAN, WOJCIECH ZBIJEWSKI
#
# AUTHORS: 
# STEPHEN Z. LIU (E-MAIL: SZLIU@JHMI.EDU)
# WOJCIECH ZBIJEWSKI, PH.D. (E-MAIL: WZBIJEWSKI@JHU.EDU)
#
# ADDRESS:
# I-STAR & QUANTIS LABORATORIES
# DEPARTMENT OF BIOMEDICAL ENGINEERING
# JOHNS HOPKINS UNIVERSITY SCHOOL OF MEDICINE
# 720 RUTLAND AVENUE, ROOM 624
# BALTIMORE, MD 21205, USA
# =================================================================================================================

import numpy as np
from itertools import repeat
from PythonTools import progress
from PythonTools.itertools import interleave, callnext
from scipy import optimize,io
from IPython.display import clear_output, display
import aiairecon
import QuantisToolkit
import time




@callnext
def optimizer_OSNR_generator_v0(sm,
                                alpha = 0.9, 
                                hessianUpdateSchedule = 1, 
                                hessianSimple = False, 
                                perturb = False, 
                                precompute = False, 
                                subsetschedule = None):

    # ============================================================================
    # ORDERED-SUBSET NEWTON-RAPHSON (OS-NR) FRAMEWORK 
    # FOR UNCONSTRAINED DECOMPOSITION
    # (NO LINE SEARCH INCLUDED FOR NOW)
    # ============================================================================
    
    x = yield
    
    for M in subsetschedule:
        subsetlist = list(range(M))
        for ss in interleave(subsetlist[:M // 2], subsetlist[M // 2:]): 
            sm.set_xi(x)
            ssinds = slice(ss, sm.nView, M)
            sm.set_subsetslice(ssinds)

            GradR, HessR = sm.Reg(sm.xi)
            sm.OSPCIP_make_Grad(GradR)

            if sm.count == 0:
                sm.OSPCIP_make_Hess(HessR, hessianSimple = hessianSimple, perturb = perturb, precompute = precompute)
                sm.count = hessianUpdateSchedule

            sm.count = sm.count - 1
            xgrad = alpha * (sm.Grad / sm.Hess)
            sm.clear_secondary_vars()

            x = yield xgrad




@callnext
def optimizer_OSSPS_generator_v0(sm, 
                                 alpha = 1.0, 
                                 precompute = False, 
                                 subsetschedule = None):

    # ============================================================================
    # ORDERED-SUBSET SEPARABLE PARABOLIC SURROGATE (OS-SPS) FRAMEWORK 
    # FOR UNCONSTRAINED DECOMPOSITION
    # ============================================================================

    x = yield
    
    sm.precompute = precompute
    for M in subsetschedule:
        subsetlist = list(range(M))
        for ss in interleave(subsetlist[:M // 2], subsetlist[M // 2:]): 
            sm.set_xi(x)
            ssinds = slice(ss, sm.nView, M)
            sm.set_subsetslice(ssinds)

            sm.OSPCIP_make_SPS(precompute = sm.precompute)

            xgrad = alpha * (sm.Grad / sm.Hess)
            sm.clear_secondary_vars()

            x = yield xgrad




@callnext
def optimizer_OSPCIP_generator_v0(sm, 
                                  yeta = 0.995, 
                                  alpha = 0.9, 
                                  hessianUpdateSchedule = 1, 
                                  hessianSimple = False, 
                                  perturb = False, 
                                  precompute = False, 
                                  maskOutside = True, 
                                  subsetschedule = None):

    # ============================================================================
    # ORDERED-SUBSET PREDICTOR-CORRECTOR INTERIOR-POINT (OS-PCIP) FRAMEWORK 
    # FOR CONSTRAINED DECOMPOSITION
    # ============================================================================

    x = yield
    mask = np.tile(sm.mask, [sm.nMaterial, 1, 1, 1])
    
    for M in subsetschedule:
        subsetlist = list(range(M))
        for ss in interleave(subsetlist[:M // 2], subsetlist[M // 2:]): 
            sm.OSPCIP_set_PrimalDualVars(x)
            ssinds = slice(ss, sm.nView, M)
            sm.set_subsetslice(ssinds)

            GradR, HessR = sm.Reg(sm.xi)
            sm.OSPCIP_make_Grad(GradR)

            if sm.count == 0:
                sm.OSPCIP_make_Hess(HessR, hessianSimple = hessianSimple, perturb = perturb, precompute = precompute)
                sm.count = hessianUpdateSchedule
            sm.count = sm.count - 1

            # EVALUATING PREDICTOR STEP
            sxi, slamda, stheta = sm.OSPCIP_make_Predictor(maskOutside = maskOutside)

            if np.min(sxi * mask) < 0.0:
                alphap_aff = np.minimum(1.0, np.squeeze(np.amin(-1 * (sm.xi[sxi * mask < 0.0] / sxi[sxi * mask < 0.0]))))
            else:
                alphap_aff = 1.0

            if np.min(stheta * mask) < 0.0:
                alphad_aff = np.minimum(1.0, np.squeeze(np.amin(-1 * (sm.Theta[stheta * mask < 0.0] / stheta[stheta * mask < 0.0]))))
            else:
                alphad_aff = 1.0

            # EVALUTING CENTERED CORRECTOR STEP
            centering, _, _ = sm.OSPCIP_make_CentralPath(alphap_aff, alphad_aff, sxi, slamda, stheta)
            sxi, slamda, stheta = sm.OSPCIP_make_Corrector(sxi, slamda, stheta, sigma = centering)

            if np.min(sxi * mask) < 0.0:
                alphap = np.minimum(1.0, yeta * np.squeeze(np.amin(-1 * (sm.xi[sxi * mask < 0.0] / sxi[sxi * mask < 0.0]))))
            else:
                alphap = 1.0

            if np.min(stheta * mask) < 0.0:
                alphad = np.minimum(1.0, yeta * np.squeeze(np.amin(-1 * (sm.Theta[stheta * mask < 0.0] / stheta[stheta * mask < 0.0]))))
            else:
                alphad = 1.0

            sxi[mask > 0.5] *= alphap
            slamda[..., sm.mask > 0.5] *= alphad
            stheta[mask > 0.5] *= alphad
            xgrad = -1 * alpha * np.concatenate((sxi, slamda, stheta), axis = 0)
            sm.clear_secondary_vars()

            x = yield xgrad




@callnext
def optimizer_OSSPSPCIP_generator_v0(sm, 
                                     yeta = 0.995, 
                                     alpha = 0.9, 
                                     precompute = False, 
                                     maskOutside = True, 
                                     subsetschedule = None):

    # ==============================================================================================
    # ORDERED-SUBSET PREDICTOR-CORRECTOR INTERIOR-POINT WITH SPS CURVATURE (OS-SPS-PCIP) FRAMEWORK 
    # FOR CONSTRAINED DECOMPOSITION
    # ==============================================================================================

    x = yield
    mask = np.tile(sm.mask, [sm.nMaterial, 1, 1, 1])
    sm.precompute = precompute

    for M in subsetschedule:
        subsetlist = list(range(M))
        for ss in interleave(subsetlist[:M // 2], subsetlist[M // 2:]): 
            sm.OSPCIP_set_PrimalDualVars(x)

            ssinds = slice(ss, sm.nView, M)
            sm.set_subsetslice(ssinds)

            sm.OSPCIP_make_SPS(precompute = sm.precompute)

            # EVALUATING PREDICTOR STEP
            sxi, slamda, stheta = sm.OSPCIP_make_Predictor(maskOutside = maskOutside)

            if np.min(sxi * mask) < 0.0:
                alphap_aff = np.minimum(1.0, np.squeeze(np.amin(-1 * (sm.xi[sxi * mask < 0.0] / sxi[sxi * mask < 0.0]))))
            else:
                alphap_aff = 1.0

            if np.min(stheta * mask) < 0.0:
                alphad_aff = np.minimum(1.0, np.squeeze(np.amin(-1 * (sm.Theta[stheta * mask < 0.0] / stheta[stheta * mask < 0.0]))))
            else:
                alphad_aff = 1.0

            # EVALUATING CENTERED CORRECTOR STEP
            centering, _, _ = sm.OSPCIP_make_CentralPath(alphap_aff, alphad_aff, sxi, slamda, stheta)
            sxi, slamda, stheta = sm.OSPCIP_make_Corrector(sxi, slamda, stheta, sigma = centering)

            if np.min(sxi * mask) < 0.0:
                alphap = np.minimum(1.0, yeta * np.squeeze(np.amin(-1 * (sm.xi[sxi * mask < 0.0] / sxi[sxi * mask < 0.0]))))
            else:
                alphap = 1.0

            if np.min(stheta * mask) < 0.0:
                alphad = np.minimum(1.0, yeta * np.squeeze(np.amin(-1 * (sm.Theta[stheta * mask < 0.0] / stheta[stheta * mask < 0.0]))))
            else:
                alphad = 1.0

            sxi[mask > 0.5] *= alphap
            slamda[..., sm.mask > 0.5] *= alphad
            stheta[mask > 0.5] *= alphad
            xgrad = -1 * alpha * np.concatenate((sxi, slamda, stheta), axis = 0)
            sm.clear_secondary_vars()

            x = yield xgrad

