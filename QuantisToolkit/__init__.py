# =================================================================================================================
# MAIN REFERENCE: 
# "MODEL-BASED THREE-MATERIAL DECOMPOSITINO IN DUAL-ENERGY CT USING THE VOLUME CONSERVATION CONSTRAINT"
# PHYSICS IN MEDICINE AND BIOLOGY (2022)
# STEPHEN Z. LIU, MATTHEW TIVNAN, GREG M. OSGOOD, JEFFREY H. SIEWERDSEN, J. WEBSTER STAYMAN, WOJCIECH ZBIJEWSKI
# JOHNS HOPKINS UNIVERSITY, BALTIMORE, MD 21205, USA
# =================================================================================================================

import numpy as np
import aiairecon
import platform
import CudaTools
import time
import multiprocessing
from aiairecon_cudatools.matrices.system import cudatools_projector
if platform.system() != 'Windows':
	from aiairecon_cudatools.matrices.system import separablefootprints
	from aiairecon_cudatools.matrices.system import siddon
from scipy import io
from scipy import interpolate
from scipy import optimize
from scipy import signal
from scipy import linalg
from itertools import repeat
from PythonTools import progress
from aiairecon_cudatools.penalties import PnormPenalty
from PythonTools.itertools import Nesterovify05
from . import optimization



class spectralModel:
	# =======================================
	# [THIS PART MODIFIED FROM AIAISPECTRAL PACKAGE]
	# FORWARD MODEL
	# yi  =  BGSexp(-QAxi)
	# =======================================
	def __init__(self,
		     nVoxelX, # NUMBER OF VOXELS IN X
		     nVoxelY, # NUMBER OF VOXELS IN Y
		     nVoxelZ, # NUMBER OF VOXELS IN Z
		     nMaterial, # NUMBER OF MATERIALS
		     nEnergy, # NUMBER OF ENERGY BINS
		     nPixelU, # NUMBER OF PIXELS IN U
		     nPixelV, # NUMBER OF PIXELS IN V
		     nView, # NUMBER OF VIEWS
		     voxelSpacingX, # VOXEL SIZE IN X (MM)
		     voxelSpacingY, # VOXEL SIZE IN Y (MM)
		     voxelSpacingZ, # VOXEL SIZE IN Z (MM)
		     pixelSpacingU, # PIXEL SIZE IN U (MM)
		     pixelSpacingV, # PIXEL SIZE IN V (MM)
		     y = None, # MEASUREMENT (1 X NVIEW X NPIXELV X NPIXELU)
		     x0 = None, # INITIALIZATION (NMATERIAL X NVOXELZ X NVOXELY X NVOXELX)
		     projectorName = 'SeparableFootprints', # PROJECTOR TYPE
		     Emin = 1.0, # LOWEST ENERGY BIN
		     energySpacing = 1.0, # ENERGY BIN SPACING
		    ):

		self.nVoxelX = nVoxelX
		self.nVoxelY = nVoxelY 
		self.nVoxelZ = nVoxelZ 
		self.nMaterial = nMaterial 
		self.nEnergy = nEnergy 
		self.nPixelU = nPixelU 
		self.nPixelV = nPixelV 
		self.nView = nView 
		self.voxelSpacingX = voxelSpacingX 
		self.voxelSpacingY = voxelSpacingY 
		self.voxelSpacingZ = voxelSpacingZ 
		self.pixelSpacingU = pixelSpacingU 
		self.pixelSpacingV = pixelSpacingV 
		self.y = y 
		self.x0	= x0 
		self.projectorName = projectorName	
		self.Emin = Emin 
		self.energySpacing = energySpacing	

		self.sourceSpectra = None
		self.preFilterSpectra = None
		self.interactionSpectra = None
		self.conversionSpectra = None
		self.massAttenuationSpectra = None
		self.gains = None
		self.subsetslice = slice(None)
		self.clear_secondary_vars(reset_hessian = True)
		self.A = None
		self.rec = None
		self.Q = None
		self.S = None
		self.G = None
		self.B = None
		self.Sigmay_inv = None
		self.projMat = None

		if self.y is None: self.y = 5e5 * np.ones([1, self.nView, self.nPixelV, self.nPixelU], dtype = np.float32)
		if self.x0 is None: self.x0 = 0.0 * np.ones([self.nMaterial, self.nVoxelZ, self.nVoxelY, self.nVoxelX] , dtype = np.float32)
		if self.B is None: self.B = aiairecon.matrices.diagonal.Identity()
		if self.K is None: self.K = aiairecon.matrices.diagonal.Identity()

		self.Regularizer = None
		self.xi = self.x0
		self.Emax = self.Emin + self.nEnergy * self.energySpacing
		self.subsetRatio = 1.0




	def copy(self):
		tmp = spectralModel(self.nVoxelX,
				    self.nVoxelY,
				    self.nVoxelZ,
				    self.nMaterial,
				    self.nEnergy,
				    self.nPixelU,
				    self.nPixelV,
				    self.nView,
				    self.voxelSpacingX,
				    self.voxelSpacingY,
				    self.voxelSpacingZ,
				    self.pixelSpacingU,
				    self.pixelSpacingV,
				    y = self.y,
				    x0 = self.x0,
				    projectorName = self.projectorName,
				    Emin = self.Emin,
				    energySpacing = self.energySpacing)

		if self.projMat is not None: tmp.set_projectionMatrices_direct(self.projMat)
		if self.A is not None: tmp.make_A()
		if self.Q is not None: tmp.make_Q(self.massAttenuationSpectra)
		if self.S is not None:
		    if self.preFilterSpectra is not None:
			    tmp.make_S(self.sourceSpectra, self.preFilterSpectra, self.interactionSpectra, self.conversionSpectra)
		    else:
			    tmp.make_S_StraightForward(self.sourceSpectra)

		if self.G is not None: tmp.make_G(self.gains)
		if self.B is not None: tmp.make_B(self.B)
		if self.Sigmay_inv is not None: tmp.make_Sigmay_inv(self.Sigmay_inv)

		return tmp




	def set_projectionMatrices_direct(self, 
					  projection_matrices):

		self.projMat = projection_matrices




	def set_projectionMatrices_circular(self,
					    cetas,
					    SAD,
					    SDD,
					    fpd_offset_u,
					    fpd_offset_v,
					    src_offset_u,
					    src_offset_v):

		self.projMat = np.array([aiairecon.geometry.circular_projmat(SDD,
									     SAD, 
									     fpd_offset_u,
									     fpd_offset_v, 
									     src_offset_u, 
									     src_offset_v, 
									     0, 0, 
									     ceta) 
					 for ceta in cetas])




	def clear_secondary_vars(self, 
				 reset_hessian = False):

		self.z = None
		self.yi = None
		self.y_tilde = None
		self.y_tilde_tilde = None
		self.R = None
		self.gradientR = None
		self.L = None
		self.gradientL = None
		self.gradient = None
		self.curvatureL = None
		self.curvatureR = None
		self.Grad = None

		if (reset_hessian):
			self.H = None
			self.invH = None
			self.K = None
			self.invK = None
			self.hessianR = None
			self.hessianL = None
			self.hessian = None
			self.curvatureR = None
			self.curvatureL = None
			self.Hess = None




	def set_xi(self, 
		   xi):

		self.xi = xi
		self.xi.shape = self.nMaterial, self.nVoxelZ, self.nVoxelY, self.nVoxelX
		self.clear_secondary_vars()




	def set_y(self,
		  y):

		self.y = y
		self.y.shape = 1, self.nView, self.nPixelV, self.nPixelU
		self.clear_secondary_vars(reset_hessian=True)




	def set_subsetslice(self, 
			    subsetslice):

		self.subsetslice = subsetslice




	def make_A(self):
		
		if ((self.projectorName == 'CudaTools') and (self.rec is None)): self.rec = CudaTools.Reconstruction()
		self.volAffine = aiairecon.Image.VolumeAffine((self.nVoxelZ, self.nVoxelY, self.nVoxelX),
							      spacing = [self.voxelSpacingZ, self.voxelSpacingY, self.voxelSpacingX])

		self.projAffine = aiairecon.Image.ProjectionsAffine((self.nView, self.nPixelV, self.nPixelU), 
								    self.projMat,
								    spacing = [self.pixelSpacingU, self.pixelSpacingV])

		if (self.projectorName == 'CudaTools'): A1_singleMaterial = cudatools_projector(self.rec, self.volAffine, self.projAffine)
		elif (self.projectorName == 'SeparableFootprints'): A1_singleMaterial = separablefootprints(self.volAffine, self.projAffine)
		elif (self.projectorName == 'Siddon'): A1_singleMaterial = siddon(self.volAffine, self.projAffine)
		else: raise Exception('INVALID PROJECTOR NAME!!!')
		
		A2_singleMaterial = aiairecon.matrices.expansion.DuplicateAndScale(np.ones((1, 1, 1, 1), dtype = np.float32))
		self.A_singleMaterial = aiairecon.matrices.composite.TransposableLinearSeries((A2_singleMaterial, A1_singleMaterial))

		A1 = aiairecon.matrices.composite.BlockDiagonal([A1_singleMaterial] * self.nMaterial)
		A2 = aiairecon.matrices.expansion.DuplicateAndScale(np.ones((1, 1, 1, 1, 1), dtype = np.float32))
		self.A = aiairecon.matrices.composite.TransposableLinearSeries((A2, A1))
		self.clear_secondary_vars(reset_hessian=True)




	def make_Q(self, 
		   massAttenuationSpectra = None, 
		   materialsList = None):

		self.massAttenuationSpectra = massAttenuationSpectra
		self.massAttenuationSpectra.shape = self.nEnergy, 1, self.nMaterial, 1, 1, 1
		#self.Q_singleProjection = aiairecon.matrices.dense.NdArrayWrapper(self.massAttenuationSpectra)
		Q1 = aiairecon.matrices.expansion.DuplicateAndScale(self.massAttenuationSpectra)
		Q2 = aiairecon.matrices.reduction.AxisSum(axis = 2, axis_len=self.nMaterial)
		Q3 = aiairecon.matrices.unitary.Reordering([1, 0, 2, 3, 4])
		self.Q = aiairecon.matrices.composite.TransposableLinearSeries((Q3, Q2, Q1))

		self.clear_secondary_vars(reset_hessian = True)



	def make_Q_singleMaterial(self, 
		   massAttenuationSpectra = None, 
		   materialsList = None):

		massAttenuationSpectra.shape = self.nEnergy, 1, 1, 1, 1, 1
		#self.Q_singleProjection = aiairecon.matrices.dense.NdArrayWrapper(massAttenuationSpectra)
		Q1 = aiairecon.matrices.expansion.DuplicateAndScale(massAttenuationSpectra)
		Q2 = aiairecon.matrices.reduction.AxisSum(axis = 2, axis_len = 1)
		Q3 = aiairecon.matrices.unitary.Reordering([1, 0, 2, 3, 4])
		self.Q_singleMaterial = aiairecon.matrices.composite.TransposableLinearSeries((Q3, Q2, Q1))



	def make_S(self, 
		   sourceSpectra, 
		   preFilterSpectra, 
		   interactionSpectra, 
		   conversionSpectra):

		self.sourceSpectra = sourceSpectra.copy()
		self.preFilterSpectra = preFilterSpectra.copy()
		self.interactionSpectra = interactionSpectra.copy()
		self.conversionSpectra = conversionSpectra.copy()

		S0a = aiairecon.matrices.diagonal.ArrayScale(sourceSpectra)
		S0b = aiairecon.matrices.diagonal.ArrayScale(preFilterSpectra)
		self.S0 = aiairecon.matrices.composite.TransposableLinearSeries((S0b, S0a))

		self.S1 = aiairecon.matrices.diagonal.ArrayScale(interactionSpectra)

		S2a = aiairecon.matrices.diagonal.ArrayScale(conversionSpectra)
		S2b = aiairecon.matrices.reduction.AxisSum(axis = 1, axis_len = self.nEnergy)
		self.S2 = aiairecon.matrices.composite.TransposableLinearSeries((S2b, S2a))

		fullSpectra = sourceSpectra[np.newaxis] * preFilterSpectra[np.newaxis] * interactionSpectra * conversionSpectra
		Sa = aiairecon.matrices.diagonal.ArrayScale(fullSpectra)
		Sb = aiairecon.matrices.reduction.AxisSum(axis = 1, axis_len = self.nEnergy)
		self.S = aiairecon.matrices.composite.TransposableLinearSeries((Sb, Sa))

		self.clear_secondary_vars(reset_hessian=True)




	def make_S_StraightForward(self,
		   fullSpectra):
		
		Sa = aiairecon.matrices.diagonal.ArrayScale(fullSpectra)
		Sb = aiairecon.matrices.reduction.AxisSum(axis = 1, axis_len = self.nEnergy)
		self.S = aiairecon.matrices.composite.TransposableLinearSeries((Sb, Sa))
		self.clear_secondary_vars(reset_hessian=True)



	def make_S_KnownComponent(self, 
		   fullSpectra,
		   knownComponent):
		
		knownComponent.shape = self.nVoxelZ, self.nVoxelY, self.nVoxelX
		knownComponent = self.A_singleMaterial.dot(knownComponent)
		knownComponent = self.Q_singleMaterial.dot(knownComponent)
		knownComponent = np.exp(-1 * knownComponent)
		Sa = aiairecon.matrices.diagonal.ArrayScale(fullSpectra * knownComponent)
		Sb = aiairecon.matrices.reduction.AxisSum(axis = 1, axis_len = self.nEnergy)
		self.S = aiairecon.matrices.composite.TransposableLinearSeries((Sb, Sa))



	def make_G(self, 
		   gains = None, 
		   uniformGain = None):

		if not (uniformGain is None): gains = uniformGain * np.ones([1, self.nView, self.nPixelV, self.nPixelU], dtype = np.float32)
		self.G = aiairecon.matrices.diagonal.ArrayScale(gains)
		self.clear_secondary_vars(reset_hessian=True)




	def make_B(self, 
		   B = None):

		if B is None:
			B = aiairecon.matrices.diagonal.Identity()
		
		self.B = B
		self.clear_secondary_vars(reset_hessian=True)




	def make_Sigmay_inv(self, 
						Sigmay_inv = None):

		if Sigmay_inv is None:
			Sigmay_inv = aiairecon.matrices.diagonal.Identity()

		self.Sigmay_inv = Sigmay_inv

		self.clear_secondary_vars(reset_hessian=True)




	def make_z(self):

		if self.z is not None:
			return self.z

		tmp = self.xi.copy()
		tmp = self.A.dot(tmp, subsetslice = self.subsetslice)
		tmp = self.Q.dot(tmp, subsetslice = self.subsetslice)
		tmp = np.exp(-tmp)
		self.z = tmp.copy()

		tmp = np.zeros([1, self.nEnergy, self.nView, self.nPixelV, self.nPixelU], dtype = np.float32)
		tmp[:, :, self.subsetslice] = self.z

		self.Dz = aiairecon.matrices.diagonal.ArrayScale(tmp)
		self.W = aiairecon.matrices.composite.TransposableLinearSeries((self.B, self.G, self.S, self.Dz))

		return self.z.copy()




	def make_yi(self):

		self.make_z()
		if self.yi is None:
			tmp = self.z.copy()
			tmp = self.S.dot(tmp, subsetslice = self.subsetslice)
			tmp = self.G.dot(tmp, subsetslice = self.subsetslice)
			tmp = self.B.dot(tmp, subsetslice = self.subsetslice)
			self.yi = tmp.copy()

		return self.yi.copy()




	def OSPCIP_make_regularization_pnorm(self,
					     betas,
					     pnorm = 2,
					     delta = 0.001):

		self.Reg = PnormPenalty(self.volAffine.shape, 
					self.rec, 
					np.squeeze(np.array([betas])).tolist(),
					pnorm = pnorm, 
					delta = delta)

		gradientR, hessianR = self.Reg(self.xi)
		self.gradientR = gradientR
		self.hessianR = hessianR




	def OSPCIP_make_Aones(self, 
			      hessianSimple = False):

		if hessianSimple:
			tmp = 0.0 * self.xi.copy() + 1.0
			tmp = self.Q.dot(self.A.dot(tmp))
			self.Aones = tmp.copy()

		else:
			Aones = ()
			for jj in np.arange(0, self.nMaterial):
				tmp = 0.0 * self.xi.copy()
				tmp[jj] = 1.0
				tmp = self.Q.dot(self.A.dot(tmp))
				Aones = Aones + (tmp,)
			self.Aones = Aones.copy()




	def OSPCIP_make_Grad(self, 
			     GradR):
		
		if self.Grad is None:
			self.make_yi()
			tmp = self.y[:,self.subsetslice] - self.yi
			tmp = self.Sigmay_inv.dot(tmp, subsetslice = self.subsetslice)
			tmp = self.W.Tdot( tmp, subsetslice = self.subsetslice)
			tmp = self.Q.Tdot( tmp, subsetslice = self.subsetslice)
			tmp = self.A.Tdot( tmp, subsetslice = self.subsetslice)
			tmp = tmp * self.subsetslice.step
			self.Grad = tmp.copy() + GradR

		return self.Grad




	def OSPCIP_make_Hess(self, 
			     HessR, 
			     hessianSimple = False, 
			     perturb = False, 
			     precompute = False):

		self.Hess = np.zeros([self.nMaterial, self.nVoxelZ, self.nVoxelY, self.nVoxelX], dtype = np.float32)
		if hessianSimple:
			if precompute:
				tmp = self.Aones[:, :, self.subsetslice].copy()
			else:
				tmp = 0.0 * self.xi.copy() + 1.0
				tmp = self.A.dot(tmp, subsetslice = self.subsetslice)
				tmp = self.Q.dot(tmp, subsetslice = self.subsetslice)

			tmp = self.W.dot(tmp, subsetslice = self.subsetslice)
			tmp = self.Sigmay_inv.dot(tmp, subsetslice = self.subsetslice)
			tmp = self.W.Tdot(tmp, subsetslice = self.subsetslice)
			tmp = self.Q.Tdot(tmp, subsetslice = self.subsetslice)
			tmp = self.A.Tdot(tmp, subsetslice = self.subsetslice)
			self.Hess = tmp.copy()

		else:
			for iMaterial in np.arange(0, self.nMaterial):
				if precompute:
					tmp = self.Aones[iMaterial].copy()
					tmp = tmp[:, :, self.subsetslice]
				else:
					tmp = 0.0 * self.xi.copy()
					tmp[iMaterial] = 1.0
					tmp = self.A.dot(tmp, subsetslice = self.subsetslice)
					tmp = self.Q.dot(tmp, subsetslice = self.subsetslice)
		
				tmp = self.W.dot(tmp, subsetslice = self.subsetslice)
				tmp = self.Sigmay_inv.dot(tmp, subsetslice = self.subsetslice)
				tmp = self.W.Tdot(tmp, subsetslice = self.subsetslice)
				tmp = self.Q.Tdot(tmp, subsetslice = self.subsetslice)
				tmp = self.A.Tdot(tmp, subsetslice = self.subsetslice)
				self.Hess[iMaterial] += np.sum(tmp.copy(), axis = 0)
		
		if perturb:
			Hess_min = np.min(self.Hess)
			if Hess_min == 0.0:
				self.Hess += 1e-3
			elif Hess_min < 0.0:
				self.Hess += 1.2 * (-1 * Hess_min)

		self.Hess *= self.subsetslice.step
		self.Hess += HessR

		return self.Hess




	def OSPCIP_make_SPS(self, 
			    precompute = False):
        
        # ============================================================================
        # COMPUTE GRADIENT AND HESSIAN OF OBJECTIVE SPS
        # SEE DERIVATION AND PSEUDOCODE IN:
        # [TILLEY ET AL., IEEE TRANS. MED. IMAG., 37(4), 988-999 (2017)]
        # ============================================================================

		if not precompute:
			self.bigA = aiairecon.matrices.composite.TransposableLinearSeries((self.Q, self.A))
			self.bigB = aiairecon.matrices.composite.TransposableLinearSeries((self.B, self.G, self.S))
			self.BTWB = aiairecon.matrices.composite.BTKinvB_basic(self.bigB, aiairecon.matrices.diagonal.ArrayScale(1 / self.Sigmay_inv.array))
			self.BTWy = self.bigB.Tdot(self.Sigmay_inv.dot(self.y))
			self.A1 = self.bigA.dot(np.ones([self.nMaterial, self.nVoxelZ, self.nVoxelY, self.nVoxelX], dtype = np.float32))
			self.BTWB1 = self.BTWB.dot(np.ones([self.nEnergy, self.nView, self.nPixelV, self.nPixelU], dtype = np.float32))
			self.sps = aiairecon.optimization.NLPWLS(self.y, 
								 self.bigA, 
								 self.BTWB, 
								 self.Reg, 
								 self.A1, 
								 BTWy = self.BTWy, 
								 BTWB1 = self.BTWB1, 
								 gpu_curvature_num = 0)

			self.precompute = True  # ONLY PRECOMPUTE FOR THE FIRST SUBITERATION
			print('JUST PRECOMPUTED!')

		gradL, curvL, gradR, curvR = self.sps.SPS_coefficients(self.xi, 
								       subsetslice = self.subsetslice, 
								       M = self.subsetslice.step)
		
		self.gradientL = gradL
		self.gradientR = gradR
		self.Grad = gradL + gradR
		self.Hess = curvL + curvR

		return self.Grad, self.Hess




	def OSPCIP_make_Predictor(self, 
				  maskOutside = True):

        # ============================================================================
        # COMPUTE PREDICTOR STEPS IN CMBMD PARTITIONS 
        # SEE DERIVATION AND PSEUDOCODE IN:
        # [LIU ET AL., PHYS. MED. BIOL., IN PRESS (2022)]
        # ============================================================================

		step_xi = np.zeros_like(self.xi, dtype = np.float32)
		step_Lamda = np.zeros_like(self.Lamda, dtype = np.float32)
		step_Theta = np.zeros_like(self.Theta, dtype = np.float32)

		# PARTITIONING -- UPDATING CONSTRAINED SUBSETS
		step_xi[..., self.mask < 0.5] += 1 / self.Hess[..., self.mask < 0.5]
		step_xi[..., self.mask < 0.5] *= self.Grad[..., self.mask < 0.5] * (-1)
		if maskOutside: step_xi[-1, ...] *= 0.0

		# PARTITIONING -- UPDATING UNCONSTRAINED SUBSETS
		self.zetta = self.Theta / np.maximum(self.xi, 1e-20)
		self.Hess = self.Hess + self.zetta  # NOTE THAT HESSIAN IS NO LONGER THE OBJECTIVE HESSIAN
		self.Grad = -1 * self.Grad + self.Lamda

		step_Lamda[..., self.mask > 0.5] += (1.0 - np.sum(self.xi[..., self.mask > 0.5], axis = 0))
		step_Lamda[..., self.mask > 0.5] -= np.sum(self.Grad[..., self.mask > 0.5] / self.Hess[..., self.mask > 0.5], axis = 0)
		step_Lamda[..., self.mask > 0.5] /= np.sum(1 / self.Hess[..., self.mask > 0.5], axis = 0)
		step_xi[..., self.mask > 0.5] += (self.Grad[..., self.mask > 0.5] + step_Lamda[..., self.mask > 0.5]) / self.Hess[..., self.mask > 0.5]
		step_Theta[..., self.mask > 0.5] -= (self.zetta[..., self.mask > 0.5] * step_xi[..., self.mask > 0.5] + self.Theta[..., self.mask > 0.5])	

		return step_xi, step_Lamda, step_Theta




	def OSPCIP_make_Corrector(self, 
				  step_xi, 
				  step_Lamda, 
				  step_Theta, 
				  sigma = 1.0):

        # ============================================================================
        # COMPUTE CENTERED-CORRECTOR STEPS IN CMBMD PARTITIONS 
        # SEE DERIVATION AND PSEUDOCODE IN:
        # [LIU ET AL., PHYS. MED. BIOL., IN PRESS (2022)]
        # ============================================================================

		gamma = np.zeros_like(step_xi, dtype = np.float32)
		gamma[..., self.mask > 0.5] = step_xi[..., self.mask > 0.5] * step_Theta[..., self.mask > 0.5] - sigma * self.OSPCIP_xi
		gamma[..., self.mask > 0.5] /= np.maximum(self.xi[..., self.mask > 0.5], 1e-20)

		step_Lamda[..., self.mask > 0.5] = 1.0 - np.sum(self.xi[..., self.mask > 0.5], axis = 0)
		step_Lamda[..., self.mask > 0.5] -= np.sum((self.Grad[..., self.mask > 0.5] - gamma[..., self.mask > 0.5]) / self.Hess[..., self.mask > 0.5], axis = 0)
		step_Lamda[..., self.mask > 0.5] /= np.sum(1 / self.Hess[..., self.mask > 0.5], axis = 0)
		step_xi[..., self.mask > 0.5] = (self.Grad[..., self.mask > 0.5] - gamma[..., self.mask > 0.5] + step_Lamda[..., self.mask > 0.5]) / self.Hess[..., self.mask > 0.5]
		step_Theta[..., self.mask > 0.5] = -1 * gamma[..., self.mask > 0.5] - self.zetta[..., self.mask > 0.5] * step_xi[..., self.mask > 0.5] - self.Theta[..., self.mask > 0.5]

		return step_xi, step_Lamda, step_Theta




	def OSPCIP_make_CentralPath(self, 
				    alpha_prim, 
				    alpha_dual, 
				    step_xi, 
				    step_Lamda, 
				    step_Theta):

	# ============================================================================
        # COMPUTE CENTERING PARAMETER (SIGMA) IN CMBMD PARTITIONS 
        # SEE DERIVATION AND PSEUDOCODE IN:
        # [LIU ET AL., PHYS. MED. BIOL., IN PRESS (2022)]
        # NOTE THAT SIGMA IS RELATED TO TAU (BARRIER STRENGTH)
        # ============================================================================

		self.OSPCIP_xi = np.sum(self.xi[..., self.mask > 0.5] * self.Theta[..., self.mask > 0.5])
		self.OSPCIP_xi /= (self.nMaterial * self.mask_count)
		self.OSPCIP_xi = np.maximum(self.OSPCIP_xi, 1e-200)

		OSPCIP_xi_aff = np.sum((self.xi[..., self.mask > 0.5] + alpha_prim * step_xi[..., self.mask > 0.5]) * (self.Theta[..., self.mask > 0.5] + alpha_dual * step_Theta[..., self.mask > 0.5]))
		OSPCIP_xi_aff /= (self.nMaterial * self.mask_count)
		sigma = (OSPCIP_xi_aff / self.OSPCIP_xi) ** 3 # EMPIRICAL CUBE

		return sigma, OSPCIP_xi_aff, self.OSPCIP_xi




	def OSPCIP_set_PrimalDualVars(self, 
				      xxx):

		self.xi = xxx[0 : self.nMaterial, ...].reshape(self.nMaterial, self.nVoxelZ, self.nVoxelY, self.nVoxelX)
		self.Lamda = xxx[self.nMaterial, ...].reshape(1, self.nVoxelZ, self.nVoxelY, self.nVoxelX)
		self.Theta = xxx[self.nMaterial + 1 : , ...].reshape(self.nMaterial, self.nVoxelZ, self.nVoxelY, self.nVoxelX)
		self.clear_secondary_vars()




	def OSPCIP_set_mask(self, 
			    mask):

		self.mask = mask.copy()
		mask = np.minimum(mask, 1)
		self.mask_count = np.squeeze(np.sum(mask))
		self.mask.shape = self.nVoxelZ, self.nVoxelY, self.nVoxelX

