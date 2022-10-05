#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import signal,io,interpolate
import h5py
import aiairecon
import os
import CudaTools
import QuantisToolkit
import pandas as pd
import os.path
import math
from PythonTools.itertools import Nesterovify05
from itertools import repeat
from PythonTools import progress
from IPython.display import clear_output, display
from aiairecon_cudatools.matrices.system import cudatools_projector
from aiairecon_cudatools.fbp import FDK
from aiairecon_cudatools.penalties import PnormPenalty


# EXAMPLE FOR DUAL-ENERGY TWO-MATERIAL DECOMPOSITION USING SIMPLE SIMULATIONS
# S.Z. LIU, SZLIU@JHMI.EDU


slurm_path = '~result/~mbmd_ossps_tibia'
if not os.path.exists(slurm_path): os.makedirs(slurm_path)


# FILENAMES
class pFNS:
    ProjectionID        = '~data/PROJ_Tibia_60140switch_2e6gain_556pix_380x1dim.mat'  # PROJECTIONA FOR S1, S2, S3 (ASSUME #FRAME X #V X #U)
    InitialID           = '~initialization/START_MBMD_Tibia_60140switch_2e6gain_500vox_300x1x300dim.mat'  # INITIALIZATION
    GeometryID          = ''  # GEOMETRY FILE
    ReconstructionID    = slurm_path + '/MBMD_Tibia_60140switch_2e6gain_500vox_300x1x300dim.mat'  # RECONSTRUCTION (FOR FILE SAVING)
 
class pSYS:
    Flux                = 2e6;  # PHOTON GAIN
    SAD                 = 400;  # SOURCE AXIS DISTANCE (GARBAGE IF USING GEOMETRY FILES)
    SDD                 = 540;  # SOURCE DETECTOR DISTANCE (GARBAGE IF USING GEOMETRY FILES)
    ProjectAngle        = np.linspace(0, 359, 360, endpoint = 'True').astype(np.float);  # PROJECTION ANGLE
    OffsetTubeU         = 0;  # SOURCE OFFSET IN U (GARBAGE IF USING GEOMETRY FILES)
    OffsetTubeV         = 0;  # SOURCE OFFSET IN V (GARBAGE IF USING GEOMETRY FILES)
    OffsetDetectorU     = 0;  # DETECTOR OFFSET IN U (GARBAGE IF USING GEOMETRY FILES)
    OffsetDetectorV     = 0;  # DETECTOR OFFSET IN U (GARBAGE IF USING GEOMETRY FILES)
    RhoScint            = 4.51e-3;  # CSI SCINTILLATOR DENSITY
    ThickScint          = 0.6;  # CSI SCINTOLLATOR THICKNESS
    EnergyCount         = 131;  # NUMBER OF ENERGY BINS
    EnergyMin           = 10;  # SMALLEST ENERGY BIN
    EnergyBin           = 1.0; # WIDTH OF ENERGY BIN
    SpectrumHE          = '~doc/spectrum_140kvp_2000al+250cu+225ag.mat';  # HE SOURCE SPECTRUM
    SpectrumLE          = '~doc/spectrum_60kvp_2000al+250cu.mat';  # LE SOURCE SPECTRUM
    AttenScint          = '~doc/linearatten_csi.mat';  # LINEAR ATTENUATION OF CSI SCINTILLATOR
    Atten1              = '~doc/linearatten_fat.mat';  # LINEAR ATTENUATION OF BASIS 1
    Atten2              = '~doc/linearatten_bone.mat';  # LINEAR ATTENUATION OF BASIS 2

class pIMG:
    ImageX              = 300;  # RECONSTRUCTED IMAGE SIZE (NUMBER OF VOXELS) IN X
    ImageY              = 1;  # RECONSTRUCTED IMAGE SIZE (NUMBER OF VOXELS) IN Y -- NOTE: THIS IS ROTATION AXIS
    ImageZ              = 300;  # RECONSTRUCTED IMAGE SIZE (NUMBER OF VOXELS) IN Z
    VoxelX              = 0.50;  # VOXEL SIZE (MM) IN X
    VoxelY              = 0.50;  # VOXEL SIZE (MM) IN Y
    VoxelZ              = 0.50;  # VOXEL SIZE (MM) IN Z
    
class pPRJ:
    ProjU               = 380;  # PROJECTION SIZE (NUMBER OF PIXELS) IN U
    ProjV               = 1;  # PROJECTION SIZE (NUMBER OF PIXELS) IN V
    PixelU              = 0.556;  # PIXEL SIZE (MM) IN U
    PixelV              = 0.556;  # PIXEL SIZE (MM) IN V
    
class pREC:
    NumIteration        = 300;  # NUMBER OF OUTER ITERATIONS
    NumSubset           = 15;  # NUMBER OF INNER ITERATIONS (ORDERED SUBSETS)
    HuberDelta          = 0.001;  # HUBER PENALTY THRESHOLD
    HuberBeta1          = 1e2;  # HUBER PENALTY STRENGTH FOR BASIS 1
    HuberBeta2          = 1e3;  # HUBER PENALTY STRENGTH FOR BASIS 2
    Steplength          = '';  # PRE-DEFINED STEP LENGTH FOR OS-NR (BETWEEN 0-1) -- GARBAGE FOR OS-SPS
    HessianUpdate       = '';  # NUMBER OF ITERATIONS TO UPDATE HESSIAN -- GARBAGE FOR OS-SPS
    
    
    
# DEFINE FUNCTIONS
def get_atten(fname, minkVp = pSYS.EnergyMin, maxkVp = pSYS.EnergyMin + pSYS.EnergyCount - 1):
    file = io.loadmat(fname)
    atten = np.squeeze(file['atten'].astype(np.float32))
    kvp = np.arange(minkVp, maxkVp + 1)
    atten = atten[(kvp - 1).astype(np.int)]
    return np.stack((kvp, atten), axis = -1)

def get_spectra(fname, minkVp = pSYS.EnergyMin, maxkVp = pSYS.EnergyMin + pSYS.EnergyCount - 1):
    file = io.loadmat(fname)
    spec = np.squeeze(file['spec'].astype(np.float32))
    kvp = np.arange(minkVp, maxkVp + 1)
    spec = spec[(kvp - 1).astype(np.int)]
    return np.stack((kvp, spec), axis = -1)



# ALLOCATE
print('>> ALLOCATE SPECTRAL MODEL...')
sm = QuantisToolkit.spectralModel(nVoxelX = pIMG.ImageX,
    nVoxelY = pIMG.ImageY,
    nVoxelZ = pIMG.ImageZ,
    nMaterial = 2,
    nEnergy = pSYS.EnergyCount,
    nPixelU = pPRJ.ProjU,
    nPixelV = pPRJ.ProjV,
    nView = len(pSYS.ProjectAngle),
    voxelSpacingX = pIMG.VoxelX,
    voxelSpacingY = pIMG.VoxelY,
    voxelSpacingZ = pIMG.VoxelZ,
    pixelSpacingU = pPRJ.PixelU,
    pixelSpacingV = pPRJ.PixelV,
    y = None,
    x0 = None,
    projectorName = 'CudaTools',
    Emin = pSYS.EnergyMin,
    energySpacing = pSYS.EnergyBin)



# SET GEOMETRY
print('>> BUILDING GEOMETRY...')
cetas = pSYS.ProjectAngle * 2.0 * np.pi / 360.0 
sm.set_projectionMatrices_circular(cetas, pSYS.SAD, pSYS.SDD, pSYS.OffsetDetectorU, pSYS.OffsetDetectorV, pSYS.OffsetTubeU, pSYS.OffsetTubeV)
sm.make_A()
del cetas



# SET MASS ATTENUATION
print('>> BUILDING MASS ATTENUATIONS...')
atten1 = get_atten(pSYS.Atten1).astype(np.float32)
atten2 = get_atten(pSYS.Atten2).astype(np.float32)
atten_full = np.stack((atten1[:, 1], atten2[:, 1]), axis = -1)
sm.make_Q(massAttenuationSpectra = atten_full)
del atten1
del atten2



# SET SPECTRA RESPONSE
print('>> BUILDING SPECTRAL RESPONSE...')
specHE = get_spectra(pSYS.SpectrumHE).astype(np.float32)
specLE = get_spectra(pSYS.SpectrumLE).astype(np.float32)
CsI = get_atten(pSYS.AttenScint).astype(np.float32)

kvps = specHE[:, 0]
specHE = specHE[:, 1]
specLE = specLE[:, 1]
specHE *= kvps * pSYS.EnergyBin * (1 - np.exp(-pSYS.ThickScint * CsI[:, 1]))
specLE *= kvps * pSYS.EnergyBin * (1 - np.exp(-pSYS.ThickScint * CsI[:, 1]))
specHE /= np.sum(specHE[:])
specLE /= np.sum(specLE[:])

spec_full = np.zeros((1, pSYS.EnergyCount, len(pSYS.ProjectAngle), pPRJ.ProjV, pPRJ.ProjU), dtype = np.float32)
for slices in range(0, pPRJ.ProjV):
    spec_full[..., ::2, slices, :] = specLE.reshape(1, pSYS.EnergyCount, 1, 1)
    spec_full[..., 1::2, slices, :] = specHE.reshape(1, pSYS.EnergyCount, 1, 1)
    
sm.make_S_StraightForward(spec_full)
del spec_full
del specHE
del specLE
del CsI



# SET GAIN
print('>> BUILDING GAIN...')
sm.make_G(uniformGain = pSYS.Flux)
sm.make_B(B = None)



# SET REGULARIZATION
print('>> SETTING REGULARIZATION...')
sm.OSPCIP_make_regularization_pnorm([pREC.HuberBeta1, pREC.HuberBeta2], pnorm = 2, delta = pREC.HuberDelta)



# SET PROJECTIONS, NOISE MODEL AND INITIALIZATION
print('>> INITIALIZING...')
projections = io.loadmat(pFNS.ProjectionID)
y = projections['y'].astype(np.float32)
y = np.maximum(y, 1e-1)

initialization = io.loadmat(pFNS.InitialID)
x0 = initialization['xi'].astype(np.float32)
x0 = np.maximum(x0, 0.0)

sm.xi = x0.copy()
sm.set_y(y)
sm.make_Sigmay_inv(Sigmay_inv = aiairecon.matrices.diagonal.ArrayScale(1.0 / y))
del projections
del initialization
del y



# START OPTIMIZATION
solver = Nesterovify05(x0.copy(),
    generator = QuantisToolkit.optimization.optimizer_OSSPS_generator_v0(sm,
        alpha = 1.0,
        precompute = False,
        subsetschedule = repeat(pREC.NumSubset)))

del x0

for i, xi in progress.iterator(enumerate(solver), num_iterations = pREC.NumIteration * pREC.NumSubset):
    if i % (20*pREC.NumSubset) == 0:
        io.savemat(pFNS.ReconstructionID[ :-4] + '_iter' + str(np.int(i/(pREC.NumSubset))) + '.mat', {'xi': xi})
    if i == pREC.NumIteration * pREC.NumSubset: break   
    if i % pREC.NumSubset: continue
    
io.savemat(pFNS.ReconstructionID, {'xi': xi})
print('...FINISHED!')
quit()