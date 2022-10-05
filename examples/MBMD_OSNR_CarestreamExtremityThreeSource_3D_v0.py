#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# TEST IMPORT
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
from PythonTools import io as istario
from IPython.display import clear_output, display
from aiairecon_cudatools.matrices.system import cudatools_projector
from aiairecon_cudatools.fbp import FDK
from aiairecon_cudatools.penalties import PnormPenalty

# EXAMPLE FOR DUAL-ENERGY TWO-MATERIAL DECOMPOSITION USING CARESTREAM ONSIGHT 3-SOURCE SCANNER GEOMETRY FILES
# OS-NR ALGORITHM
# S.Z. LIU, SZLIU@JHMI.EDU


slurm_path = '~result/~mbmd_osnr_carestream'
if not os.path.exists(slurm_path): os.makedirs(slurm_path)


class pFNS:
    ProjectionID        = ['~data/PROJ_Carestream_T1_200fms_1112pix_150x200dim.mat',
                           '~data/PROJ_Carestream_T2_200fms_1112pix_150x200dim.mat',
                           '~data/PROJ_Carestream_T3_200fms_1112pix_150x200dim.mat']  # PROJECTIONS FOR S1, S2, S3 (ASSUME #U X #V X #FRAME)
    GeometryID          = ['~data/g1.geo',
                           '~data/g2.geo',
                           '~data/g3.geo']  # GEOMETRY FILES FOR S1, S2, S3
    InitializationID    = '~data/START_MBMD_Carestream_T123_200fms_1000vox_150x250x150dim.mat'  # INITIALIZATION
    ReconstructionID    = slurm_path + '/MBMD_Carestream_T123_200fms_1000vox_150x250x150dim.mat'  # RECONSTRUCTION (FOR FILE SAVING)

class pSYS:
    Flux                = 2e6;  # PHOTON GAIN
    SAD                 = '';  # SOURCE AXIS DISTANCE (GARBAGE IF USING GEOMETRY FILES)
    SDD                 = '';  # SOURCE DETECTOR DISTANCE (GARBAGE IF USING GEOMETRY FILES)
    ProjectAngle        = np.zeros((600, ), dtype = np.float);  # PROJECTION ANGLE
    OffsetTubeU         = '';  # SOURCE OFFSET IN U (GARBAGE IF USING GEOMETRY FILES)
    OffsetTubeV         = '';  # SOURCE OFFSET IN V (GARBAGE IF USING GEOMETRY FILES)
    OffsetDetectorU     = '';  # DETECTOR OFFSET IN U (GARBAGE IF USING GEOMETRY FILES)
    OffsetDetectorV     = '';  # DETECTOR OFFSET IN U (GARBAGE IF USING GEOMETRY FILES)
    EnergyCount         = 91;  # NUMBER OF ENERGY BINS
    EnergyMin           = 10;  # SMALLEST ENERGY BIN
    EnergyBin           = 1.0; # WIDTH OF ENERGY BIN
    Spectrum1           = '~doc/spectrum_s1_hvl.mat';  # S1 SOURCE SPECTRUM
    Spectrum2           = '~doc/spectrum_s2_hvl.mat';  # S2 SOURCE SPECTRUM
    Spectrum3           = '~doc/spectrum_s3_hvl.mat';  # S3 SOURCE SPECTRUM
    SpectrumFPD         = '~doc/spectrum_detector_650csi.mat';  # DETECTOR SPECTRUM
    Atten1              = '~doc/linearatten_polyethylene.mat';  # LINEAR ATTENUATION OF BASIS 1
    Atten2              = '~doc/linearatten_aluminum.mat';  # LINEAR ATTENUATION OF BASIS 2

class pIMG:
    ImageX              = 150;  # RECONSTRUCTED IMAGE SIZE (NUMBER OF VOXELS) IN X
    ImageY              = 250;  # RECONSTRUCTED IMAGE SIZE (NUMBER OF VOXELS) IN Y -- NOTE: THIS IS ROTATION AXIS
    ImageZ              = 150;  # RECONSTRUCTED IMAGE SIZE (NUMBER OF VOXELS) IN Z
    VoxelX              = 1.0;  # VOXEL SIZE (MM) IN X
    VoxelY              = 1.0;  # VOXEL SIZE (MM) IN Y
    VoxelZ              = 1.0;  # VOXEL SIZE (MM) IN Z
    
class pPRJ:
    ProjU               = 150;  # PROJECTION SIZE (NUMBER OF PIXELS) IN U
    ProjV               = 200;  # PROJECTION SIZE (NUMBER OF PIXELS) IN V
    PixelU              = 1.112;  # PIXEL SIZE (MM) IN U
    PixelV              = 1.112;  # PIXEL SIZE (MM) IN V

class pREC:
    NumIteration        = 300;  # NUMBER OF OUTER ITERATIONS
    NumSubset           = 20;  # NUMBER OF INNER ITERATIONS (ORDERED SUBSETS)
    HuberDelta          = 0.001;  # HUBER PENALTY THRESHOLD
    HuberBeta1          = 8e3;  # HUBER PENALTY STRENGTH FOR BASIS 1
    HuberBeta2          = 8e4;  # HUBER PENALTY STRENGTH FOR BASIS 2
    Steplength          = 0.90;  # PRE-DEFINED STEP LENGTH FOR OS-NR (BETWEEN 0-1)
    HessianUpdate       = 7;  # NUMBER OF ITERATIONS TO UPDATE HESSIAN
    

    
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

def get_detector(fname, minkVp = pSYS.EnergyMin, maxkVp = pSYS.EnergyMin + pSYS.EnergyCount - 1):
    file = io.loadmat(fname)
    spec = np.squeeze(file['det'].astype(np.float32))
    kvp = np.arange(minkVp, maxkVp + 1)
    spec = spec[(kvp - 1).astype(np.int)]
    return np.stack((kvp, spec), axis = -1)



# LOAD GEOMETRY
print('>> PLAYING WITH INTRINSIC AND EXTRINSICE GEOMETRY MATRICES...')
rec = CudaTools.Reconstruction()
GHAMMAS = np.zeros((len(pSYS.ProjectAngle), 3, 4), dtype = np.float32)
for ghamma in range(0, 3):
    geo = np.loadtxt(pFNS.GeometryID[ghamma], skiprows = 1)
    rec.SetGeometryBenchFull(geo[..., 0 : 3], geo[..., 3 : 6], geo[..., 6 : 9], geo[..., 9 : 12], geo[..., 12], geo[..., 13], (1076, 884), (0.278, 0.278))
    PM = rec.GetGeometry()
    
    zeeta = rec.GetGeometryDecomposed()
    zeeta_extr = zeeta[0]
    zeeta_intr = zeeta[1]
    PERM = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype = np.float32)
    
    zeeta_intr = zeeta_intr.dot(PERM)
    
    for omecron in range(0, PM.shape[0]): PM[omecron, ...] = zeeta_extr[omecron, ...].dot(zeeta_intr[omecron, ...])
    PM[abs(PM) < 1e-10] = 0
    
    GHAMMAS[ghamma::3, ...] = PM.copy()
    pSYS.ProjectAngle[ghamma::3] = np.squeeze(np.flip(geo[..., -1]))
    del PM
del rec



# ALLOCATION
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
sm.set_projectionMatrices_direct(GHAMMAS)
sm.make_A()
del GHAMMAS
    
    

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
spec1 = get_spectra(pSYS.Spectrum1).astype(np.float32)
spec2 = get_spectra(pSYS.Spectrum2).astype(np.float32)
spec3 = get_spectra(pSYS.Spectrum3).astype(np.float32)
det = get_detector(pSYS.SpectrumFPD).astype(np.float32)

spec1 = spec1[:, 1] * det[:, 1]
spec2 = spec2[:, 1] * det[:, 1]
spec3 = spec3[:, 1] * det[:, 1]
spec1 /= np.sum(spec1[:])
spec2 /= np.sum(spec2[:])
spec3 /= np.sum(spec3[:])

spec_full = np.zeros((1, pSYS.EnergyCount, len(pSYS.ProjectAngle), pPRJ.ProjV, pPRJ.ProjU), dtype = np.float32)
for omegaa in range(0, pPRJ.ProjV):
    spec_full[..., 0::3, omegaa, :] = spec1.reshape(1, pSYS.EnergyCount, 1, 1)
    spec_full[..., 1::3, omegaa, :] = spec2.reshape(1, pSYS.EnergyCount, 1, 1)
    spec_full[..., 2::3, omegaa, :] = spec3.reshape(1, pSYS.EnergyCount, 1, 1)
    
sm.make_S_StraightForward(spec_full)
del spec_full
del spec1
del spec2
del spec3
del det



# SET GAIN
print('>> BUILDING GAIN...')
sm.make_G(uniformGain = pSYS.Flux)
sm.make_B(B = None)



# SET REGULARIZATION
print('>> SETTING REGULARIZATION...')
sm.OSPCIP_make_regularization_pnorm([pREC.HuberBeta1, pREC.HuberBeta2], pnorm = 2, delta = pREC.HuberDelta)



# LOAD DATA
print('>> LOADING DATA, IF YOU HAVE ANY...')
YEETAS = np.zeros((len(pSYS.ProjectAngle), pPRJ.ProjV, pPRJ.ProjU), dtype = np.float32)
for yeeta in range(0, 3):
    y = io.loadmat(pFNS.ProjectionID[yeeta])['y'].astype(np.float32)
    y = np.transpose(y, (2, 1, 0))
    y *= pSYS.Flux
    YEETAS[yeeta::3, ...] = y.copy()
    del y

YEETAS.shape = 1, len(pSYS.ProjectAngle), pPRJ.ProjV, pPRJ.ProjU

    
    
# INITIALIZATION
print('>> ... [^_^] ...')
if not bool(pFNS.InitializationID):
    print('>> INITIALIZATION NOT PROVIDED, SO RUNNING FELDKAMP RECON ...')
    x0 = FDK(sm.rec, sm.projAffine.to_Projections(YEETAS), sm.volAffine, I0 = pSYS.Flux, hamming = 0.5, cutoff = 0.5, parker = False)
    x0 = np.maximum(xi, 0.0)
else: x0 = io.loadmat(pFNS.InitializationID)['xi'].astype(np.float32)
x0 = np.maximum(x0, 1e-5)

sm.xi = x0.copy()
sm.set_y(YEETAS)
sm.make_Sigmay_inv(Sigmay_inv = aiairecon.matrices.diagonal.ArrayScale(1.0 / YEETAS))
sm.OSPCIP_make_Aones(hessianSimple = True) 
sm.count = 0
del YEETAS



# LINE [sm.OSPCIP_make_Aones] CAN BE REMOVED, IF YOU HAVE SMALL MEMORY
# ACCORDINGLY, [optimizer_OSNR_generator_v0(..., precompute = False, ...)] NEED TO BE SET
# START OPTIMIZATION
solver = Nesterovify05(x0.copy(),
    generator = QuantisToolkit.optimization.optimizer_OSNR_generator_v0(sm,
        alpha = pREC.Steplength,
        hessianUpdateSchedule = pREC.HessianUpdate, 
        hessianSimple = True,
        perturb = True, 
        precompute = True,
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
    

