# Quantis Toolkits
Python implementations of constrained and unconstrained optimization algorithms for DE CBCT:
- Liu, S. Z., Tivnan, M., Osgood, G. M., Siewerdsen, J. H., Stayman, J. W., & Zbijewski, W. (2022) "Model-based three-material decomposition in dual-energy CT using the volume conservation constraint," *Phys. Med. Biol.*, **67**(14), 145006. DOI: https://doi.org/10.1088/1361-6560/ac7a8b
- Liu, S. Z., Cao, Q., Tivnan, M., Tilley II, S., Siewerdsen, J. H., Stayman, J. W., & Zbijewski, W. (2020). "Model-based dual-energy tomographic image reconstruction of objects containing known metal components," *Phys. Med. & Biol.*, **65**(24), 245046. DOI: https://doi.org/10.1088/1361-6560/abc5a9

## Contact
- Wojciech Zbijewski: wzbijewski@jhu.edu

## Intro
We develop a model-based optimization algorithm for‘one-step’ dual-energy (DE) CT decomposition of three materials directly from projection measurements. Since the three-material problem is inherently undetermined, we incorporate the volume conservation principle (VCP) as a pair of equality and nonnegativity constraints into the objective function of the recently reported model-based material decomposition (MBMD). An optimization algorithm (constrained MBMD, CMBMD)is derived that utilizes voxel-wise separability to partition the volume into a VCP-constrained region solved using interior-point iterations, and an unconstrained region (air surrounding the object, where VCP is violated) solved with conventional two-material MBMD. CMBMD is validated in simulations and experiments in application to bone composition measurements in the presence of metal hardware using DE cone-beam CT (CBCT). A kV-switching protocol with non-coinciding low- and high-energy (LE and HE) projections was assumed. CMBMD with decomposed base materials of cortical bone, fat, and metal (titanium, Ti) is compared to MBMD with (i)fat-bone and (ii) fat-Ti bases. Main results. Three-material CMBMD exhibits a substantial reduction in metal artifacts relative to the two-material MBMD implementations. The accuracies of cortical bone volume fraction estimates are markedly improved using CMBMD, with ~5-10x lower normalized root mean squared error in simulations with anthropomorphic knee phantoms (depending on the complexity of the metal component) and ~2-2.5x lower in an experimental test-bench study. In conclusion, we demonstrated one-step three-material decomposition of DE CT using volume conservation as an optimization constraint. The proposed method might be applicable to DE applications such as bone marrow edema imaging (fat-bone-water decomposition) or multi-contrast imaging, especially on CT/CBCT systems that do not provide coinciding LE and HE ray paths required for conventional projection-domain DE decomposition.

## Example
Below is an quick example demonstrating the unconstrained (MBMD) and constrained (CMBMD) algorithms for quantitative kV-switching CBCT imaging of tibia with surgical intramedullary nail. The three material bases were fat, cortical bone and titanium in CMBMD.
![Picture1](https://user-images.githubusercontent.com/108881232/178122539-884d9f7a-5f97-4ebc-ba5d-46b1e53ca4b8.png)

## Installation
### Major dependencies
  ```diff
  # Notice the specific versions
  conda create -n aloha python=3.6.10 anaconda
  conda activate aloha
  conda install gitpython=3.1.3
  conda install -c conda-forge transforms3d
  conda install -c http://aiai-tartarus.jhmi.edu/conda-bld-dev istar-cudatools
  conda install -c http://aiai-tartarus.jhmi.edu/conda-bld aiairecon_cuda=0.1.147
  # if the above line is not working (esp. for Win users), check next section or talk to Stephen
  git clone http://git.lcsr.jhu.edu/istar/pythontools.git
  conda develop pythontools
  git clone http://git.lcsr.jhu.edu/istar/aiairecon.git
  conda develop aiairecon
  git clone http://git.lcsr.jhu.edu/istar/aiairecon_cudatools.git
  conda develop aiairecon_cudatools
  ```

### aiairecon_cuda dependency
  ```diff
  # Install aiairecon_cuda via conda may not work, particularly for windows users.
  # If so, grab aiairecon_cuda package from http://git.lcsr.jhu.edu/istar/aiairecon_cuda
  # Compile (CMake & MS Visual Studio 2017)
  # Open ./aiairecon_cuda/interface.py in site-packages and add the following at the beginning:
    import os
    os.environ['PATH'] = 'Fill in your aiairecon_cuda build directory' + ';' + os.environ['PATH']
  # Run the test scripts and check if it works properly:
    python -m aiairecon_cuda.run_lib_tests
    python -m aiairecon_cuda.run_py_tests
  ```

### Install QuantisToolkit
  ```
  git clone http://git.lcsr.jhu.edu/istar/QuantisToolkit.git
  conda develop cmbmd
  ```
