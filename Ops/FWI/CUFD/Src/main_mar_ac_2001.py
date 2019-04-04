
# coding: utf-8

# In[ ]:


# coding: utf-8
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import subprocess
import os as os
import CUFD as CUFD
import surveyfile_pre as sp
import parafile_pre as pp
os.system('rm ../Data/Shot*')
os.system('rm ../Data/Syn_Shot*')
os.system('rm ../Data/Residual_Shot*')


# In[ ]:


# load model
Model_path = "../Models/"
MAT = sio.loadmat(Model_path + 'Mar_models.mat')
Cp = MAT['Vtrue'].astype('float32')
(nz, nx) = Cp.shape

Cs = 0.0*np.ones((nz, nx)).astype('float32')

Den = 1000.0*np.ones((nz, nx)).astype('float32')
dz = 24.0
dx = 24.0
nPml = 32
dt = 0.0025
f0 = 4.5
nSteps = 2001
isAc = True
withAdj = False
if_res = False
para_fname = '../Par/Par_file.json'
pp.parafile_pre(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, nPml, isAc, withAdj, if_res, para_fname)


# In[ ]:


plt.imshow(Cp)
plt.colorbar()
plt.show()


# In[ ]:


# for Mar test
x_src = (np.arange(3, 384, 8)).astype(int)
# x_src = (np.arange(3,4)).astype(int)
z_src = (np.ones(x_src.shape[0])).astype(int)
x_rec = (np.arange(2,381)).astype(int)
z_rec = (np.ones(x_rec.shape[0])).astype(int)

survey_fname = '../Par/survey_fname.json'
Src_fname = "../Models/Mar_source_2001.bin"
sp.surveyfile_pre(z_src, x_src, z_rec, x_rec, survey_fname, Src_fname)
print(x_src)
print(x_rec)


# In[ ]:


cmdlaunch='CUDA_VISIBLE_DEVICES=0 ./CUFD ' + para_fname + ' ' + survey_fname + ' > out1'
subprocess.call(cmdlaunch,stdout=None, stderr=None,shell=True)


# In[ ]:


Shot = np.fromfile("../Data/Shot0.bin", dtype='float32', count=-1)
Shot = np.reshape(Shot, (nSteps, -1), order='F')
plt.imshow(Shot, aspect='auto', cmap='gray')
# plt.imshow(Shot, aspect='auto', cmap='gray', clim=(-1e-5, 1e-5))
plt.colorbar()
plt.show()


# In[ ]:


# plt.imshow(Shot[0+550:128+550, 0:128], aspect='auto', cmap='gray', clim=(-10, 10))
# plt.colorbar()
# plt.show()


# In[ ]:


# load model
Model_path = "../Models/"
MAT = sio.loadmat(Model_path + 'Mar_models.mat')
Cp = MAT['Vsm'].astype('float32')
(nz, nx) = Cp.shape


Cs = 0.0*np.ones((nz, nx)).astype('float32')

Den = 1000.0*np.ones((nz, nx)).astype('float32')
dz = 24.0
dx = 24.0
nPml = 32
dt = 0.0025
f0 = 4.5
nSteps = 2001
isAc = True
withAdj = True
if_res = True
para_fname = '../Par/Par_file.json'
pp.parafile_pre(Cp, Cs, Den, nz, nx, dz, dx, nSteps, dt, f0, nPml, isAc, withAdj, if_res, para_fname)


# In[ ]:


plt.imshow(Cp)
plt.colorbar()
plt.show()


# In[ ]:


cmdlaunch='CUDA_VISIBLE_DEVICES=0 ./CUFD ' + para_fname + ' ' + survey_fname + ' > out2'
subprocess.call(cmdlaunch,stdout=None, stderr=None,shell=True)


# In[ ]:


Shot = np.fromfile("../Data/Syn_Shot0.bin", dtype='float32', count=-1)
Shot = np.reshape(Shot, (nSteps, -1), order='F')
plt.imshow(Shot, aspect='auto', cmap='gray')
# plt.imshow(Shot, aspect='auto', cmap='gray', clim=(-1e-5, 1e-5))
plt.colorbar()
plt.show()

Shot = np.fromfile("../Data/Residual_Shot0.bin", dtype='float32', count=-1)
Shot = np.reshape(Shot, (nSteps, -1), order='F')
plt.imshow(Shot, aspect='auto', cmap='gray')
# plt.imshow(Shot, aspect='auto', cmap='gray', clim=(-1e-5, 1e-5))
plt.colorbar()
plt.show()


# In[ ]:


Image = np.fromfile("./CpGradient.bin", dtype='float32', count=-1)
Image = np.reshape(Image, (224, -1), order='F')
plt.imshow(Image[33:31+134,:], aspect='auto', cmap='gray')
# plt.imshow(Shot, aspect='auto', cmap='gray', clim=(-1e-5, 1e-5))
plt.colorbar()
plt.show()

