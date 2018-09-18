#!/usr/bin/env python
#--------------------------------------------------
# Coupled-dipole theory 
# Band structure 
# 2018-09-14 00:15:59						  711er
#--------------------------------------------------

# import cde_utility
from cde_utility import *

lat = square
# Define high-symmetry k-points
Gamma = lat.reci_vec(0, 0, 0)
X = lat.reci_vec(0.5, 0, 0)
M = lat.reci_vec(0.5, 0.5, 0)

# Generate a Gamma-X-M-Gamma k-vector list
nk = 51
kmesh = np.concatenate([kpath(Gamma, X, nk), kpath(X, M, nk), \
	kpath(M, Gamma, nk, end_point=True)])
k_index = np.arange(len(kmesh))
# plot_grid(kmesh)

# parameters
a_o_d = 1.0/3.0
kw = 0.0
r0 = [0, 0, 0]
N_R_z = 200
N_R_xy = 200

# Transverse mode(p_z dipole mode)
beta_z = -np.real( [ Rsum_direct(fR3, N_R_z, r0, kvec, kw, lat) \
	for kvec in kmesh ] )
w_T = np.sqrt( 1.0 - a_o_d**3*beta_z )

# In-plane mode
fR3yx = lambda Rvec: fnRij(Rvec, 3, 1, 0)
fR3xx = lambda Rvec: fnRij(Rvec, 3, 0, 0)
fR3yy = lambda Rvec: fnRij(Rvec, 3, 1, 1)

# timeit.timeit(
beta_11 = np.array( [ Rsum_direct(fR3xx, N_R_xy, r0, kvec, kw, lat) \
	for kvec in kmesh ] )
# print('Time for calculating beta_11: {:.4f} (s)'.format)
beta_11 = 3.0*beta_11 + beta_z

beta_22 = np.array( [ Rsum_direct(fR3yy, N_R_xy, r0, kvec, kw, lat) \
	for kvec in kmesh ] )
beta_22 = 3.0*beta_22 + beta_z

beta_21 = np.array( [ Rsum_direct(fR3yx, N_R_xy, r0, kvec, kw, lat) \
	for kvec in kmesh ] )

w_TI = 1.0 - a_o_d**3/2.0*\
( beta_11 + beta_22 + np.sqrt( (beta_11 - beta_22)**2 + 4.0*beta_21**2 ) )
w_LI = 1.0 - a_o_d**3/2.0*\
( beta_11 + beta_22 - np.sqrt( (beta_11 - beta_22)**2 + 4.0*beta_21**2 ) )
w_TI = np.sqrt(np.real(w_TI))
w_LI = np.sqrt(np.real(w_LI))

np.savetxt('band-sqaure-qsa-Nz-{}-Nxy-{}.txt'.format(N_R_z, N_R_xy),
           np.c_[w_T, w_TI, w_LI],
           fmt='%.6f', header='T \t TI \t LI',
           comments='nk={:d}\n'.format(nk))

# Plot band structure

marker_style = dict(linestyle='-', lw=3, marker='',markersize=5, 
					#markerfacecolor='none', \
                    markerfacecoloralt='gray', markeredgewidth=1.5, \
                    clip_on=False)

hi_sym_k = [0, nk-1, 2*nk-2, -1]
hi_sym_k = k_index[hi_sym_k]
k_labels = ['$\Gamma$', '$X$', '$M$', '$\Gamma$']

plt.figure(figsize=(8,6))

plt.plot( k_index, w_T, 'o', color='orange', **marker_style, label='T mode' )
plt.plot( k_index, np.real(w_TI), 'ro', **marker_style, label='TI mode' )
plt.plot( k_index, np.real(w_LI), 'bo', **marker_style, label='LI mode' )
plt.xlim((k_index[0], k_index[-1]))
axes = plt.gca()
ymin, ymax = axes.get_ylim()
plt.ylim((ymin, ymax))
plt.xticks( hi_sym_k, k_labels, fontsize=22 )
plt.vlines(hi_sym_k, ymin, ymax, lw=1, color='k')
plt.ylabel(r'$\omega/\omega_0$')
plt.title('Square Lattice Quasi-static Band Structure')
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('band-qs-square.pdf')
# plt.show()


# ### Triangular Lattice

# ## Dynamical Response

# In[ ]:


# def eps_drude(kw, wp, gamma):
#     w_pl = 11.34
#     eps_a, eps_b, gamma = 5.45, 6.18, 0.05