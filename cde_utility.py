import numpy as np
import scipy as sci
import scipy.special as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import time

# mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['figure.figsize'] = 8, 6
# mpl.rcParams['axes.titlesize'] = 20
# mpl.rcParams['axes.labelsize'] = 20
# mpl.rcParams['lines.linewidth'] = 2
# mpl.rcParams['lines.markersize'] = 3
# mpl.rcParams['xtick.labelsize'] = 16
# mpl.rcParams['ytick.labelsize'] = 16
# mpl.rcParams['legend.fontsize'] = 16

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['figure.figsize'] = 7.2, 5.4
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 18

vec_norm = np.linalg.norm
det = np.linalg.det
erfc = sp.erfc
twopi = 2.0*np.pi

class Lattice:
    def __init__(self, a1, a2, a3, lat_size=1.0, gvec_normalize=False):
        if det([a1, a2, a3]) == 0.0:
            raise ValueError("Basis vectors are linear dependent!")
        else:
            self.lat_size = lat_size
            self.a1 = np.asarray(a1)*self.lat_size
            self.a2 = np.asarray(a2)*self.lat_size
            self.a3 = np.asarray(a3)*self.lat_size
            self.lat_basis = np.array([self.a1, self.a2, self.a3])
            self.uc_vol = np.abs( det([self.a1, self.a2, self.a3]) )
            if gvec_normalize:
                twopi_o_vol = 1.0
            else:
                twopi_o_vol = twopi/self.uc_vol
            self.b1 = twopi_o_vol*np.cross(self.a2, self.a3)
            self.b2 = twopi_o_vol*np.cross(self.a3, self.a1)
            self.b3 = twopi_o_vol*np.cross(self.a1, self.a2)
            self.reci_basis = np.array([self.b1, self.b2, self.b3])
    
    def lat_vec(self, c1, c2, c3=0.0):
        return c1*self.a1 + c2*self.a2 + c3*self.a3
    
    def reci_vec(self, g1, g2, g3=0.0):
        return g1*self.b1 + g2*self.b2 + g3*self.b3

#   Define a few lattices
# square lattice for Tsang-Kong book examples.
square0=Lattice([0.95, 0, 0], [0, 0.95, 0], [0, 0, 1])

square = Lattice(a1=[1, 0, 0], a2=[0, 1, 0], a3=[0, 0, 1])

triangular = Lattice(a1=[0.5, -0.5*np.sqrt(3), 0], 
                     a2=[0.5, 0.5*np.sqrt(3), 0], 
                     a3=[0, 0, 1])

def gvec(n1, n2, lat):
    return [ lat.reci_vec(g1, g2) 
            for g1 in np.linspace(-n1, n1, 2*n1+1)
            for g2 in np.linspace(-n2, n2, 2*n2+1) ]

def rmesh(nx, ny, lat, rmin=1j, rmax=1e8):
    x = np.arange(-nx, nx+1)
    y = np.arange(-ny, ny+1)
    xs, ys = np.meshgrid(x, y)
    ax, ay = lat.a1, lat.a2
    mask = ( (xs*ax[0])**2 + (ys*ay[1])**2 > rmin**2 ) &\
    ( (xs*ax[0])**2 + (ys*ay[1])**2 < rmax**2 )
    grid = np.tensordot(xs[mask], ax, axes=0) +\
    np.tensordot(ys[mask], ay, axes=0)
    return grid
    
def gmesh(nx, ny, lat, gmin=-1.0, gmax=None):
    gx, gy = lat.b1, lat.b2
    if gmax is None:
        gmax = 1e8
    else:
        scale = 2
        nx = scale*np.int(np.floor(gmax/vec_norm(gx)))
        ny = scale*np.int(np.floor(gmax/vec_norm(gy)))
    x = np.arange(-nx, nx+1)
    y = np.arange(-ny, ny+1)
    xs, ys = np.meshgrid(x, y)
    grid = np.tensordot(xs, gx, axes=0) + np.tensordot(ys, gy, axes=0)
    mask = ( vec_norm(grid, axis=2) > gmin ) &\
    ( vec_norm(grid, axis=2) <= gmax )
    return grid[mask]

def g1term(rvec, kvec, kw, E, g, lat):
    g = np.atleast_2d(g)
    k = np.asarray(kvec)
    r = np.asarray(rvec)
    kz = sci.sqrt(kw**2 - vec_norm(k+g, axis=1)**2)
    z = r[2]
    uc_vol = lat.uc_vol
    return np.pi*1j/uc_vol*np.exp(1j*np.dot(k+g, r))/kz*\
    ( np.exp(1j*kz*z)*erfc(-1j*kz/2.0/E - E*z) + \
        np.exp(-1j*kz*z)*erfc(-1j*kz/2.0/E + E*z) )

def g1_sum(nx, ny, rvec, kvec, kw, E, lat, grid=None):
    if grid is None:
        grid = gmesh(nx, ny, lat)
    return np.sum( g1term(rvec, kvec, kw, E, grid, lat) )

def g2term(rvec, kvec, kw, E, Rvec, lat):
    k = np.asarray(kvec)
    r = np.asarray(rvec)
    R = np.atleast_2d(Rvec)
    eik = np.exp(1j*np.dot(R, k))
    rR = vec_norm(r-R, axis=1)
    ikw2E = 1j*kw/2.0/E
    return 0.5*eik/rR*( np.exp(1j*kw*rR)*erfc(E*rR + ikw2E) +\
        np.exp(-1j*kw*rR)*erfc(E*rR - ikw2E) )

def g2_sum(nx, ny, rvec, kvec, kw, E, lat, grid=None):
    if grid is None:
        grid = rmesh(nx, ny, lat)
    return np.sum( g2term(rvec, kvec, kw, E, grid, lat) )

def g3(kw, E):
    return - 1.0*1j*kw*erfc(-1j*kw/2.0/E) -\
    1.0*2*E/np.sqrt(np.pi)*np.exp(kw**2/4.0/E**2)

def S_ewald(k, kw, N_G, N_R, E, lat, c1, c2, c3):
    r0 = [0, 0, 0]
    rs = rmesh(N_R, N_R, lat, rmin=0)
    Gsum = g1_sum(N_G, N_G, r0, k, kw, E, lat)
    Rsum = g2_sum(N_R, N_R, r0, k, kw, E, lat, grid=rs)
    other = - 1.0*1j*kw*erfc(-1j*kw/2.0/E) - \
    1.0*2*E/np.sqrt(np.pi)*np.exp(kw**2/4.0/E**2)
    return c1*Gsum + c2*Rsum + c3*other

def fRn(Rvec, n):
    '''
    Returns an algebraically decaying potential f(R) = 1/R^n, R=|Rvec|
    '''
    Rvec = np.atleast_2d( Rvec )
    return 1.0/vec_norm(Rvec, axis=1)**n

fR1 = lambda Rvec: fRn(Rvec, 1)
fR2 = lambda Rvec: fRn(Rvec, 2)
fR3 = lambda Rvec: fRn(Rvec, 3)

def Rsum_direct(func, numR, rvec, kvec, kw, lat):
    '''
    numpy vectorized version for direct real-space 
    lattice sum S(k, kw)
    '''
    grid = rmesh(numR, numR, lat, rmin=0)
    terms = np.exp( 1j*np.dot(grid, kvec) ) * \
    np.exp( 1j*kw*vec_norm(rvec - grid, axis=1) ) * func(grid)
    return np.sum(terms)

def plot_grid(grid, marker='o', pointsize=4, \
    if_save=False, filename='grid.pdf'):
    plt.figure(figsize=(6,6))
#     X = [ r[0] for r in grid ]
#     Y = [ r[1] for r in grid ]
    X = grid[:, 0]
    Y = grid[:, 1]
    nx = len(X)
    ny = len(Y)
    plt.plot(X, Y, marker, markersize=pointsize)
    plt.grid('on')
    plt.axis('equal')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'${}$ Grid Points'.format(nx))
    plt.tight_layout()
    if if_save:
        plt.savefig(filename)
    plt.show()

def rel_err(x, y):
    return (x - y)/np.fabs(y)

marker_style = dict(linestyle='-', marker='o', \
    markersize=10, markerfacecolor='none', \
    markerfacecoloralt='gray', markeredgewidth=1.5, clip_on=True)

# ################################################
# Comparison of two versions of G-grid generation
# ################################################
# %time grid = gmesh(1000, 1000, square)
# %time grid2 = gvec(1000, 1000, square)
# '''
# CPU times: user 165 ms, sys: 129 ms, total: 294 ms
# Wall time: 324 ms
# CPU times: user 22.8 s, sys: 626 ms, total: 23.4 s
# Wall time: 23.8 s
# '''

# #############################
# Grid function gmesh test
# #############################
# nx, ny = 200, 200
# print(2*(nx+1)*2*(ny+1))
# lat = square
# %time grid = gmesh(nx, ny, lat, gmin=nx/10*vec_norm(lat.b1), \
#                    gmax=nx/2*vec_norm(lat.b1))
# %time plot_grid(grid)

def Ggrid(z, U, lat, scale=2):
    z = np.asarray(z)
    Gmax = U/z
    nx = np.int(np.floor(Gmax/vec_norm(lat.b1)))
    ny = np.int(np.floor(Gmax/vec_norm(lat.b2)))
    nx = scale*nx
    ny = scale*ny
    grid = gmesh(nx, ny, lat, gmin=0.0, gmax=Gmax)
    return grid

def Lsum(z, U, lat):
    Gs = Ggrid(z, U, lat)
    V_G = twopi**2/lat.uc_vol
    G = vec_norm(Gs, axis=1)
    return V_G/twopi*np.sum( np.exp(-G*z)/G )

def f1z(z, U, lat):
    z = np.asarray(z)
    L = Lsum(z, U, lat)
    return np.log( z*L + np.exp(-U) )/z

def f2z(z, U, lat):
    z = np.asarray(z)
    L = Lsum(z, U, lat)
    return L + ( np.exp(-U) - 1.0 )/z

def D_const(U, lat, nz=10):
    zz = np.linspace(0.005, 0.05, nz)
    f1 = [ f1z(z, U, lat) for z in zz ]
    f2 = [ f2z(z, U, lat) for z in zz ]
    d1 = np.polyfit(zz, f1, 1)[-1]
    d2 = np.polyfit(zz, f2, 1)[-1]
    return 0.5*(d1+d2)

def g0term(k, kw, g):
    g = np.atleast_2d(g)
    return 1.0/sci.sqrt(kw**2 - vec_norm(k + g, axis=1)**2) - \
    1.0/vec_norm(g, axis=1)/1.0j

def sumG(k, kw, Gmax, lat):
    grid = Ggrid(1.0, Gmax, lat, scale=2)
    return np.sum( g0term(k, kw, grid) )

def S_simowski(k, kw, Gmax, D, lat):
    sg = twopi*1j/lat.uc_vol*sumG(k, kw, Gmax, lat)
    kz = sci.sqrt(kw**2-np.dot(k, k))
    return D + twopi*1j/lat.uc_vol/kz - 1j*kw + sg

def kpath(k1, k2, nk, end_point=False):
    k1 = np.array(k1)
    k2 = np.array(k2)
    dk = k2 - k1
    ks = [ k1 + ik*dk  for ik in np.linspace(0, 1, nk) ]
    if not end_point:
        ks = ks[:-1]
    return ks

def fnRij(Rvec, n, i, j):
    i, j = int(i), int(j)
    assert 0<= i < 3, "i should be 0, 1, 2"
    assert 0<= j < 3, "j should be 0, 1, 2"
    Rvec = np.atleast_2d( Rvec )
    Ri = Rvec[:, i]
    Rj = Rvec[:, j]
    R = vec_norm(Rvec, axis=1)
    return Ri*Rj/np.power(R, n+2)
    
def dterm(x):
    x = np.asarray(x)
    return erfc(np.sqrt(np.pi)*x)/x