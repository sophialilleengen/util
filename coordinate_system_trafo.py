import numpy as np
from astropy import coordinates as coords

import matplotlib.pyplot as plt

k = 4.74047

AG = np.array([[-0.0548755604, 0.4941094279, -0.8676661490], \
               [-0.8734370902, -0.4448296300, -0.1980763734],\
               [-0.4838350155, 0.7469822445, 0.4559837762]])

R_phirad = np.array([[-0.4776303088, -0.1738432154, 0.8611897727],\
                     [0.510844589, -0.8524449229, 0.111245042],\
                     [0.7147776536, 0.4930681392, 0.4959603976]])

def M_UVW_pm(phi1, phi2):
    if not isinstance(phi1, np.ndarray):
        #print('create matrix for 1 point')
        M = np.array([[np.cos(phi1) * np.cos(phi2), - np.sin(phi1), - np.cos(phi1) * np.sin(phi2)],\
                      [np.sin(phi1) * np.cos(phi2), np.cos(phi1), - np.sin(phi1) * np.sin(phi2)],\
                      [np.sin(phi2), 0., np.cos(phi2)]])
    else:
        reshaping = (len(phi1), 3, 3)
        M = np.array([])
        for i in range(len(phi1)):
            M = np.append(M, np.array([[np.cos(phi1[i]) * np.cos(phi2[i]), - np.sin(phi1[i]), - np.cos(phi1[i]) * np.sin(phi2[i])],\
                          [np.sin(phi1[i]) * np.cos(phi2[i]), np.cos(phi1[i]), - np.sin(phi1[i]) * np.sin(phi2[i])],\
                          [np.sin(phi2[i]), 0., np.cos(phi2[i])]]))

        M = M.reshape(reshaping)
    return M

# general rotation matrix
def Mrot(alpha_pole, delta_pole, phi1_0):
    '''
    Computes the rotation matrix to coordinates aligned with a pole 
    where alpha_pole, delta_pole are the poles in the original coorindates
    and phi1_0 is the zero point of the azimuthal angle, phi_1, in the new coordinates
    Critical: All angles must be in degrees!
    '''
    
    alpha_pole *= np.pi / 180.
    delta_pole = (90. - delta_pole) * np.pi / 180.
    phi1_0 *= np.pi / 180.
    
    M1 = np.array([[np.cos(alpha_pole), np.sin(alpha_pole), 0.],
                   [-np.sin(alpha_pole), np.cos(alpha_pole), 0.],
                   [0., 0., 1.]])
    
    M2 = np.array([[np.cos(delta_pole), 0., -np.sin(delta_pole)],
                   [0., 1., 0.],
                   [np.sin(delta_pole), 0., np.cos(delta_pole)]])

    
    M3 = np.array([[np.cos(phi1_0), np.sin(phi1_0), 0.],
                   [-np.sin(phi1_0), np.cos(phi1_0), 0.],
                   [0., 0., 1.]])

    return np.dot(M3, np.dot(M2, M1))

### galactocentric to galactic
def galcenrect_to_gal(xg, yg, zg, vxg, vyg, vzg, x_sun = -8.122, z_sun = 0., v_sun=[11.1, 242.947, 7.3]):
    d_GC = np.linalg.norm([x_sun, z_sun])
    costheta, sintheta = x_sun / d_GC, z_sun / d_GC
    x_out = np.dot(np.array([[-costheta, 0., -sintheta], [0., 1., 0.], [-np.sign(x_sun)*sintheta, 0., np.sign(x_sun) * costheta]]), \
                   np.array([xg, yg, zg])).T + np.array([d_GC, 0., 0.])
    v_out = np.dot(np.array([[-costheta, 0., -sintheta], [0., 1., 0.], [-np.sign(x_sun)*sintheta, 0., np.sign(x_sun) * costheta]]), \
                   np.array([vxg - v_sun[0], vyg - v_sun[1], vzg - v_sun[2]])).T    
    return x_out, v_out # X, Y, Z, U, V, W

def gal_to_galcenrect(X, Y, Z, U, V, W, x_sun = -8.122, z_sun = 0., v_sun=[11.1, 242.947, 7.3]):
    d_GC = np.linalg.norm([x_sun, z_sun])
    costheta, sintheta = x_sun / d_GC, z_sun / d_GC
    xg_out = np.dot(np.array([[costheta, 0., -sintheta], [0.,1.,0.], [sintheta, 0., costheta]]), np.array([- X + d_GC, Y , np.sign(x_sun)*Z])).T    
    vg_out = np.dot(np.array([[costheta, 0., -sintheta], [0.,1.,0.], [sintheta, 0., costheta]]), np.array([- U, V, np.sign(x_sun) * W])).T + np.array(v_sun)
    return xg_out, vg_out

def xyz_to_lbd(x, y, z):
    if not isinstance(x, np.ndarray):
        #print('x, y, z are single coordinates')
        d = np.linalg.norm([x, y, z])
    else:
        d = np.linalg.norm([x, y, z], axis = -2)
    b = np.arcsin(z / d)   
    cosl = x / d / np.cos(b)
    sinl = y / d / np.cos(b)
    l = np.arctan2(sinl, cosl)

    l = l * 180. / np.pi
    b = b * 180. / np.pi
    return l, b, d

def lbd_to_xyz(l, b, d):
    try: 
        X, Y, Z = np.array([d * np.cos(b * np.pi / 180.) * np.cos(l * np.pi / 180.), \
                        d * np.cos(b * np.pi / 180.) * np.sin(l * np.pi / 180.), \
                        d * np.sin(b * np.pi / 180.)]).T
    except:
        X, Y, Z = np.array([d * np.cos(b * np.pi / 180.) * np.cos(l * np.pi / 180.), \
                        d * np.cos(b * np.pi / 180.) * np.sin(l * np.pi / 180.), \
                        d * np.sin(b * np.pi / 180.)])
    return X, Y, Z
    
### galactic to equatorial
def lb_to_radec(l, b, d):    
    input_vec = np.array([d * np.cos(l * np.pi / 180.) * np.cos(b * np.pi / 180.), \
                          d * np.sin(l * np.pi / 180.) * np.cos(b * np.pi / 180.), \
                          d * np.sin(b * np.pi / 180.)])#T
    res = np.matmul(AG, input_vec)
    
    res0 = res[0] # = d * cos(alpha) * cos(delta)
    res1 = res[1] # = d * sin(alpha) * cos(delta)
    res2 = res[2] # = d * sin(delta)
    
    dec = np.arcsin(res2 / d)
    cosra = res0 / d / np.cos(dec)
    sinra = res1 / d / np.cos(dec)
    ra = np.arctan2(sinra, cosra)

    ra = ra * 180./ np.pi
    dec = dec * 180. / np.pi
    return ra, dec, d

def radec_to_lb(ra, dec, d):
    input_vec = np.array([d * np.cos(ra * np.pi / 180.) * np.cos(dec * np.pi / 180.), \
                          d * np.sin(ra * np.pi / 180.) * np.cos(dec * np.pi / 180.), \
                          d * np.sin(dec * np.pi / 180.)])#T
    res = np.matmul(AG.transpose(), input_vec)
    
    res0 = res[0] # = d * cos(l) * cos(b)
    res1 = res[1] # = d * sin(l) * cos(b)
    res2 = res[2] # = d * sin(b)
    
    b = np.arcsin(res2 / d)
    cosl = res0 / d / np.cos(b)
    sinl = res1 / d / np.cos(b)
    l = np.arctan2(sinl, cosl)

    l = l * 180./ np.pi
    b = b * 180. / np.pi
    return l, b, d    
    
def radec_to_streams(ra, dec, d, R_phirad = R_phirad):
    input_vec = np.array([d * np.cos(ra * np.pi / 180.) * np.cos(dec * np.pi / 180.), \
                          d * np.sin(ra * np.pi / 180.) * np.cos(dec * np.pi / 180.), \
                          d * np.sin(dec * np.pi / 180.)])
    
    res =  np.matmul(R_phirad, input_vec)
    
    res0 = res[0] # = d * cos(phi1) * cos(phi2)
    res1 = res[1] # = d * sin(phi1) * cos(phi2)
    res2 = res[2] # = d * sin(phi2)
    
    phi2 = np.arcsin(res2 / d)
    cosphi1 = res0 / d / np.cos(phi2)
    sinphi1 = res1 / d / np.cos(phi2)
    phi1 = np.arctan2(sinphi1, cosphi1)

    phi1 = phi1 * 180. / np.pi
    phi2 = phi2 * 180. / np.pi
    return phi1, phi2, d

def streams_to_radec(phi1, phi2, d, R_phirad = R_phirad):
    input_vec = np.array([d * np.cos(phi1 * np.pi / 180.) * np.cos(phi2 * np.pi / 180.), \
                          d * np.sin(phi1 * np.pi / 180.) * np.cos(phi2 * np.pi / 180.), \
                          d * np.sin(phi2 * np.pi / 180.)])
    
    res =  np.matmul(R_phirad.transpose(), input_vec)
    
    res0 = res[0] # = d * cos(ra) * cos(dec)
    res1 = res[1] # = d * sin(ra) * cos(dec)
    res2 = res[2] # = d * sin(dec)
    
    dec = np.arcsin(res2 / d)
    cosra = res0 / d / np.cos(dec)
    sinra = res1 / d / np.cos(dec)
    ra = np.arctan2(sinra, cosra)

    ra = ra * 180. / np.pi
    dec = dec * 180. / np.pi
    return ra, dec, d

def UVW_to_pm(U, V, W, d, phi1, phi2, R_phirad = R_phirad):
    input_vec = np.array([U, V, W]).T
    phi1, phi2 = phi1 * np.pi / 180., phi2 * np.pi / 180.
    M_mat = M_UVW_pm(phi1, phi2)    
    
    if not isinstance(phi1, np.ndarray):
        print('Single values for U, V, W.')
        res = np.matmul(np.matmul(np.matmul(M_mat.transpose(), R_phirad), AG), input_vec)
    else:
        res = np.zeros((len(M_mat), 3))

        for ii in range(len(M_mat)):
            res[ii] = np.matmul(np.matmul(np.matmul(M_mat[ii].transpose(), R_phirad), AG), input_vec[ii])
            
        res = res.T
    
    vr = res[0]
    mphi1_s = res[1] / k / d# / np.cos(phi2)
    mphi1 = res[1] / k / d / np.cos(phi2)
    mphi2 = res[2] / k / d
    return vr, mphi1_s, mphi2

def pm_to_UVW(phi1, phi2, d, vr, mphi1_s, mphi2, R_phirad = R_phirad):
    input_vec = np.array([vr, k * d * mphi1_s, k * d * mphi2]).T
    phi1, phi2 = phi1 * np.pi / 180., phi2 * np.pi / 180.
    M_mat = M_UVW_pm(phi1, phi2)    
    
    if not isinstance(phi1, np.ndarray):
        #print('Single values for vr, mphi1_s, mphi2.')
        res =  np.matmul(np.matmul(np.matmul(AG.transpose(), R_phirad.transpose()), M_mat), input_vec)
    else:
        res = np.zeros((len(M_mat), 3))

        for ii in range(len(M_mat)):
            res[ii] = np.matmul(np.matmul(np.matmul(AG.transpose(), R_phirad.transpose()), M_mat[ii]), input_vec[ii])
            
        res = res.T    
    U, V, W = res
    return U, V, W

def pmrapmdec_to_UVW(ra, dec, d, pmra, pmdec, vr):
    input_vec = np.array([vr, k * d * pmra, k * d * pmdec]).T
    ra, dec = ra * np.pi / 180., dec * np.pi / 180.
    M_mat = M_UVW_pm(ra, dec)    
    
    if not isinstance(ra, np.ndarray):
        #print('Single values for vr, mphi1_s, mphi2.')
        res =  np.matmul(np.matmul(AG.transpose(), M_mat), input_vec)
    else:
        res = np.zeros((len(M_mat), 3))

        for ii in range(len(M_mat)):
            res[ii] = np.matmul(np.matmul(AG.transpose(), M_mat[ii]), input_vec[ii])
            
        res = res.T    
    U, V, W = res
    return U, V, W

def UVW_to_pmrapmdec(ra, dec, d, U, V, W):
    # radec and phi1phi2 not sure, might be wrong here
    input_vec = np.array([U, V, W]).T
    ra, dec = ra * np.pi / 180., dec * np.pi / 180.
    M_mat = M_UVW_pm(ra, dec)    
    
    if not isinstance(ra, np.ndarray):
        #print('Single values for vr, mphi1_s, mphi2.')
        res =  np.matmul(np.matmul(M_mat.transpose(), AG), input_vec)
    else:
        res = np.zeros((len(M_mat), 3))

        for ii in range(len(M_mat)):
            res[ii] = np.matmul(np.matmul(M_mat[ii].transpose(), AG), input_vec[ii])
            
        res = res.T    
    vr = res[0]
    mra_s = res[1] / k / d# / np.cos(phi2)
    mra   = res[1] / k / d / np.cos(dec)
    mdec  = res[2] / k / d
    return vr, mra_s, mdec

def galcenrect_to_streams(xg, yg, zg, vxg, vyg, vzg, x_sun = -8.122, z_sun = 0., v_sun = [11.1, 242.947, 7.3], R_phirad = R_phirad, pms = 'pmstreams'):
    # mphi1, mphi2 are placeholders and can be used as mra, mdec as well
    x_out, v_out = galcenrect_to_gal(xg, yg, zg, vxg, vyg, vzg, x_sun = x_sun, z_sun = z_sun, v_sun = v_sun)
    if not isinstance(xg, np.ndarray):
        x, y, z = x_out
        vx, vy, vz = v_out
    else:
        x, y, z = x_out[:,0], x_out[:,1], x_out[:,2]
        vx, vy, vz = v_out[:,0], v_out[:,1], v_out[:,2]
    l, b, d = xyz_to_lbd(x, y, z)
    ra, dec, d = lb_to_radec(l, b, d)
    phi1, phi2, d = radec_to_streams(ra, dec, d, R_phirad)
    U, V, W = vx, vy, vz # what about solar reflex motion here?
    if pms == 'pmstreams':
        vr, mphi1, mphi2 = UVW_to_pm(U, V, W, d, phi1, phi2, R_phirad)
    elif pms == 'pmradec':
        vr, mphi1, mphi2 = UVW_to_pmrapmdec(ra, dec, d, U, V, W)
    else: raise NameError('Please define pms as "pmstreams" or "pmradec".')    
    return phi1, phi2, d, vr, mphi1, mphi2

def galcenrect_to_streams_pos(xg, yg, zg, x_sun = -8.122, z_sun = 0., v_sun = [11.1, 242.947, 7.3], R_phirad = R_phirad):
    vxg, vyg, vzg = 0., 0., 0.
    x_out, v_out = galcenrect_to_gal(xg, yg, zg, vxg, vyg, vzg, x_sun = x_sun, z_sun = z_sun, v_sun = v_sun)
    if not isinstance(xg, np.ndarray):
        x, y, z = x_out
    else:
        x, y, z = x_out[:,0], x_out[:,1], x_out[:,2]
    l, b, d = xyz_to_lbd(x, y, z)
    ra, dec, d = lb_to_radec(l, b, d)
    phi1, phi2, d = radec_to_streams(ra, dec, d, R_phirad)   
    return phi1, phi2, d
    
def streams_to_galcenrect_pos(phi1, phi2, d, x_sun = -8.122, z_sun = 0., v_sun = [11.1, 242.947, 7.3], R_phirad = R_phirad):
    # mphi1, mphi2 are placeholders and can be used as mra, mdec as well
    ra, dec, d = streams_to_radec(phi1, phi2, d, R_phirad)    
    l, b, d = radec_to_lb(ra, dec, d)
    #print(l, b, d)
    try:
        X, Y, Z = lbd_to_xyz(l, b, d)
    except ValueError:
        #print(np.transpose((l, b, d)))
        arr = np.transpose((l, b, d))
        l = arr[:,0]
        b = arr[:,1]
        d = arr[:,2]
        X, Y, Z = lbd_to_xyz(l, b, d)
    return X, Y, Z

def streams_to_galcenrcirc_pos(phi1, phi2, d,x_sun = -8.122, z_sun = 0., v_sun = [11.1, 242.947, 7.3], R_phirad = R_phirad):
    # mphi1, mphi2 are placeholders and can be used as mra, mdec as well
    ra, dec, d = streams_to_radec(phi1, phi2, d, R_phirad)    
    l, b, d = radec_to_lb(ra, dec, d)
    return l, b, d

def streams_to_galcenrect(phi1, phi2, d, vr, mphi1_s, mphi2, x_sun = -8.122, z_sun = 0., v_sun = [11.1, 242.947, 7.3], R_phirad = R_phirad, pms = 'pmstreams'):
    # mphi1, mphi2 are placeholders and can be used as mra, mdec as well
    ra, dec, d = streams_to_radec(phi1, phi2, d, R_phirad)
    
    if pms == 'pmstreams':
        U, V, W = pm_to_UVW(phi1, phi2, d, vr, mphi1_s, mphi2, R_phirad)
    elif pms == 'pmradec':
        U, V, W = pmrapmdec_to_UVW(ra, dec, d, mphi1_s, mphi2, vr)
    else: raise NameError('Please define pms as "pmstreams" or "pmradec".')
    
    l, b, d = radec_to_lb(ra, dec, d)
    X, Y, Z = lbd_to_xyz(l, b, d)
    xg_out, vg_out = gal_to_galcenrect(X, Y, Z, U, V, W, x_sun = x_sun, z_sun = z_sun, v_sun = v_sun)
    if not isinstance(phi1, np.ndarray):
        x, y, z = xg_out
        vx, vy, vz = vg_out
    else:
        x, y, z = xg_out[:,0], xg_out[:,1], xg_out[:,2]
        vx, vy, vz = vg_out[:,0], vg_out[:,1], vg_out[:,2]    
    return x, y, z, vx, vy, vz

def radecpmrapmdec_to_streams(ra, dec, d, pmra, pmdec, vr, R_phirad = R_phirad):
    phi1, phi2, d = radec_to_streams(ra, dec, d)
    U, V, W = pmrapmdec_to_UVW(ra, dec, d, pmra, pmdec, vr)
    vr, mphi1_s, mphi2 = UVW_to_pm(U, V, W, d, phi1, phi2, R_phirad)
    return phi1, phi2, d, vr, mphi1_s, mphi2

def radecpms_to_xyzv(ra, dec, d, pmra, pmdec, vr, x_sun = -8.122, z_sun = 0., v_sun = [11.1, 242.947, 7.3]):
    U, V, W = pmrapmdec_to_UVW(ra, dec, d, pmra, pmdec, vr)
    l, b, d = radec_to_lb(ra, dec, d)
    X, Y, Z = lbd_to_xyz(l, b, d)
    xg_out, vg_out = gal_to_galcenrect(X, Y, Z, U, V, W, x_sun = x_sun, z_sun = z_sun, v_sun = v_sun)
    if not isinstance(ra, np.ndarray):
        x, y, z = xg_out
        vx, vy, vz = vg_out
    else:
        x, y, z = xg_out[:,0], xg_out[:,1], xg_out[:,2]
        vx, vy, vz = vg_out[:,0], vg_out[:,1], vg_out[:,2]    
    return x, y, z, vx, vy, vz    