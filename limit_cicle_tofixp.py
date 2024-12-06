import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/4.4k13_1.5w5.5_3d/'
name = '_4.4k13_1.5w5.5'
namenp = name + '.npy'

solnumb_file = root_path+ f'solnumb' + namenp
xsol_file = root_path+f'xsol' +namenp
ysol_file = root_path+f'ysol' +namenp

# params = np.load(root_path+'params71.npy')
# print('worked')
# exit()
k = np.load(root_path+ 'kparam' + namenp)
w = np.load(root_path+ 'wparam' + namenp)
d = np.load(root_path+ 'dparam' + namenp)

solution_number = np.load(solnumb_file)
xsol = np.load(xsol_file)
ysol = np.load(ysol_file)
gridpoints = solution_number.shape[0]

stability_plus = np.load(root_path+'stability_plus.npy')
stability = np.load(root_path+'stability.npy')
stab_zsol = np.ones_like(solution_number)*np.nan
stab_xsol = np.ones_like(solution_number)*np.nan
stab_ysol = np.ones_like(solution_number)*np.nan
for ik, kval in enumerate(k):
    for iw, wval in enumerate(w):
        for id, dval in enumerate(d):
            if stability_plus[ik,iw,id]:
                # print(kval,wval,dval)
                for n in range(int(solution_number[ik,iw,id])):
                    if stability[ik,iw,id,n]==2:
                        stab_xsol[ik,iw,id]=xsol[ik,iw,id,n]
                        stab_ysol[ik,iw,id] = ysol[ik,iw,id,n]
                        stab_zsol[ik,iw,id] = m_z(xsol[ik,iw,id,n],ysol[ik,iw,id,n],wval,kval,Gam)
                        break
                
k_cut = 72
w_cut = 20
d_cut = 42


from random import random
def starting_points(numb):
    points = np.zeros((numb,3))
    for i in range(numb):
        phi = random()*2*np.pi
        theta = random()*np.pi
        r = random()*0.5
        points[i,:] = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
    return points

def stab_find(kval,wval,dval):
    arguments = wval,kval, dval, Gam, sgam 
    traj = solve_ivp(gl2,(0,400),y0=np.reshape(starting_points(1),3),args=arguments,rtol=1e-13,atol=1e-25)
    ind = int(3/4*len(traj.t))
    return np.mean(traj.y[:,ind:],axis=1) 
    
    

def endp_wrapper(index):
    i , j = index//gridpoints, index%gridpoints
    k_cut = 72
    w_cut = 20
    d_cut = 42
    if stability_plus[k_cut,i,j]==0:
        solk = stab_find(k[k_cut],w[i],d[j])
    else:
        solk = np.array([stab_xsol[k_cut,i,j],stab_ysol[k_cut,i,j],stab_zsol[k_cut,i,j]])
    if stability_plus[i,w_cut,j]==0:
        solw = stab_find(k[i],w[w_cut],d[j])
    else:
        solw = np.array([stab_xsol[i,w_cut,j],stab_ysol[i,w_cut,j],stab_zsol[i,w_cut,j]])
    if stability_plus[i,j,d_cut]==0:
        sold = stab_find(k[i],w[j],d[d_cut])
    else:
        sold = np.array([stab_xsol[i,j,d_cut],stab_ysol[i,j,d_cut],stab_zsol[i,j,d_cut]])
    return solk, solw, sold

from multiprocessing import Pool
po = Pool(20)
k_cutarray = np.zeros((gridpoints,gridpoints,3))
w_cutarray = np.zeros_like(k_cutarray)
d_cutarray = np.zeros_like(k_cutarray)
result = po.imap(endp_wrapper,range(gridpoints**2),chunksize=100)
for index, res in enumerate(result):
    i , j = index//gridpoints, index%gridpoints
    k_cutarray[i,j], w_cutarray[i,j], d_cutarray[i,j] = res

np.save(root_path+'k_cutarray',k_cutarray)
np.save(root_path+'w_cutarray',w_cutarray)
np.save(root_path+'d_cutarray',d_cutarray)
