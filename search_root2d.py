import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics')
from physical_functions_mf import *
from add_solutions_functions import *
from scipy.optimize import root
from multiprocessing import Pool
from time import time
t1 = time()
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/'
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


mgridpoints = 40
mgrid = np.zeros((mgridpoints,mgridpoints,2))
rho = np.linspace(0,0.5,mgridpoints)
phi = np.linspace(0,2*np.pi,mgridpoints)
for i, dist in enumerate(rho):
    for j, ang in enumerate(phi):
        mgrid[i,j,:]= dist*np.cos(ang), dist*np.sin(ang)

def root2d(k,w,d):
    solnumb =0
    rejections=0
    xsol_list =[]
    ysol_list = []
    rejected = []
    for i in range(mgrid.shape[0]):
        for j in range(mgrid.shape[1]):
            r = root(Foriginal_vec,args=(w,k,d,Gam,sgam),x0=mgrid[i,j])
            xsol_list, ysol_list, solnumb, rejections, rejected = add_sol2d(k,w,d,r,xsol_list,ysol_list,solnumb,rejections,rejected)
    return solnumb, xsol_list, ysol_list, rejections

check_sol = np.zeros_like(solution_number)
for ik, kval in enumerate(k):
    for iw, wval in enumerate(w):
        for id, dval in enumerate(d):
            for i in range(-2,3):
                if solution_number[ik,iw,(id+i)%gridpoints]:
                    check_sol[ik,iw,id]=1
                elif solution_number[ik,(iw+i)%gridpoints,id]:
                    check_sol[ik,iw,id]=1
                elif solution_number[(ik+i)%gridpoints,iw,id]:
                    check_sol[ik,iw,id]=1

def wrapper(index):
    ik, iw, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    kval, wval, dval = k[ik], w[iw], d[id]
    if check_sol[ik,iw,id] and dval!=0:
        return root2d(kval,wval,dval)
    else:
        return [0]
    

po = Pool(20)
result = po.imap(wrapper,range(gridpoints**3))

reject = np.zeros_like(solution_number)
for index, res in enumerate(result):
    ik, iw, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    if check_sol[ik,iw,id] and d[id]!=0:
        soli, xsol[ik,iw,id,:soli], ysol[ik,iw,id,:soli], reject[ik,iw,id] = res
        solution_number[ik,iw,id] = soli



solnumb_file = root_path+ f'solnumb' + name
xsol_file = root_path+f'xsol' +name
ysol_file = root_path+f'ysol' +name

np.save(solnumb_file,solution_number)
np.save(xsol_file,xsol)
np.save(ysol_file,ysol)
print(time()-t1)
