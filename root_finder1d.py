from math import isclose
import sys
from scipy.optimize import root
sys.path.append(r'/home/christian/Documents/Bachelor/numerics/')
from physical_functions_mf import *
from add_solutions_functions import *
from math import isclose
from multiprocessing import Pool

gridpoints = 91
root_path = '/home/christian/Documents/Bachelor/numerics/mean_field/'
name = '_4.4k13_1.5w5.5'

solnumb_file = root_path + f'solnumb' + name
xsol_file = root_path + f'xsol' +name
ysol_file = root_path + f'ysol' +name
params_file = root_path + f'params' +name
Gam = 1
sgam = 0.2

x_range = np.linspace(-0.5,0.5)


# params = "4.20004 0.75000007 0.3599999999999999"
# params=params.split()
# params = [float(p) for p in params]
# k0,w0,d0 = params

# p_range = 0.5
# k= np.linspace(k0-p_range,k0+p_range,gridpoints)
# w = np.linspace(w0-p_range,w0+p_range,gridpoints)
# d = np.linspace(d0-p_range,d0+p_range,gridpoints)

k= np.linspace(4.4,13,gridpoints)
w = np.linspace(1.5,5.5,gridpoints)
d = np.linspace(-3,3,gridpoints)
xsol = np.ones((gridpoints,gridpoints,gridpoints,30))
ysol = np.ones_like(xsol)
yvariant1 = np.zeros_like(xsol)
# xgrid = np.linspace(-0.5-1/10**5,0.5+1/10**5,40)
solution_number = np.zeros((gridpoints,gridpoints,gridpoints),dtype=np.int16)
def wrapper1d(index):
    xlist = np.ones(5)
    ylist = np.ones(5)
    sol_numb_tot =0
    ik, iw, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    kval, wval, dval = k[ik], w[iw], d[id]
    s=1
    yofx = y1ofx
    # else:
    if dval==0:
        if kval/(Gam+sgam)>1:
            try:
            #     1
            # except:
            #     pass
            # if True:
                rcrit = root_scalar(FKK,args=(kval/(Gam+sgam)),method='brentq',bracket=(1/10**7,10**6))
                if rcrit.root*np.sqrt(Gam*(Gam+sgam))>=wval:
                    sol_numb=3
                    r=root_scalar(Foriginal,args=(wval,kval,Gam,sgam),method='bisect',bracket=(-1,minF(wval,kval,Gam,sgam)))#,maxF(a,kwval)))
                    
                    xlist[0]= 0
                    ylist[0]=r.root
                    

                
                    r=root_scalar(Foriginal,args=(wval,kval,Gam,sgam),method='bisect',bracket=(minF(wval,kval,Gam,sgam),maxF(wval,kval,Gam,sgam)))#,maxF(a,kwval)))

                    xlist[1]=0
                    ylist[1]= r.root
                    


                    r=root_scalar(Foriginal,args=(wval,kval,Gam,sgam),method='bisect',bracket=(maxF(wval,kval,Gam,sgam),1))#,maxF(a,kwval)))
                    xlist[2] =0
                    ylist[2]= r.root
                    

                    
                
                else: 
                    r=root_scalar(Foriginal,args=(wval,kval,Gam,sgam),method='bisect',bracket=(-1,minF(wval,kval,Gam,sgam)))#,maxF(a,kwval)))
                    xlist[0] =0
                    ylist[0]= r.root
                    sol_numb=1

                return sol_numb, xlist, ylist
                # print(i,FKK(1/10**8,K[i]),FKK(10**7,K[i]))
            except:
                print(ik,iw,id)
                return None
        else:
            r=root_scalar(Foriginal,args=(wval,kval,Gam,sgam),method='bisect',bracket=(-1,minF(wval,kval,Gam,sgam)))#,maxF(a,kwval)))
                    
            xlist[0] =0
            ylist[0]= r.root
            sol_numb=1
            return sol_numb, xlist, ylist
    # elif wval==0:
    #     xlist[0]=0
    #     ylist[0]=0
    #     continue   
    sol_numb =0
    xgrid, points = set_xgird(wval,kval,dval,Gam,sgam,points=True)
    Func = np.zeros_like(xgrid)
    if xgrid[0]==-2:
        
        try:
            r = root(Foriginal_vec,args=(wval,kval,dval,Gam,sgam),x0=np.array([0,0])) 
            xlist[0]=r.x[0]
            ylist[0]=r.x[1]
            sol_numb=1
            x= r.x[0]
            y=r.x[1]
            z=m_z(x,y,wval,kval,Gam)
            if  not ((isclose(gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam)[0],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam)[1],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam)[2],0,rel_tol=10e-7,abs_tol=1/10**7))):
                # print('no sol for testcase 0,0',gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam),Fx(x,wval,kval,dval,Gam,sgam,s),x,y,z,'params: ',kval,wval,dval)
                xlist[0]=1
                ylist[0]=1
                sol_numb=0

        except:
            print('no solution: ',kval,wval,dval)


        
        # plt.figure()
        # plt.plot(x_range,insq(x_range,wval,kval,dval,Gam,sgam))
        # plt.show()
        return sol_numb, xlist, ylist
            # print(xgrid[-1],points[-1],xgrid[-1]==points[-1])
    xsol_list = []
    ysol_list = []
    variant_list = []
    for j in range(len(xgrid)-1):
        if np.any(xgrid[j]==points):
            y = y1ofx(xgrid[j],wval,kval,dval,Gam,sgam)
        if np.any(xgrid[j+1]!=points[1::2]):
            # print(True)
            Fl = Fx(xgrid[j],wval,kval,dval,Gam,sgam,s)
            Fu = Fx(xgrid[j+1],wval,kval,dval,Gam,sgam,s)
            if Fl*Fu<=0:
                r = root_scalar(Fx,args=(wval,kval,dval,Gam,sgam,s),method='bisect',bracket=(xgrid[j],xgrid[j+1]))
                xsol_list, ysol_list , variant_list, sol_numb =add_sol1(xlist,kval,wval,dval,r,xsol_list,ysol_list,variant_list,sol_numb,s)
                
            Fl = Fx2(xgrid[j],wval,kval,dval,Gam,sgam,s)
            Fu = Fx2(xgrid[j+1],wval,kval,dval,Gam,sgam,s)
            
            if Fl*Fu<=0:
                r = root_scalar(Fx2,args=(wval,kval,dval,Gam,sgam,s),method='bisect',bracket=(xgrid[j],xgrid[j+1]))
                xsol_list, ysol_list , variant_list, sol_numb =add_sol2(xlist,kval,wval,dval,r,xsol_list,ysol_list,variant_list,sol_numb,s)
                
                
    i=0
    relative_tol = 10e-7
    absolute_to = 1/10**7
    while i<sol_numb:
        x = xsol_list[i]
        y = ysol_list[i]
        z = m_z(x,y,wval,kval,Gam)
        g = gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam)[:,0]
        if  not ((isclose(g[0],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(g[1],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(g[2],0,rel_tol=10e-7,abs_tol=1/10**7))):
            # print(gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam),Fx(x,wval,kval,dval,Gam,sgam,s),x,y,z,variant_list[i],'params: ',kval,wval,dval)
            sol_numb-=1
            xsol_list.pop(i)
            ysol_list.pop(i)
            variant_list.pop(i)
        i+=1
    try:
        for ind in range(sol_numb):
            xlist[ind]= xsol_list[ind]
            ylist[ind]= ysol_list[ind]
    except:
        print('soli_nu:',sol_numb,kval,wval,dval)
        return None
        
    sol_numb_tot+=sol_numb

    s=2
    yofx = y2ofx
    xsol_list = []
    ysol_list = []
    variant_list = []
    sol_numb = 0
    for j in range(len(xgrid)-1):
        
        if np.any(xgrid[j+1]!=points[1::2]):
            # print(True)
            Fl = Fx(xgrid[j],wval,kval,dval,Gam,sgam,s)
            Fu = Fx(xgrid[j+1],wval,kval,dval,Gam,sgam,s)
            if Fl*Fu<=0:
                r = root_scalar(Fx,args=(wval,kval,dval,Gam,sgam,s),method='bisect',bracket=(xgrid[j],xgrid[j+1]))
                xsol_list, ysol_list , variant_list, sol_numb =add_sol1(xlist,kval,wval,dval,r,xsol_list,ysol_list,variant_list,sol_numb,s)
                
            Fl = Fx2(xgrid[j],wval,kval,dval,Gam,sgam,s)
            Fu = Fx2(xgrid[j+1],wval,kval,dval,Gam,sgam,s)
            
            if Fl*Fu<=0:
                r = root_scalar(Fx2,args=(wval,kval,dval,Gam,sgam,s),method='bisect',bracket=(xgrid[j],xgrid[j+1]))
                xsol_list, ysol_list , variant_list, sol_numb =add_sol2(xlist,kval,wval,dval,r,xsol_list,ysol_list,variant_list,sol_numb,s)
                
    for p in points:
        
        y = y1ofx(p,wval,kval,dval,Gam,sgam)
        r = root(Foriginal_vec,args=(wval,kval,dval,Gam,sgam),x0=np.array([p,y]))  
        new_sol=1
        for sol in xlist[:]:
            if isclose(sol,r.x[0],abs_tol=1/10**5):
                new_sol = 0
                break
        for sol in xsol_list:
            if isclose(sol,r.x[0],abs_tol=1/10**5):
                new_sol = 0
                break
        if new_sol:
            xsol_list.append(r.x[0])
            ysol_list.append(r.x[1])
            variant_list.append(0)
            sol_numb+=1
        y = y2ofx(p,wval,kval,dval,Gam,sgam)
        r = root(Foriginal_vec,args=(wval,kval,dval,Gam,sgam),x0=np.array([p,y]))  
        new_sol=1
        for sol in xlist[:]:
            if isclose(sol,r.x[0],abs_tol=1/10**5):
                new_sol = 0
                break
        for sol in xsol_list:
            if isclose(sol,r.x[0],abs_tol=1/10**5):
                new_sol = 0
                break
        if new_sol:
            xsol_list.append(r.x[0])
            ysol_list.append(r.x[1])
            variant_list.append(0)
            sol_numb+=1
    i=0
    while i<sol_numb:
        x = xsol_list[i]
        y = ysol_list[i]
        z = m_z(x,y,wval,kval,Gam)
        g = gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam)[:,0]
        if  not ((isclose(g[0],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(g[1],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(g[2],0,rel_tol=10e-7,abs_tol=1/10**7))):
            # print(gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam),Fx(x,wval,kval,dval,Gam,sgam,s),x,y,z,variant_list[i],'params: ',kval,wval,dval)
            sol_numb-=1
            xsol_list.pop(i)
            ysol_list.pop(i)
            variant_list.pop(i)
        else:
            i+=1
    try:
        for ind in range(sol_numb):
            xlist[int(sol_numb_tot)+ind]= xsol_list[ind]
            ylist[int(sol_numb_tot)+ind]= ysol_list[ind]
    except:
        print('soli_nu:',sol_numb,kval,wval,dval)
        return None
        
    sol_numb_tot+=sol_numb

            # print(kval,wval,dval)
            # print(-Gam*dval/(2*kval*wval))
            # print(xgrid)
    return sol_numb_tot, xlist, ylist


po = Pool(20)
result = po.imap(wrapper1d,range(gridpoints**3),chunksize=100)

reject = np.zeros_like(solution_number)
for index, res in enumerate(result):
    ik, iw, id = index//gridpoints**2,(index%gridpoints**2)//gridpoints, index%gridpoints
    soli, xsol[ik,iw,id,:5], ysol[ik,iw,id,:5] = res
    solution_number[ik,iw,id] = soli

# for i in range(50):
#     print(wrapper1d(i))



np.save(solnumb_file,solution_number)
np.save(xsol_file,xsol)
np.save(ysol_file,ysol)


np.save(root_path +'kparam'+name,k)
np.save(root_path +'wparam'+name,w)
np.save(root_path +'dparam'+name,d)
