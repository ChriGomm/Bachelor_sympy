from math import isclose
import sys
sys.path.append(r'/home/christian/Documents/Bachelor/numerics/')
from physical_functions_mf import *
# add_solution functions
Gam = 1
sgam = 0.2
def add_sol1(xsol,kval,wval,dval,r,xsol_list,ysol_list,variant_list,sol_numb,s,comment=False):    
    if s==1:
        yofx = y1ofx
    else:
        yofx= y2ofx
    new_sol =1
    for sol in xsol:
        if isclose(sol,r.root,abs_tol=1/10**6):
            new_sol = 0
            if comment:
                print('different sol: ',sol-r.root)
            break
    for sol in xsol_list:
        if isclose(sol,r.root,abs_tol=1/10**6):
            new_sol = 0
            if comment:
                print('different sol: ',sol-r.root)
            break
    if new_sol:
        y =yofx(r.root,wval,kval,dval,Gam,sgam)
        if  isclose(r.root,0,abs_tol=1/10**9):
            if comment:
                print('root at 0: ',r.root)
            pass
        elif isclose(y,0,abs_tol=1/10**9):
            if (kval>Gam+sgam and isclose(check_spec(wval,kval,dval,Gam,sgam),0,abs_tol=1/10**9)):
                if isclose(Fx2(r.root,wval,kval,dval,Gam,sgam,s),0,abs_tol=1/10**9):
                    if comment:
                        print('closeness to root: ',Fx(r.root,wval,kval,dval,Gam,sgam,s))
                    xsol_list.append(r.root)
                    ysol_list.append(y)
                    variant_list.append(s)
                    sol_numb+=1
        else:
            if isclose(Fx2(r.root,wval,kval,dval,Gam,sgam,s),0,abs_tol=1/10**9):
                if comment:
                    print('closeness to root: ',Fx(r.root,wval,kval,dval,Gam,sgam,s))
                xsol_list.append(r.root)
                ysol_list.append(y)
                variant_list.append(s)
                sol_numb+=1
    return xsol_list, ysol_list, variant_list, sol_numb

def add_sol2(xsol,kval,wval,dval,r,xsol_list,ysol_list,variant_list,sol_numb,s,comment=False):
    if s==1:
        yofx = y1ofx
    else:
        yofx= y2ofx
    new_sol=1
    for sol in xsol:
        if isclose(sol,r.root,abs_tol=1/10**6):
            new_sol = 0
            if comment:
                print('different sol: ',sol-r.root)
            break
    for sol in xsol_list:
        if isclose(sol,r.root,abs_tol=1/10**6):
            new_sol = 0
            if comment:
                print('different sol: ',sol-r.root)
            break
    if new_sol:
        y = yofx(r.root,wval,kval,dval,Gam,sgam)
        if  isclose(r.root,0,abs_tol=1/10**9):
            if comment:
                print('root at 0: ',r.root)
            pass
        elif isclose(y,0,abs_tol=1/10**9):
            if (kval>Gam+sgam and isclose(check_spec(wval,kval,dval,Gam,sgam),0,abs_tol=1/10**9)):
                if isclose(Fx(r.root,wval,kval,dval,Gam,sgam,s),0,abs_tol=1/10**9):
                    if comment:
                        print('closeness to root: ',Fx(r.root,wval,kval,dval,Gam,sgam,s))
                    xsol_list.append(r.root)
                    ysol_list.append(y)
                    variant_list.append(s)
                    sol_numb+=1
        else:
            if isclose(Fx(r.root,wval,kval,dval,Gam,sgam,s),0,abs_tol=1/10**9):
                if comment:
                    print('closeness to root: ',Fx(r.root,wval,kval,dval,Gam,sgam,s))
                xsol_list.append(r.root)
                ysol_list.append(y)
                variant_list.append(s)
                sol_numb+=1
    return xsol_list, ysol_list, variant_list, sol_numb

def add_sol2d(kval,wval,dval,r,xsol_list,ysol_list,sol_numb,rejections,rejected,comment=False):    
    
    new_sol =1
    for i in range(len(xsol_list)):
        if isclose(xsol_list[i],r.x[0],abs_tol=1/10**6) and isclose(ysol_list[i],r.x[1],abs_tol=1/10**6):
            new_sol = 0
            if comment:
                print('different sol: ',sol-r.root)
            break
    if new_sol:
        for sol in rejected:
            if isclose(sol[0],r.x[0],abs_tol=1/10**9) and isclose(sol[1],r.x[1],abs_tol=1/10**6):
                new_sol = 0
                if comment:
                    print('rejected sol: ',sol-r.root)
                break
    if new_sol:
        x = r.x[0]
        y = r.x[1]
        z = m_z(x,y,wval,kval,Gam)
        g  = gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam)[:,0]
        if  not ((isclose(g[0],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(g[1],0,rel_tol=10e-7,abs_tol=1/10**7)) and (isclose(g[2],0,rel_tol=10e-7,abs_tol=1/10**7))):
            # print(gl_np(0,np.array([x,y,z]),wval,kval,dval,Gam,sgam),x,y,z,'params: ',kval,wval,dval)
            rejections +=0
            rejected.append((x,y))
        else:
            xsol_list.append(x)
            ysol_list.append(y)
            sol_numb+=1
    return xsol_list, ysol_list, sol_numb, rejections, rejected
