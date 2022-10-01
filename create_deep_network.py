import numpy as np
import pandas as pd
import sympy as sy
import random
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid.axislines import SubplotZero
import time
import sys
import S_to_neural_net



def spline_interpolation(points_spline_interpolation):
    #returns the basis representation via splines
    #input: list of tuples (x_breakpoints,y_breakpoints) of a function, inclusive the endpoints
    #output: list of 4-tuples (x_breakpoints,weights for the splines (x_x_breakpoints)_+,linear part a, bias b)
    w=[]
    for s in range(len(points_spline_interpolation)):
        #print(s)
        breaks=copy.copy(points_spline_interpolation[s][0])
        y_breaks=copy.copy(points_spline_interpolation[s][1])
        if breaks[0]!=0:
            a=0
            b=y_breaks[0]
        else:
            a=(y_breaks[1]-y_breaks[0])/breaks[1]
            b=y_breaks[0]
            breaks=breaks[1:]
            y_breaks=y_breaks[1:]

        weights_spline=[]
        for k in range(len(breaks)-1):
            #print(k)
            c=y_breaks[k+1]-a*breaks[k+1]-b
            for l in range(k):
                #print(l)
                c+=-weights_spline[l]*(breaks[k+1]-breaks[l])
            weights_spline.append(c/(breaks[k+1]-breaks[k]))
        w.append((breaks[:-1],weights_spline,a,b))
    return w

def CPwL_to_S(breaks, weights, lin, const):
    #return S=T-l(x)=T(x)-(ax+b)
    #input: breakpoints, weights of breakpoints, linear part ax, constant part b of new CPwL function with vanishing on the endpoints; a,b for linear part
    #output: breaks, weights, linear part , const. part (=0) of S; linear part(a) and constant part(b) of the linear function l(x) 
    linear=evaluation_cpwl_basis_splines(1,breaks, weights, lin, const)-const
    return copy.copy(breaks), copy.copy(weights), lin-linear,0,linear,const

def evaluation_cpwl_basis_splines(x,breakpoints,weights,linear,constant):
    #Evaluation of a CPwL function f(x) with given representation f(x)=a*x+b+sum(m_j*(x-xi_j)_+)
    out=linear*x+constant
    for k in range(0,len(breakpoints)):
        if x>=breakpoints[k]:
            out+=weights[k]*(x-breakpoints[k])
    return out

def create_subfunctions(breakpoints,y_breakpoints,W):
    #input: breakpoints and values of a CPwL functions (including endpoints);width W
    #output: list of tuples (breakpoints, y_breakpoints) for the subfunctions;function l(x)=a*x+b (-> necessary for neural net at the end)
    
    n=len(breakpoints)-2
    q=sy.floor((W-2)/6)
    N0=(W-2)*q
    c=-sy.floor(-n/(q*(W-2)))
    
    breaks,weights,a_or,b_or=spline_interpolation([(breakpoints,y_breakpoints)])[0] #get the (inner) breakpoints, weights of the splines and the term m*x+b
    breakps,w,lin,const,a,b=CPwL_to_S(breaks, weights,a_or,b_or) #move the function st it holds f(0)=f(1)=0; computing the (new) breakpoints, weights, linear part and constant part and save the linear function(is important for the neural net)

   
    #adding arttificial breakpoints to get subfunctions with N0 breakpoints
    missing_breaks=c*N0-n
    left=breakps[-1]
    break_new=copy.copy(breakps)
    break_new=np.append(break_new,np.random.uniform(left,1,missing_breaks))
    break_new=np.sort(break_new)

    #computing the new values of our shifted and (by artificial breakpoints added) function
    y_new=[]
    for x in break_new:
        y_new.append(evaluation_cpwl_basis_splines(x,breakps,weights,lin,const))
    y_new=np.array(y_new)

    break_new=np.append(np.append(0,break_new),1) #add the endpoints 0 and 1 (because we considered above only the inner breakpoints)
    y_new=np.append(np.append(0,y_new),0)

    #creating the list of breakpoints,y_breakpoints of the subfunctions (e.g the breakpoints x_1,...,x_{N0} plus vanishiing on the neighboured breakpoints x_0 and x_{N0+1}(=N0+2 points))
    subs_points=[]
    for t in range(1,break_new.shape[0]-2,N0):
        x_breaks,y_breaks=copy.copy(break_new[t-1:t+N0+1]),copy.copy(y_new[t-1:t+N0+1]) #N0 breakpoints plus the two neighbored breakpoints
        y_breaks[0],y_breaks[-1]=0,0    
        subs_points.append((x_breaks,y_breaks))
    return subs_points,a,b

def create_neural_net_subs(subs_points,a,b,W):
    #input: list of tuples (breakpoints, y_breakpoints) of the subfunctions; linear function l(x)=a*x+b st S(x)=T(x)-l(x)
    #output: list of matrices, list of biases that determine the special ReLU network
    
    # create the matrices st every subfunction can created via a neural ne with 2 hidden layers 
    M0,bias,M1,M2,b0,b1=[],[],[],[],[],[]
    for k in range(len(subs_points)):
        matrix_0,bias_0,matrix_1,bias_1,matrix_2=S_to_neural_net.create_neural_net_from_S(W,subs_points[k][0],subs_points[k][1])
        M0.append(matrix_0)
        b0.append(bias_0),b1.append(bias_1)
        bias.append(bias_0),bias.append(bias_1)
        M1.append(matrix_1)
        M2.append(matrix_2)
    
    #create all matrices for the (deep) collected neural net
    
    #every second matrix= combintion of M0 of previous and M2 of next neural net
    M_even=[]
    M_even.append(copy.copy(M0[0]))
    for k in range(0,len(M0)-1):
        matrix=np.zeros((W,W))
        matrix[:,0]=copy.copy(M0[k+1]).transpose()
        matrix[-1,:]=copy.copy(M2[k])
        matrix[-1,-1]=1#collation channel
        M_even.append(matrix)
    M_even.append(copy.copy(M2[-1]))

    #combining all matrices in one list in the right order
    M_all=[]
    for k in range(len(M1)):
        M_all.append(M_even[k]),M_all.append(M1[k])
    M_last=copy.copy(M_even[-1])
    M_last[0]=a #adding a*x to the output node
    M_last[-1]=1#intact collation channel!
    M_all.append(M_last)

    bias.append(np.array([b])) #adding b to the output node 
    
    return M_all,bias

def create_deep_neural_net(breakpoints,y_breakpoints,W):
    #whole function
    #input: breakpoints,y_breakpoints of function including endpoints 0 and 1; width W
    #output: weight matrices and biases of deep neural net
    
    global width
    width=copy.copy(W)
    subs_points,a,b=create_subfunctions(breakpoints,y_breakpoints,width) #create subfunctions via the breakpoints,y_breakpoints
    M_all,bias=create_neural_net_subs(subs_points,a,b,width) #create matrices and biases
    
    return M_all,bias

def ReLU_special(a):
    #a: np array
    out=[]
    out.append(a[0])
    for k in range(1,a.shape[0]-1):
        out.append(max(0,a[k]))
    out.append(a[-1])
    return np.array(out)

def neural_deep(x,M,bias):
    if len(M)!=len(bias):
        return 'not compatible'
    res=ReLU_special(M[0].dot(np.array([x]))+bias[0])
    #print(res)
    for k in range(1,len(M)-1):
        res=ReLU_special(M[k]@res+bias[k])
        #print(res)
    #print(M[-1],bias[-1])
    res=M[-1]@res+bias[-1]
    return res[0]
