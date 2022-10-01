import numpy as np
import pandas as pd
import sympy as sy
import random
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid.axislines import SubplotZero
import time


def evaluation_cpwl_basis_splines(x,breakpoints,weights,linear,constant):
    #Evaluation of a CPwL function f(x) with given representation f(x)=a*x+b+sum(m_j*(x-xi_j)_+)
    out=linear*x+constant
    for k in range(0,len(breakpoints)):
        if x>=breakpoints[k]:
            out+=weights[k]*(x-breakpoints[k])
    return out

    
    
def H(s,i,j,breakpoints,W):
    #function H_i,j(s)
    q=sy.floor((W-2)/6)
    
    xi=copy.copy(breakpoints[j*q])
    left=copy.copy(breakpoints[j*q-i])
    right=copy.copy(breakpoints[j*q+1])

    if s<left or s>right:
        return 0
    if s<=xi:
        return (s-left)/(xi-left)
    if s>xi:
        return (s-right)/(xi-right)

    
    



def func_evaluation_H_basis(x,M,weights,breakpoints):
    #evaluation of S(x) with the basis representation of phi_k with belaonging weights and the original breakpoints
    c=0
    for k in range(len(M)):
        i,j,_=M[k]
        a=weights[k]
        c+=a*H(x,i,j,breakpoints,W)
    return c

def S_into_basis_phi(breakpoints,y_breakpoints,W):
    #input: S(x_0,x_1,...,x_n,x_N+1) as S(x_k)=y_k (saved in breakpoints, y_breapoints)
    #W: width of later produced neural net
    #output: weights (to the functions phi_k), principal breakpoints xi_j=x_(q*j)
    
   
    def weights_basis_phi(breakpoints,y_breakpoints,M):
        #calculation of weights to the basis phi_k
        #input: breakpoints x_0,x_1,...,x_N,x_N+1 with the condition f(x_0)=f(x_N+1)=0
        #input: M= triple of (i,j,k) s.t. phi_k=H_i,j (sorted by k)
        #output: list a with the weights
        if y_breakpoints[0]!=0 or y_breakpoints[-1]!=0:
            return 'Function endpoints not right'
        a=[]
        a.append(y_breakpoints[1]/H(breakpoints[1],q,1,breakpoints,W)) 
        support=np.array([[k for k in range(1,q+1)] for _ in range(W-2)]).flatten()

        c_out,c_in=0,0
        c1,c2=0,0

        for k in range(2,len(breakpoints)-1):

            supp=support[k-1]

            start1=time.time()
            y=y_breakpoints[k]
            x=breakpoints[k]
            c=y #counter for y+sum a_l*phi_l(x_k),l=1,...,k-1
            for l in range(k-(supp-1),k):
                i,j,_=M[l-1]
                c+=-a[l-1]*H(x,i,j,breakpoints,W)
            i_,j_,_=M[k-1] 
            a.append(c/H(x,i_,j_,breakpoints,W)) #dividing the sum through phi_k(x_k)
        return a




    q=sy.floor((W-2)/6)
    if q>1:
        w=weights_basis_phi(breakpoints,y_breakpoints,bijection_ij_k(W))
        principal_breakpoints=breakpoints[q::q]
    elif q==1:
        w=weights_basis_phi(breakpoints,y_breakpoints,bijection_ij_k(W))
        principal_breakpoints=breakpoints[1:-1]
    
    return np.array(w), principal_breakpoints
    
    
    
    
def Lambdas(weights,W):
    #inout: weights of the basis functions phi_k,k=1,...,N; W as width
    #output: sets Lambda+ and Lambda- 
    
    q=sy.floor((W-2)/6)
    N=(W-2)*q
    
    Lambda_plus=[]
    Lambda_minus=[]
    for k in range(3*q):
        Lp=[]
        Lm=[]
        for j in range(1,N+1):
            if j % (3*q)==k and weights[j-1] >=0:
                Lp.append(j)
            if j % (3*q)==k and weights[j-1] <0:
                Lm.append(j)
        if Lp:
            Lambda_plus.append(Lp)
        if Lm:
            Lambda_minus.append(Lm)
    return Lambda_plus,Lambda_minus 


def H_extended(s,i,j,breakpoints,W):
    #function H_i,j(s)
    q=sy.floor((W-2)/6)
    xi=copy.copy(breakpoints[j*q])
    left=copy.copy(breakpoints[j*q-i])
    right=copy.copy(breakpoints[j*q+1])

    if s<=xi:
        return (s-left)/(xi-left)
    if s>xi:
        return (s-right)/(xi-right)
    
    
def from_k_to_ij(k,W):
    #returns the indices (i,j) of the hat function H_i,j st phi_k=H_i,j
    q=sy.floor((W-2)/6)
    return (-k%q)+1,sy.floor((k-1)/q)+1

def bijection_ij_k(W):
    #reproduce the triple (i,j,k) s.t. phi_k=H_i,j (sorted by k)
    #input: W s.t. N=(W-2)q with q=floor((W-2)/6) and function is element of S(x_0,x_1,...,x_N,x_N+1)
    q=sy.floor((W-2)/6)
    L=[]
    for j in range(1,W-1):
        for i in range(1,q+1):
            L.append((i,j,j*q-i+1))
    M=[]
    for j in range(1,W-1):
        for i in range(1,q+1):
            M.append(L[j*q-i])
    return M

def Sk_points_for_interpolation_plus(Lp,principal_breakpoints,breakpoints,weights,W):
    #function that returns for any index set of Lp_k of Lp the breakpoints (x,y) for construction of the function S_k
    #input: index set Lp, principal breakpoints xi_j, original breakpoints x_k
    #output format: List of tuples (x_breakpoints,y_breakpoints) for the functions S_k
    points_spline_interpolation=[]
    for lists in Lp:
        breaks=[]
        y_breaks=[]
        for l in lists:
            i,j=from_k_to_ij(l,W)
            w=weights[l-1]

            if j!=1:
                breaks.append(principal_breakpoints[j-2])
                y_breaks.append(w*H_extended(principal_breakpoints[j-2],i,j,breakpoints,W))
            else:
                breaks.append(0)
                y_breaks.append(w*H_extended(0,i,j,breakpoints,W))

            breaks.append(principal_breakpoints[j-1])
            y_breaks.append(w)

            if j!=len(principal_breakpoints):
                breaks.append(principal_breakpoints[j])
                y_breaks.append(w*H_extended(principal_breakpoints[j],i,j,breakpoints,W))
            else:
                breaks.append(breakpoints[-1])
                y_breaks.append(w*H_extended(breakpoints[-1],i,j,breakpoints,W))
        points_spline_interpolation.append((np.array(breaks),np.array(y_breaks)))
    return points_spline_interpolation

def Sk_points_for_interpolation_minus(Lp,principal_breakpoints,breakpoints,weights,W):
    #function that returns for any index set of Lp_k of Lp the breakpoints (x,y) for construction of the function S_k
    #input: index set Lp, principal breakpoints xi_j, original breakpoints x_k
    #output format: List of tuples (x_breakpoints,y_breakpoints) for the functions S_k
    points_spline_interpolation=[]
    for lists in Lp:
        breaks=[]
        y_breaks=[]
        for l in lists:
            i,j=from_k_to_ij(l,W)
            w=weights[l-1]

            if j!=1:
                breaks.append(principal_breakpoints[j-2])
                y_breaks.append(-w*H_extended(principal_breakpoints[j-2],i,j,breakpoints,W))
            else:
                breaks.append(0)
                y_breaks.append(-w*H_extended(0,i,j,breakpoints,W))

            breaks.append(principal_breakpoints[j-1])
            y_breaks.append(-w)

            if j!=len(principal_breakpoints):
                breaks.append(principal_breakpoints[j])
                y_breaks.append(-w*H_extended(principal_breakpoints[j],i,j,breakpoints,W))
            else:
                breaks.append(breakpoints[-1])
                y_breaks.append(-w*H_extended(breakpoints[-1],i,j,breakpoints,W))
        points_spline_interpolation.append((np.array(breaks),np.array(y_breaks)))
    return points_spline_interpolation


def Sk_spline_interpolation(points_spline_interpolation):
    #returns the basis representation via splines
    #input: list of tuples (x_breakpoints,y_breakpoints) for every S_k-function
    #output: list of 4-tuples (x_breakpoints,weights for the splines (x_x_breakpoints)_+,linear part a, bias b)
    w=[]
    for s in range(len(points_spline_interpolation)):
        breaks=points_spline_interpolation[s][0]
        y_breaks=points_spline_interpolation[s][1]
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
            c=y_breaks[k+1]-a*breaks[k+1]-b
            for l in range(k):
                c+=-weights_spline[l]*(breaks[k+1]-breaks[l])
            weights_spline.append(c/(breaks[k+1]-breaks[k]))
        w.append((breaks[:-1],weights_spline,a,b))
    return w


def ReLU(a):
    #a: np array
    out=[]
    for k in a:
        out.append(max(0,k))
    return np.array(out)


def net(x,matrix_0,bias_0,matrix_1,bias_1,matrix_2):
    nodes_1=ReLU(matrix_0.dot(np.array([x]))+bias_0)
    nodes_2=ReLU(matrix_1.dot(nodes_1)+bias_1)
    out=matrix_2@nodes_2
    return out



def create_neural_net_from_S(W,breakpoints,y_breakpoints):
    global width
    width=copy.copy(W)
    if W>=8:
        q=sy.floor((width-2)/6)
        N=(width-2)*q
        if N+2!=len(breakpoints):
            return 'number of breakpoints does not match'
        weights,principal_breakpoints=S_into_basis_phi(breakpoints,y_breakpoints,width) #create weights and pricipal breakpoint for basis of {phi_k}

        Lp,Lm=Lambdas(weights,width) #create index sets Lambda
        Sk_points_interpolation_plus,Sk_points_interpolation_minus=Sk_points_for_interpolation_plus(Lp,principal_breakpoints,breakpoints,weights,width),Sk_points_for_interpolation_minus(Lm,principal_breakpoints,breakpoints,weights,width) #create for every S_k the interpoinlation points
        weights_Sk_p,weights_Sk_m=Sk_spline_interpolation(Sk_points_interpolation_plus),Sk_spline_interpolation(Sk_points_interpolation_minus) #create spline basis for every S_k
        weigths_Sk=weights_Sk_p+weights_Sk_m #summarize all weights 

        #creating first bunch of nodes x,(x-xi_1)_+,...,(x_xi_(W-2))_+,0
        matrix_0=np.ones((width,1))
        matrix_0[-1]=0

        bias_0=np.zeros(width)
        bias_0[1:width-1]=-np.array(principal_breakpoints)
        bias_0

        #creating weight matrix between first and second hidden layer
        bias_1=np.zeros(width)
        matrix_1=np.zeros((width,width))
        matrix_1[0,0]=1
        matrix_1[-1,-1]=1 #weight 1 of collation channel
        c=1
        for s in range(len(weigths_Sk)):
            vector_1=np.zeros((width-2))
            for breakp,weight in zip(weigths_Sk[s][0],weigths_Sk[s][1]):
                vector_1[np.where(principal_breakpoints==breakp)[0]]=weight
            vector_1=np.append(weigths_Sk[s][2],vector_1)
            matrix_1[c,0:width-1]=vector_1
            c+=1
            bias_1[s+1]=weigths_Sk[s][3]

        #creating matrix to create output node
        matrix_2=np.zeros(width)
        matrix_2[1:len(weights_Sk_p)+1]=1
        matrix_2[len(weights_Sk_p)+1:len(weights_Sk_p)+len(weights_Sk_m)+1]=-1

        return matrix_0,bias_0,matrix_1,bias_1,matrix_2
    if width>=8 and width<14:
        q=sy.floor((width-2)/6)
        #return q
        N=(width-2)*q
        weights,principal_breakpoints=S_into_basis_phi(breakpoints,y_breakpoints,width) #create weights and pricipal breakpoint for basis of {phi_k}
        Lp,Lm=Lambdas(weights,width) #create index sets Lambda
        Sk_points_interpolation_plus,Sk_points_interpolation_minus=Sk_points_for_interpolation_plus(Lp,principal_breakpoints,breakpoints,S_into_basis_phi(breakpoints,y_breakpoints,width)[0],width),Sk_points_for_interpolation_minus(Lm,principal_breakpoints,breakpoints,S_into_basis_phi(breakpoints,y_breakpoints,width)[0],width) #create for every S_k the interpoinlation points
        weights_Sk_p,weights_Sk_m=Sk_spline_interpolation(Sk_points_interpolation_plus),Sk_spline_interpolation(Sk_points_interpolation_minus) #create spline basis for every S_k
        weigths_Sk=weights_Sk_p+weights_Sk_m #summarize all weihts 

        #creating first bunch of nodes x,(x-xi_1)_+,...,(x_xi_(W-2))_+,0
        matrix_0=np.ones((width,1))
        matrix_0[-1]=0

        bias_0=np.zeros(width)
        bias_0[1:width-1]=-np.array(principal_breakpoints)
        bias_0

        #creating weight matrix between first and second hidden layer
        bias_1=np.zeros(width)
        matrix_1=np.zeros((width,width))
        matrix_1[0,0]=1
        c=1
        for s in range(len(weigths_Sk)):
            vector_1=np.zeros((width-2))
            for breakp,weight in zip(weigths_Sk[s][0],weigths_Sk[s][1]):
                vector_1[np.where(principal_breakpoints==breakp)[0]]=weight
            vector_1=np.append(weigths_Sk[s][2],vector_1)
            matrix_1[c,0:width-1]=vector_1
            c+=1
            bias_1[s+1]=weigths_Sk[s][3]

        #creating matrix to create output node
        matrix_2=np.zeros(width)
        matrix_2[1:len(weights_Sk_p)+1]=1
        matrix_2[len(weights_Sk_p)+1:len(weights_Sk_p)+len(weights_Sk_m)+1]=-1
        
        return matrix_0,bias_0,matrix_1,bias_1,matrix_2
