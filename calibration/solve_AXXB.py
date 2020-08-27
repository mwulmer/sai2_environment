import numpy as np
import cv2


# solution to AX=XB
def cal_X(AA,BB):
    [m,n] = AA.shape
    n = int(n/4)

    A = np.zeros((9*n,9))
    b = np.zeros((9*n,1))

    for i in range(1,n+1):
        Ra = AA[0:3,4*i-4:4*i-1]
        Rb = BB[0:3,4*i-4:4*i-1]
        iden = np.identity(3)  
        A[9*i-9:9*i,:] = np.array(np.kron(Ra,iden )+np.kron(-iden,Rb.T))
        a = np.array(np.kron(Ra,iden )+np.kron(-iden,Rb.T))
        print(a)
    
    u,s,v = np.linalg.svd(A)
    x = v[:,-1]
    R = np.resize(x[0:9],(3,3))
    R = R.T
    
    R = np.sign(np.linalg.det(R))/abs(np.linalg.det(R)**(1/3)*R)
    u,s,v = np.linalg.svd(R)
    R = u*v.T
    if np.linalg.det(R<0):
        R = u*np.diag([1,1,-1])*v.T
    C = np.zeros(3*n,3)
    d = np.zeros(3*n,1)
    I = np.identity(3)

    for i in range(1,n+1):
        C[3*i-3:3*i,:] = I - AA[0:3,4*i-4:4*i-1]
        d[3*i-3:3*i,:] = AA[0:3,4*i-1] - R.dot(BB[0:3,4*i-1])
    
    t = np.linalg.solve(C,d) #lsqt

    X = np.concatenate((R,t.T),axis = 1)
    X = np.concatenate((X,[[0,0,0,1]]))
    return X

def T_matrix():
    theta = np.random.uniform(0,1.57)
    ra = np.array([[1,0,0]])
    rb = np.array([[0,np.cos(theta),-np.sin(theta)]])
    rc = np.array([[0,np.sin(theta),np.cos(theta)]])
    t = np.random.normal(loc=0.0, scale=5, size=(3,1))
    temp = np.concatenate((ra.T,rb.T,rc.T,t),axis=1)
    T = np.concatenate((temp,[[0,0,0,1]]))
    return T
def B_matrix():
    theta = np.random.uniform(0,1.57)
    ra = np.array([[np.cos(theta),-np.sin(theta),0]])
    rb = np.array([[np.sin(theta),np.cos(theta),0]])
    rc = np.array([[0,0,1]])
    t = np.random.normal(loc=0.0, scale=5, size=(3,1))
    temp = np.concatenate((ra.T,rb.T,rc.T,t),axis=1)
    T = np.concatenate((temp,[[0,0,0,1]]))
    return T

if __name__ == "__main__":

    # bot =np.array([[0,0,0,1]])

    A1 = T_matrix()
    A2 = T_matrix()
    AA1 = np.linalg.inv(A2).dot(A1)

    A1 = T_matrix()
    A2 = T_matrix()
    AA2 = np.linalg.inv(A2).dot(A1)

    A1 = T_matrix()
    A2 = T_matrix()
    AA3 = np.linalg.inv(A2).dot(A1)

    A1 = T_matrix()
    A2 = T_matrix()
    AA4 = np.linalg.inv(A2).dot(A1)
    
    AA = np.concatenate((AA1,AA2,AA3,AA4),axis=1)

    C1 = T_matrix()
    C2 = T_matrix()
    CC1 = C2.dot(np.linalg.inv(C1))

    C1 = T_matrix()
    C2 = T_matrix()
    CC2 = C2.dot(np.linalg.inv(C1))

    C1 = T_matrix()
    C2 = T_matrix()
    CC3 = C2.dot(np.linalg.inv(C1))

    C1 = T_matrix()
    C2 = T_matrix()
    CC4 = C2.dot(np.linalg.inv(C1))
    CC = np.concatenate((CC1,CC2,CC3,CC4),axis=1)
    #X = cal_X(AA,CC)

    Ar = []
    At = []
    Br = []
    Bt = []
    for i in range(5):
        Ar.append(T_matrix()[0:3,0:3]) 
        At.append(T_matrix()[0:3,-1])
        Br.append(B_matrix()[0:3,0:3])
        Bt.append(B_matrix()[0:3,-1])

    

    
    R_cam2gripper,t_cam2gripper = cv2.calibrateHandEye(Ar,At,Br,Bt,method = cv2.CALIB_HAND_EYE_TSAI )
    print(R_cam2gripper,t_cam2gripper)



    
