import numpy as np
class KF:
    def __init__(self , init_x: float , 
                        init_v:float , 
                        acc_var:float) ->None:
        self._x = np.array([init_x,init_v])
        self._acc_var = acc_var
        self._P = np.eye(2)


    def predict (self , dt: float) ->None:
        #x_new = F*x_old
        # P_new = F * P * Ft + G * Gt * a
        F = np.array([[1,dt],[0,1]])
        new_x = F.dot(self._x)

        G = np.array([0.5*dt**2 , dt]).reshape((2,1))
        new_p = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._acc_var

        self._P = new_p
        self._x = new_x

    def update(self , meas_val:float , meas_var:float):
        # y = z - H * x_old
        # S = H * P * Ht + R
        # K = P * Ht * S^-1
        # x_new = x_old + K * y
        # P_new = (I - K * H) * P

        H = np.array([1,0]).reshape((1,2))

        z = np.array([meas_val])
        R = np.array([meas_var])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R 
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x


    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x
    
    @property
    def pos(self) ->float:
        return self._x[0]
    
    @property
    def vel(self) ->float:
        return self._x[1]