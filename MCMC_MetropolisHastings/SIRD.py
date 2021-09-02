# A function that stores all four differential equations for S, I, R, and D
def SIRD(y, t, p):
    '''
    Returns ordinary differential equation expressions
        Parameters:
            y (tuple): contains the initial values of S(usceptible), I(nfected),
                       R(ecovered), and D(ead), i.e. y=[s0,i0,r0,d0]
            t (numpy array): a grid of time points
            p (tuple): in this case, p only has one element, which is the value
                       of R nought
    '''
    mu = 2.5/p[0] - 1 # Calculate the value of mu using R nought
    dSdt = -2.5*y[0]*y[1] # dSdt = -beta * S * I
    dIdt = 2.5*y[0]*y[1] - (1+mu)*y[1] # dIdt = beta * S * I - (gamma + mu) * I,
                                       # here assuming gamma = 1
    dRdt = y[1] # dRdt = gamma * I, here assuming gamma = 1
    dDdt = mu*y[1] # dDdt = mu * I
    return [dSdt, dIdt, dRdt, dDdt] 
                                    
def IRD(y, t, p):
    '''
    Returns ordinary differential equation expressions
        Parameters:
            y (tuple): contains the initial values of D(ead), I(nfected), and 
                       R(ecovered), i.e. y=[d0,i0,r0]
            t (numpy array): a grid of time points
            p (tuple): in this case, p only has one element, which is the value
                       of R nought
    '''
    mu = 2.5/p[0] - 1 # Calculate the value of mu using R nought
    S = 1 - y[0] - y[1] - y[2] # Because y=[d0,i0,r0] and S+I+R+D=1 
    dIdt = 2.5*S*y[1] - (1+mu)*y[1] # dIdt = beta * S * I - (gamma + mu) * I,
                                    # here assuming gamma = 1
#     dRdt = y[1]
    dDdt = mu*y[1] # dDdt = mu * I
    return [dDdt, dIdt, y[1]]

def SID(y, t, p):
    '''
    Returns ordinary differential equation expressions
        Parameters:
            y (tuple): contains the initial values of S(usceptible), I(nfected),
                       and D(ead), i.e. y=[s0,i0,d0]
            t (numpy array): a grid of time points
            p (tuple): in this case, p only has one element, which is the value
                       of R nought
    '''
    mu = 2.5/p[0] - 1 # Calculate the value of mu using R nought
    dSdt = -2.5*y[0]*y[1] # dSdt = -beta * S * I
    dIdt = 2.5*y[0]*y[1] - (1+mu)*y[1] # dIdt = beta * S * I - (gamma + mu) * I,
                                       # here assuming gamma = 1
#     dRdt = y[1]
    dDdt = mu*y[1] # dDdt = mu * I
    return [dSdt, dIdt, dDdt]

