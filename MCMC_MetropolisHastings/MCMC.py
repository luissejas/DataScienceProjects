import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import os


def MCMC_model(sigma_prior, sigma_param, R_prior, 
               R_param, sird_model, initial, yobs, 
               sample_size, tune, sigmas, R0):
    '''
    Returns a summary table of the finished MCMC model speicified by the input
    values.
        Parameters:
            sigma_prior (int): a number label for the desired prior distribution
                for the sigmas (the detailed label meaning can be found in the
                function MCMC below)
            sigma_param (float/int): the value of the hyperparameter for the
                prior of sigmas
            R_prior (int): a number label for the desired prior distribution
                for R nought (the detailed label meaning can be found in the
                function MCMC below)
            R_param (float/int): the value of the hyperparameter for the
                prior of R nought
            sird_model: an ODE system container
            initial (list): a vector of the initial condition values
            yobs (np.array): an array storing the input curves after 
                adding noises
            sample_size (int): number of draws to be performed in the MCMC
            tune (int): number of tunings in the MCMC
            sigmas (list): true values of the sigmas
            R0 (float/int): true value of R nought
    '''
    sigma_shape = len(sigmas)
    with pm.Model() as basic_model:
        if sigma_prior not in [1, 2, 3, 4]:
            print("Input error. Try again")
            return None
        # Prior for sigmas
        if sigma_prior == 1:
            sigma = pm.HalfCauchy('sigma', sigma_param, 
                                  shape=sigma_shape)
            sigma_name = 'Half Cauchy'
        elif sigma_prior == 2:
            sigma = pm.HalfNormal('sigma', sigma_param, 
                                  shape=sigma_shape)
            sigma_name = 'Half Normal'
        elif sigma_prior == 3:
            sigma = pm.Bound(pm.HalfCauchy, 
                             lower=0.1, # If the prior for sigmas are bounded,
                                        # it is safe to assume that the sigmas
                                        # are larger or equal to 1
                             upper=1)('sigma', 
                                      sigma_param, 
                                      shape=sigma_shape)
            sigma_name = 'Bounded Half Cauchy'
        elif sigma_prior == 4:
            sigma = pm.Bound(pm.HalfNormal, 
                             lower=0.1,
                             upper=1)('sigma', 
                                      sigma_param, 
                                      shape=sigma_shape)
            sigma_name = 'Bounded Half Normal'
        if R_prior not in [1, 2]:
            print("Input error. Try again")
            return None
        # Prior for R0
        if R_prior == 1:
            R0 = pm.Bound(pm.Normal, 
                          lower=1.25, # As the values of R nought and death rate
                                      # (mu) are related to each other (detailed
                                      # relationship is explained in the main
                                      # notebook, R nought has a value range
                                      # betweeh 1.25 and 2.5 given that beta=2.5
                                      # and gamma=1)
                          upper=2.5)('R0', R_param[0], 
                                     R_param[1])
            R0_name = 'Bounded Normal'
        else:
            R0 = pm.Bound(pm.Lognormal, 
                          lower=1,
                          upper=2.5)('R0', R_param[0], 
                                     R_param[1])
            R0_name = 'Bounded Lognormal'
        
        curves = sird_model(y0=initial, theta=[R0])
        
        # Likelihood function choice: our sampling distribution for multiplicative 
        # noise around the curves
        Y = pm.Lognormal('Y', mu=pm.math.log(curves), 
                         sd=sigma, observed=yobs) # variances via sigma, data=yobs

        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(sample_size, step=step, cores=1, 
                          tune=tune, random_seed=44)
    
    # Extract the names of the prior
    print('Assuming the distributions of sigmas are '+sigma_name+\
          ', and distribution of R0 is '+R0_name+'.')
    
    lines = []
    for s in sigmas:
        lines.append(['sigma', {}, s])
    lines.append(['R0', {}, 1.67])
    pm.traceplot(trace, 
                 lines=lines) # The lines in the traceplots show the true values
                              # of the corresponding variables
    plt.show()
    plt.close()
    return pd.DataFrame(pm.summary(trace).round(5))


# The function MCMC requires interactive inputs to specify certain parameters
# of the MCMC model; the arguments here have the same meanings as the ones
# listed above in MCMC_model
def MCMC(sird_model, initial, yobs, sample_size, tune, sigmas, R0):
    sigma_prior = input("Please input the label (integer) "+\
                       "of the prior distribution of sigmas.\n"+\
                       "(1--Half Cauchy; 2--Half Normal; 3--"+\
                       "Bounded Half Cauchy; 4--Bounded Half Normal) \n")
    sigma_prior = int(sigma_prior)
    
    sigma_param = eval(input("\nPlease input the parameter of the prior "+\
                       "distribution of sigmas. \n(For Half Cauchy "+\
                       "distributions the parameter is Beta, and "+\
                       "for Half Normal distributions the parameter "+\
                       "is sigma.) \n"))
        
    R_prior = input("\nPlease input the label (integer) "+\
                     "of the prior distribution of R0.\n"+\
                     "(1--Bounded Normal; 2--Bounded Logormal)\n")
    R_prior = int(R_prior)
    
    R_param = eval(input("\nPlease input the parameters of the prior "+\
                         "distribution of R0, i.e. mean and sigma of "+\
                         "the distribution: (needs to be in list format"+\
                         "e.g., mean=2 and sigma=3 expressed as [2,3]) \n"))
    print("--------------------------------".center(os.get_terminal_size().columns))
    print("MCMC modeling starts from here".center(os.get_terminal_size().columns))
    print("--------------------------------".center(os.get_terminal_size().columns))
    # Extract the summary table from the MCMC model
    summary = MCMC_model(sigma_prior, sigma_param, R_prior, 
                         R_param, sird_model, initial, yobs, 
                         sample_size, tune, sigmas, R0)
    return summary