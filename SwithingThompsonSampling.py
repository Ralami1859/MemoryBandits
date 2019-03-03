from __future__ import division
import numpy as np
from SwitchingThompsonSampling_Modules import *

import matplotlib.pyplot as plt


"""
---------------------------------------------------------------------------------------------------------------------------
                                         Define the environment
---------------------------------------------------------------------------------------------------------------------------
"""

BernoulliMeansMatrix = np.matrix([[0.9,0.1,0.1],[0.2,0.1,0.9],[0.1,0.9,0.2]])
NbrSwitch = np.size(BernoulliMeansMatrix,0) # Overal number of switch
K = np.size(BernoulliMeansMatrix,1) # number of Arm
Horizon = 900
BernoulliMeansMatrix = constructBernoulliMeansMatrix(BernoulliMeansMatrix, Horizon)

plt.plot(BernoulliMeansMatrix, linewidth = 2.0)
plt.xlabel('Time step')
plt.ylabel('Means')
plt.axis([1, Horizon, 0, 1])
plt.title('Environment')
plt.show()

"""
--------------------------------------------------------------------------------------------------------------------------
						Parameters of the Switching Thompson Sampling 
-------------------------------------------------------------------------------------------------------------------------
"""

gamma = NbrSwitch/Horizon # Switching Rate 
#gamma = np.array([]) # for a per arm switch gamma is a K-array, each entry corresponds to the switching rate of each arm.
NbrMaxExperts = min(500, Horizon) # Maximum number of expert (memory limitation) 
alpha0 = 1 # Prior of Thompson Sampling 
beta0 = 1   # Prior of Thompson Sampling

"""
-------------------------------------------------------------------------------------------------------------------------
	  			Versions of the Switching Thompson Sampling
-------------------------------------------------------------------------------------------------------------------------
"""

#withAggregation = True          # (2017)
withAggregation = False           # (2013)


"""
-----------------------------------------------------------------------------------------------------------------------
                                           Global or Per Arm Switch ?
-----------------------------------------------------------------------------------------------------------------------
"""
#globalSwitch = True         # For a global Switch
globalSwitch = False      # For a per Arm Switch

"""
------------------------------------------------------------------------------------------------------------------------
				Option : Plotting the regret
------------------------------------------------------------------------------------------------------------------------
"""

PlotRegret = True
if PlotRegret:
	vectReward = np.array([])

"""
-------------------------------------------------------------------------------------------------------------------------
					Launching the Switching Thompson Sampling  (Enjoy !)
-------------------------------------------------------------------------------------------------------------------------
"""

(alphas, betas, ExpertWeigth, gainGlobalSTS) = STS_Initialize(K, alpha0, beta0, globalSwitch)

#Interation with the environment ...
for t in range(Horizon):
	print('t = ' + str(t+1))


	ArmToPlay = STS_recommendArm(ExpertWeigth, alphas, betas, withAggregation, globalSwitch) # Arm selection	

	reward = int ((np.random.uniform() < BernoulliMeansMatrix[t, ArmToPlay]) == True) # Play the chosen Arm
	reward = rewardCorrection(reward)

	ExpertWeigth = STS_updateChangeModel(ExpertWeigth, alphas, betas, ArmToPlay, reward, gamma, globalSwitch) #update expert distribution

	(alphas, betas, gainGlobalSTS) = STS_updateArmModel(alphas, betas, ArmToPlay, reward, alpha0, beta0, gainGlobalSTS) # Update Arms hyperparameters
	
	if np.size(ExpertWeigth,0) > NbrMaxExperts :
		(ExpertWeigth, alphas, betas) = STS_Resampling(ExpertWeigth, alphas, betas, globalSwitch) # Respecting memory limitations   

	if PlotRegret:
		vectReward = np.append(BernoulliMeansMatrix[t, ArmToPlay], vectReward) 
	
print('End Of learning ......')
print('Plotting the results.....')

"""
------------------------------------------------------------------------------------------------------------------------------------------
                                                      Plotting the results
------------------------------------------------------------------------------------------------------------------------------------------
"""

gainGlobalSTS = np.cumsum(gainGlobalSTS)
plt.plot(range(Horizon),gainGlobalSTS.tolist(),marker='.',label = "Global STS")
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('cumulative gain')
plt.title('Cumulative gain of Global STS')
plt.show()


if PlotRegret:
	gainOptimalPolicy = np.amax(BernoulliMeansMatrix,1)
	plt.plot(range(Horizon),np.cumsum(gainOptimalPolicy - vectReward.tolist()),marker='.',label = "Global STS")
	plt.legend(loc='upper left')
	plt.xlabel('Time step')
	plt.ylabel('cumulative regret')
	plt.title('Cumulative regret of Global STS')
	plt.show()


