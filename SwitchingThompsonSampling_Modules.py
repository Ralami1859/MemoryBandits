from GlobalSTS_Modules import *
from PerArmSTS_Modules import *


def STS_Initialize(K, alpha0, beta0, globalSwitch):
	if globalSwitch:
		return GlobalSTS_Initialize(K, alpha0, beta0)
	else:
		return PerArmSTS_Initialize(K, alpha0, beta0)

#----------------------------------------------------------------------------------------------------------------
#                                            Arm recommended by STS 
#-----------------------------------------------------------------------------------------------------------------

def STS_recommendArm(ExpertWeigth, alphas, betas, withAggregation, globalSwitch):
	if globalSwitch:
		if withAggregation:
			return GlobalSTS_BA_RecommendArm(ExpertWeigth, alphas, betas)
		else: 
			return GlobalSTS_RecommendArm(ExpertWeigth, alphas, betas)	
	else:
		if withAggregation:
			return PerArmSTS_BA_RecommendArm(ExpertWeigth, alphas, betas)
		else: 
			return PerArmSTS_RecommendArm(ExpertWeigth, alphas, betas)	


#-------------------------------------------------------------------------------------------------------------
#               Expert distribution update (According to the message passing algorithm (Fernhead 2010)
#------------------------------------------------------------------------------------------------------------

def STS_updateChangeModel(ExpertWeigth, alphas, betas, ArmToPlay, reward, gamma, globalSwitch):
	if globalSwitch:
		return Global_STS_updateChangeModel(ExpertWeigth, alphas, betas, ArmToPlay, reward, gamma)
	else:
		return PerArm_STS_updateChangeModel(ExpertWeigth, alphas, betas, ArmToPlay, reward, gamma)


#------------------------------------------------------------------------------------------------------------
#             Arm hyperparameters update
#-----------------------------------------------------------------------------------------------------------

def	STS_updateArmModel(alphas, betas, ArmToPlay, reward, alpha0, beta0, gainGlobalSTS):
	K = np.size(alphas,1)
	gainGlobalSTS = np.append(gainGlobalSTS, reward)
	if reward == 1:
		alphas[:,ArmToPlay] += 1;
	else:
		betas[:,ArmToPlay] += 1;
	alphas = np.vstack([alpha0*np.ones((1,K)),alphas]) #Expert Newly created put at prior
	betas = np.vstack([beta0*np.ones((1,K)),betas])
	return alphas, betas, gainGlobalSTS

#-----------------------------------------------------------------------------------------------------------------
#               Discarding the worst expert to respect memory limitation
#----------------------------------------------------------------------------------------------------------------

def STS_Resampling(ExpertWeigth, alphas, betas, globalSwitch):
	if globalSwitch:
		return GlobalSTS_Resampling(ExpertWeigth, alphas, betas)
	else:
		return PerArmSTS_Resampling(ExpertWeigth, alphas, betas)


