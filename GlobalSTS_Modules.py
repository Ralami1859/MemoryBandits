import numpy as np

import sys


def constructBernoulliMeansMatrix(BernoulliMeansMatrix, Horizon):
	Size = BernoulliMeansMatrix.shape
	NbrPeriode = Size[0]
	TimePeriode = Horizon/NbrPeriode
	K = Size[1]
	res = np.zeros((Horizon, K))
	for k in range(K):
		vect = np.array([])
		for periode in range(NbrPeriode):
			vect = np.append(vect, BernoulliMeansMatrix[periode,k]*np.ones((TimePeriode)))
		res[:,k] = np.transpose(vect)
	return res



# ------------------------------------------------------------------------------------------------------------------
#                                          Initialization of the Global Switching Thompson sampling
#-------------------------------------------------------------------------------------------------------------------

def GlobalSTS_Initialize(K, alpha0, beta0):
	if (alpha0 <=0 ):
		sys.exit('alpha0 must be > 0')
	if (beta0 <=0 ):
		sys.exit('beta0 must be > 0')
	alphas = np.ones((1,K))*alpha0
	betas = np.ones((1,K))*beta0
	ExpertWeigth = np.array([1])
	gainGlobalSTS = np.array([])	
	return alphas, betas, ExpertWeigth, gainGlobalSTS

#----------------------------------------------------------------------------------------------------------------
#                                            Arm recommended by the Global STS BA (2017)
#-----------------------------------------------------------------------------------------------------------------

def GlobalSTS_BA_RecommendArm(ExpertWeigth, alphas, betas):
	Size = alphas.shape
	Theta = np.zeros((Size[1]))
	for i in range(Size[1]):
		Theta[i] = np.dot(ExpertWeigth,np.random.beta(alphas[:,i],betas[:,i]))
	return np.argmax(Theta)


#----------------------------------------------------------------------------------------------------------------
#                                            Arm recommended by the Global STS (2013)
#-----------------------------------------------------------------------------------------------------------------


def GlobalSTS_RecommendArm(ExpertWeigth, alphas, betas):	
	ExpertWeigth = np.cumsum(ExpertWeigth)
	u = np.random.uniform()
	a= ExpertWeigth > u
	a = np.where(a == True)
	if np.size(a) == 0:
		Best = 0
	else:
		Best = a[0]
		Best = Best[0]
	Size = alphas.shape
	Theta = np.zeros((Size[1]))
	for i in range(Size[1]):
		Theta[i] = np.random.beta(alphas[Best,i],betas[Best,i])
	return np.argmax(Theta)

#-----------------------------------------------------------------------------------------------------------
#                    Correction of reward for non Bernoulli distribution 
#----------------------------------------------------------------------------------------------------------

def rewardCorrection(reward):
	if (reward < 0 or reward > 1 ):
		sys.exit('reward must be between 0 and 1')
	if (reward != 0  and reward != 1):
		reward = int((np.random.uniform() < reward)  == True) # for non-Bernoulli distribution
	return reward


#-------------------------------------------------------------------------------------------------------------
#               Expert distribution update (According to the message passing algorithm (Fernhead 2010)
#------------------------------------------------------------------------------------------------------------

def Global_STS_updateChangeModel(ExpertWeigth, alphas, betas, ArmToPlay, reward, gamma):
	NbrExperts = ExpertWeigth.size
	likelihood = np.zeros((NbrExperts))
	if reward == 1:
		for i in range(NbrExperts):
			likelihood[i] = alphas[i,ArmToPlay]/(alphas[i,ArmToPlay]+ betas[i,ArmToPlay])
	else:
		for i in range(NbrExperts):
			likelihood[i] = betas[i,ArmToPlay]/(alphas[i,ArmToPlay]+ betas[i,ArmToPlay])
	ExpertWeigth0 = gamma*np.dot(likelihood, np.transpose(ExpertWeigth)) # New expert creation
	ExpertWeigth = (1-gamma)*likelihood*ExpertWeigth
	ExpertWeigth = np.append(ExpertWeigth0,ExpertWeigth)
	ExpertWeigth = ExpertWeigth/np.sum(ExpertWeigth)
	return ExpertWeigth

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

def GlobalSTS_Resampling(ExpertWeigth, alphas, betas):
	indiceMin = np.argmin(ExpertWeigth)
	alphas = np.delete(alphas, indiceMin,0)
	betas = np.delete(betas, indiceMin,0)
	ExpertWeigth = np.delete(ExpertWeigth, indiceMin)
	return (ExpertWeigth, alphas, betas)





