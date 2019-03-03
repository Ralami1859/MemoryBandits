import numpy as np
import numpy.matlib
import sys


# ------------------------------------------------------------------------------------------------------------------
#                                          Initialization of the Per Arm Switching Thompson sampling
#-------------------------------------------------------------------------------------------------------------------


def PerArmSTS_Initialize(K, alpha0, beta0):
	if (alpha0 <=0 ):
		sys.exit('alpha0 must be > 0')
	if (beta0 <=0 ):
		sys.exit('beta0 must be > 0')
	alphas = np.ones((1,K))*alpha0
	betas = np.ones((1,K))*beta0
	ExpertWeigth = np.ones((1,K))
	gainGlobalSTS = np.array([])	
	return alphas, betas, ExpertWeigth, gainGlobalSTS

#----------------------------------------------------------------------------------------------------------------
#                                            Arm recommended by the Per Arm STS BA (2017)
#-----------------------------------------------------------------------------------------------------------------

def PerArmSTS_BA_RecommendArm(ExpertWeigth, alphas, betas):
	
	Size = alphas.shape
	Theta = np.zeros((Size[1]))
	for i in range(Size[1]):
		BetaSamples = np.random.beta(alphas[:,i],betas[:,i]) # Sampling Beta Disctributions
		a = np.sum(ExpertWeigth[:,i]*BetaSamples)
		Theta[i] = a
	return np.argmax(Theta)


#----------------------------------------------------------------------------------------------------------------
#                                            Arm recommended by the Per Arm STS (2013)
#-----------------------------------------------------------------------------------------------------------------


def PerArmSTS_RecommendArm(ExpertWeigth, alphas, betas):	
	Size = ExpertWeigth.shape
	Theta = np.zeros((Size[1]))	
	for i in range(Size[1]): 	
		ExpertWeigths = np.cumsum(ExpertWeigth[:,i])
		u = np.random.uniform()
		a = ExpertWeigths > u
		a = np.where(a == True)
		if np.size(a) == 0:
			Best = 0
		else:
			Best = a[0]
			Best = Best[0]
		Theta[i] = np.random.beta(alphas[Best,i],betas[Best,i])
	return np.argmax(Theta)

#-------------------------------------------------------------------------------------------------------------
#               Expert distribution update (According to the message passing algorithm (Fernhead 2010)
#------------------------------------------------------------------------------------------------------------

def PerArm_STS_updateChangeModel(ExpertWeigth, alphas, betas, ArmToPlay, reward, gamma):	
	NbrExperts = np.size(ExpertWeigth,0)
	K = np.size(ExpertWeigth,1)
	likelihood = np.zeros((NbrExperts))
	if reward == 1:
		for i in range(NbrExperts):
			likelihood[i] = alphas[i,ArmToPlay]/(alphas[i,ArmToPlay]+ betas[i,ArmToPlay])
	else:
		for i in range(NbrExperts):
			likelihood[i] = betas[i,ArmToPlay]/(alphas[i,ArmToPlay]+ betas[i,ArmToPlay])

	if (np.size(gamma) !=1 and np.size(gamma) != K):
		sys.exit("Length of switching rate is incorrect. Must be a vector of length = 1, or length = Nbr Arms")
	if (np.size(gamma) == 1):
		gamma = np.ones((K))*gamma


	ExpertWeigth0 = np.zeros((K))	
	ExpertWeigth0[ArmToPlay] = gamma[ArmToPlay]*np.sum(np.transpose(likelihood)*ExpertWeigth[:,ArmToPlay]) # New expert creation
	ExpertWeigth[:,ArmToPlay] = (1-gamma[ArmToPlay])*(np.transpose(likelihood)*ExpertWeigth[:,ArmToPlay]).T

	for k in range(K):
		if k != ArmToPlay:
			ExpertWeigth0[k] = gamma[k]*np.sum(ExpertWeigth[:,k])
			ExpertWeigth[:,k] = (1-gamma[k])*ExpertWeigth[:,k]
			
	
	ExpertWeigth = np.vstack([ExpertWeigth0,ExpertWeigth])
	MatSum = np.matlib.repmat(np.sum(ExpertWeigth,0),NbrExperts+1,1)
	return ExpertWeigth/MatSum # Expert Distribution normalization

#-----------------------------------------------------------------------------------------------------------------
#               Discarding the worst expert to respect memory limitation
#----------------------------------------------------------------------------------------------------------------

def PerArmSTS_Resampling(ExpertWeigth, alphas, betas):

	indiceMin = np.argmin(ExpertWeigth,0)
	K = np.size(ExpertWeigth,1)
	NbrExperts = np.size(ExpertWeigth,0)
	NewExpertWeigth = np.zeros((NbrExperts-1,K))
	NewAlphas = np.zeros((NbrExperts-1,K))
	NewBetas = np.zeros((NbrExperts-1,K))
	for k in range(K):
		vect = ExpertWeigth[:,k]
		vect = np.delete(vect, indiceMin[k])
		NewExpertWeigth[:,k] = vect.T	
		
		vect = alphas[:,k]
		vect = np.delete(vect, indiceMin[k])
		NewAlphas[:,k] = vect.T

		vect = betas[:,k]
		vect = np.delete(vect, indiceMin[k])
		NewBetas[:,k] = vect.T
	return (NewExpertWeigth, NewAlphas, NewBetas)












