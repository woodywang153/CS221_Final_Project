import numpy as np
import csv
import pandas as pd
import random
import math
from scipy import spatial

indexUsers = 47
indexRestaurants = 9
numToPredict = 50
numEpochs = 1000.0

def read_data():
	"""Reads the ratings matrix from file"""
	# This matrix has the following shape: num_movies x num_users
	# The values stored in each row i and column j is the rating for
	# movie i by user j
	#userMatrix = np.asarray(pd.read_csv("surveytruthsmod.csv", header=0))
	#userMatrix = np.transpose(userMatrix)
	dataMatrix = pd.read_csv("groundtruthsmod.csv", header=0)
	userMatrix = np.zeros(shape=(indexRestaurants+1, indexUsers+1))

	predictedDataMatrix = pd.read_csv("predictionStanford.csv", header=0)
	predictedUserMatrix = np.zeros(shape=(indexRestaurants+1, indexUsers+1))

	for user in range(0, indexUsers+1):
		for restaurant in range(0, indexRestaurants+1):
			if ((dataMatrix['restaurant_id'] == restaurant) & (dataMatrix['user_id'] == user)).any():
				framerow = dataMatrix.loc[(dataMatrix['restaurant_id'] == restaurant) & (dataMatrix['user_id'] == user)]
				userMatrix[restaurant][user] = framerow['stars']
			if ((predictedDataMatrix['restaurant_id'] == restaurant) & (predictedDataMatrix['user_id'] == user)).any():
				framerow = predictedDataMatrix.loc[(predictedDataMatrix['restaurant_id'] == restaurant) & (predictedDataMatrix['user_id'] == user)]
				predictedUserMatrix[restaurant][user] = framerow['pred']
	#print userMatrix, predictedUserMatrix
	return userMatrix, predictedUserMatrix

def randomlyRemove(userMatrix, predictedUserMatrix):
	modifiedMatrix = np.copy(userMatrix)
	predictedModifiedMatrix = np.copy(predictedUserMatrix)
	rows = modifiedMatrix.shape[0]
	cols = modifiedMatrix.shape[1]
	for numIters in range(numToPredict):
		randomrow = random.randint(0, rows-1)
		randomcol = random.randint(0, cols-1)
		modifiedMatrix[randomrow][randomcol] = 0
		predictedModifiedMatrix[randomrow][randomcol] = 0
		#remove from predictedMatrix
	return modifiedMatrix, predictedModifiedMatrix


def distance(u, v):
	uNorm = np.linalg.norm(u)
	vNorm = np.linalg.norm(v)
	if uNorm == 0 or vNorm == 0:
	  return 0
	return np.dot(u,v) / (uNorm * vNorm)

def cfilter(modifiedMatrix, userMatrix, trueUserMatrix):
	simScores = np.zeros((modifiedMatrix.shape[0], modifiedMatrix.shape[0]))


	for i in range(simScores.shape[0]):
		for j in range(simScores.shape[0]):
			#print userMatrix[i], userMatrix[j]
			simScores[i][j] = distance(modifiedMatrix[i], modifiedMatrix[j])
	totalExampleCount = 0.0
	totalWrongCount = 0.0
	totalLoss = 0.0
	MAP = 0.0
	for userIndex, userVector in enumerate(modifiedMatrix.T):
		currUserVector = userVector.T
		filledInUserVector = np.copy(currUserVector)

		for i in range(len(currUserVector)):
			if currUserVector[i]==0:
				#get simscores vector for i
				#then pick the top 3 that are the most similar that we have reviews for
				#and take the weighted average
				simVector = simScores[i]
				indices = np.argpartition(simVector, -3)[-3:]
				index = np.argwhere(indices==i)
				indices = np.delete(indices, index)
				curr_rating = 0
				normalization_factor = np.exp(simVector[indices]).sum()
				for index in indices:
					curr_rating += ((np.exp(simVector[index])*userVector[index])/normalization_factor)
					#
				filledInUserVector[i] = curr_rating
				if userMatrix[i][userIndex] != 0:
					roundedPrediction = math.ceil(userMatrix[i][userIndex])**2
					totalExampleCount += 1
					totalLoss += ((curr_rating - math.ceil(userMatrix[i][userIndex]))**2)
					if userMatrix[i][userIndex] != math.ceil(curr_rating):
						totalWrongCount += 1

		correctTopThree = np.argpartition(trueUserMatrix.T[userIndex], -3)[-3:]
		predictedTopThree = np.argpartition(filledInUserVector, -3)[-3:]
		#print correctTopThree, predictedTopThree
		averagePrec = 0.0
		numCorrect = 0.0
		for i in range(len(correctTopThree)):
			if predictedTopThree[i] in correctTopThree:
				numCorrect += 1
				averagePrec += numCorrect/(i+1) * (1.0/3)
		MAP += averagePrec
			
	return (totalExampleCount-totalWrongCount)/totalExampleCount, MAP/indexUsers

def main():
	userMatrix, predictedUserMatrix = read_data()
	averagediff=0
	averageMAP1=0
	averageMAP2=0
	for i in range(int(numEpochs)):
		modifiedMatrix, predictedModifiedMatrix = randomlyRemove(userMatrix, predictedUserMatrix)
		prec1, MAP1 = cfilter(modifiedMatrix, userMatrix, userMatrix)
		prec2, MAP2 = cfilter(predictedModifiedMatrix, predictedUserMatrix, userMatrix)
		averageMAP1 += MAP1
		averageMAP2 += MAP2
		averagediff += (prec2-prec1)
	print averagediff/numEpochs, averageMAP1/numEpochs, averageMAP2/numEpochs


if __name__== "__main__":
	main()