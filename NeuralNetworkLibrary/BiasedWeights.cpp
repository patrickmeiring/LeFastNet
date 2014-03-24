#include "stdafx.h"
#include "BiasedWeights.h"


using namespace NeuralNetworkNative;

BiasedWeights::BiasedWeights(int size) : Weights(size)
{
	Bias = Weights::RandomWeight(Size);
	BiasStepSize = 0.0;
}

void BiasedWeights::StartPreTrainingCore()
{
	BiasStepSize = 0;
}

void BiasedWeights::PreTrain(Step &downstream)
{
	FinaliseErrorSecondDerivatives(downstream);
	Weights::PreTrain(downstream);
}

void BiasedWeights::Train(Step &downstream)
{
	FinaliseErrorFirstDerivatives(downstream);
	Weights::Train(downstream);
}

void BiasedWeights::FinaliseErrorFirstDerivatives(Step &downstream)
{
	// Calculating the dEj/dWij and dEi/dOi both requires a multiplication by the derivative of the activation function,
	// it is done here once so it doesn't need to be done for each individual calculations.

	// This turns dEk/dOk into dEk/dAk by multiplying it by dOk/dAk
	for (int i = 0; i < downstream.Length; i++)
	{
		double weightedInputs = downstream.WeightedInputs[i];
		double activationDerivative = downstream.CalculateActivationDerivative(weightedInputs);
		downstream.ErrorDerivative[i] *= activationDerivative;
	}
}

void BiasedWeights::FinaliseErrorSecondDerivatives(Step &downstream)
{
	for (int i = 0; i < downstream.Length; i++)
	{
		double weightedInputs = downstream.WeightedInputs[i];
		double activationDerivative = downstream.CalculateActivationDerivative(weightedInputs);
		downstream.ErrorDerivative[i] *= activationDerivative * activationDerivative;
	}
}

void BiasedWeights::PropogateForward(Step &downstream)
{
	Weights::PropogateForward(downstream);
	FinaliseOutputs(downstream);
}


void BiasedWeights::FinaliseOutputs(Step &downstream)
{
	for (int i = 0; i < downstream.Length; i++)
	{
		downstream.Output[i] = downstream.CalculateActivation(downstream.WeightedInputs[i]);
	}
}

BiasedWeights::~BiasedWeights()
{
}
