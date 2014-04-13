#include "stdafx.h"
#include "Step.h"

using namespace NeuralNetworkNative;

Step::Step(int length, const std::vector<Step*> &upstream, bool isFinalLayer) : Output(length), WeightedInputs(length), ErrorDerivative(length), Length(length), IsFinalLayer(isFinalLayer), Upstream(upstream)
{
	assert (length >= 0);
	
	isPreTraining = false;
	wasPreTraining = false;
	ClearError();
	ClearState();
}

void Step::ClearState()
{
	for (int i = 0; i < Length; i++)
	{
		WeightedInputs[i] = 0.0;
		Output[i] = 0.0;
	}
}

void Step::ClearError()
{
	for (int i = 0; i < Length; i++)
	{
		ErrorDerivative[i] = 0.0;
	}
}

void Step::PropogateForward()
{
	Weights* weights = getWeights();
	if (weights == nullptr) return;
	ClearState();
	weights->PropogateForward(*this);
}

void Step::PropogateBackwards()
{
	Weights* weights = getWeights();
	if (weights == nullptr) return;

	// Lazily start or stop the weights pre-training.
	LazySetPreTraining();

	if (isPreTraining)
	{
		weights->PreTrain(*this);
	}
	else
	{
		weights->Train(*this);
	}
	ClearError();
}

void Step::LazySetPreTraining()
{
	if (wasPreTraining == isPreTraining) return;

	Weights* weights = getWeights();
	if (weights == nullptr) return;
	if (isPreTraining)
	{
		weights->StartPreTraining();
	}
	else
	{
		weights->CompletePreTraining();
	}
	wasPreTraining = isPreTraining;
}

bool Step::getPreTraining()
{
	return isPreTraining;
}

void Step::setPreTraining(bool value)
{
	if (isPreTraining == value) return;
	isPreTraining = value;
}

const double X_STRETCH = 2.0 / 3.0;
const double Y_STRETCH = 1.7159;
const double DERIVATIVE_STRETCH = 4.57573;

double Step::CalculateActivation(double weightedInputs)
{
	double result = Y_STRETCH * tanh(X_STRETCH * weightedInputs);
	assert(!isnan(result));
	return result;
}

double Step::CalculateActivationDerivative(double weightedInputs)
{
	double coshx = cosh(X_STRETCH * weightedInputs);
	double denominator = cosh(2.0 * X_STRETCH * weightedInputs) + 1;
	double result = DERIVATIVE_STRETCH * coshx * coshx / (denominator * denominator);
	assert(!isnan(result));
	return result;
}

void Step::CopyOutputs(double* destination) const
{
	memcpy(destination, &Output[0], sizeof(double) * Length);
}

Step::~Step()
{

}