#include "stdafx.h"
#include "FeedForwardWeights.h"

using namespace NeuralNetworkNative;

FeedForwardWeights::FeedForwardWeights(int inputLength, int outputLength) : BiasedWeights(inputLength * outputLength), InputNeurons(inputLength), OutputNeurons(outputLength)
{
	Weight = std::vector<double>(inputLength * outputLength);
	WeightStepSize = std::vector<double>(inputLength * outputLength);

	Weights::Randomise(Weight, InputNeurons);
	Weights::Clear(WeightStepSize);
}

void FeedForwardWeights::PropogateForwardCore(Step &downstream)
{
	assert(InputNeurons % downstream.Upstream.size() == 0);
	int neuronsPerUpstream = InputNeurons / downstream.Upstream.size();

	int inputIndex = 0;
	for (auto ui = downstream.Upstream.begin(), ue = downstream.Upstream.end(); ue != ui; ++ui) {
		Step &upstream = *(*ui);
		assert(inputIndex + upstream.Length <= InputNeurons);

		for (int i = 0; i < neuronsPerUpstream; i++)
		{
			PropogateForward(downstream, upstream, i, inputIndex++);
		}
	}
}

void FeedForwardWeights::PropogateForward(Step &downstream, Step &upstream, int upstreamNeuron, int inputNeuron)
{
	int weightIndex = inputNeuron * OutputNeurons;

	double upstreamNeuronOutput = upstream.Output[upstreamNeuron];
	double weightedSum = Bias;
	for (int o = 0; o < OutputNeurons; o++)
	{
		downstream.WeightedInputs[o] += upstreamNeuronOutput * Weight[weightIndex++];
	}
}

void FeedForwardWeights::StartPreTrainingCore()
{
	BiasedWeights::StartPreTrainingCore();
	Weights::Clear(WeightStepSize);
}

void FeedForwardWeights::PreTrainCore(Step &downstream)
{
	assert(InputNeurons % downstream.Upstream.size() == 0);
	int neuronsPerUpstream = InputNeurons / downstream.Upstream.size();

	int inputNeuron = 0;
	for (auto ui = downstream.Upstream.begin(), ue = downstream.Upstream.end(); ue != ui; ++ui) {
		Step &upstream = *(*ui);
		assert(upstream.Length == neuronsPerUpstream);

		for (int i = 0; i < neuronsPerUpstream; i++)
		{
			PropogateSecondDerivatives(downstream, upstream, i, inputNeuron++);
		}
	}
	EstimateBiasSecondDerivative(downstream);
}

void FeedForwardWeights::PropogateSecondDerivatives(Step &downstream, Step &upstream, int upstreamNeuron, int inputNeuron)
{
	int weightIndex = inputNeuron * OutputNeurons;

	double upstreamState = upstream.Output[upstreamNeuron];
	double upstreamErrorSecondDerivative = 0.0;

	for (int output = 0; output < OutputNeurons; output++)
	{
		double downstreamErrorSecondDerivative = downstream.ErrorDerivative[output]; // (d^2)E/(dAj)^2, where Aj is the sum of inputs to this downstream unit.

		// Here we calculate (d^2)Ej/(dWij)^2 by multiplying the 2nd derivative of E with respect to the sum of inputs, Aj
		// by the state of Oi, the upstream unit, squared. Refer to Equation 25 in document.
		// The summing happening here is described by equation 23.
		double weight2ndDerivative = downstreamErrorSecondDerivative * upstreamState * upstreamState;

		WeightStepSize[weightIndex] = weight2ndDerivative;

		double weight = Weight[weightIndex];

		// This is implementing the last sigma of Equation 27.
		// This propogates error second derivatives back to previous layer, but will need to be multiplied by the second derivative
		// of the activation function at the previous layer.
		upstreamErrorSecondDerivative += weight * weight * downstreamErrorSecondDerivative;

		weightIndex += 1;

	}

	upstream.ErrorDerivative[upstreamNeuron] += upstreamErrorSecondDerivative;
}

void FeedForwardWeights::EstimateBiasSecondDerivative(Step &downstream)
{
	for (int i = 0; i < downstream.Length; i++)
	{
		// Calculating the sum of: second derivatives of error with respect to the bias weight.
		// Note that the bias is implemented as an always-on Neuron with a (the same) weight to the outputs neurons.
		BiasStepSize += downstream.ErrorDerivative[i] * 1.0 * 1.0;
	}
}

void FeedForwardWeights::CompletePreTrainingCore()
{
	assert(preTrainingSamples > 0);
	double averageHkk = 0;

	double sampleCount = (double)preTrainingSamples;
	// Divide each of the 2nd derivative sums by the number of samples used to make the estimation, then  convert into step size.
	BiasStepSize = learningRate / (mu + (BiasStepSize / sampleCount));

	for (int i = 0; i < Size; i++)
	{
		double Hkk = WeightStepSize[i] / sampleCount;
		averageHkk += Hkk;
		WeightStepSize[i] = learningRate / (mu + Hkk);
	}

	averageHkk /= (double)(Size);

}

void FeedForwardWeights::TrainCore(Step &downstream)
{
	assert(InputNeurons % downstream.Upstream.size() == 0);
	int neuronsPerUpstream = InputNeurons / downstream.Upstream.size();

	int inputNeuron = 0;
	for (auto ui = downstream.Upstream.begin(), ue = downstream.Upstream.end(); ue != ui; ++ui) {
		Step &upstream = *(*ui);
		assert(upstream.Length == neuronsPerUpstream);
		for (int i = 0; i < neuronsPerUpstream; i++)
		{
			PropogateError(downstream, upstream, i, inputNeuron++);
		}
	}
}

void FeedForwardWeights::PropogateError(Step &downstream, Step &upstream, int upstreamNeuron, int inputNeuron)
{
	int weightIndex = inputNeuron * OutputNeurons;

	double upstreamState = upstream.Output[upstreamNeuron];

	double inputError = 0.0;
	for (int output = 0; output < OutputNeurons; output++)
	{
		double downstreamErrorDerivative = downstream.ErrorDerivative[output];

		// Calculate inputs error gradient by taking the sum, for all outputs of
		// dEk/dAj multiplied by dAj/dOj (w/sum =dEj/dOj);
		inputError += (downstreamErrorDerivative * Weight[weightIndex]);

		// Calculate the Weight's first derivative with respect to the error
		double weightErrorGradient = downstreamErrorDerivative * upstreamState;
		double deltaWeight = WeightStepSize[weightIndex] * weightErrorGradient;
		Weight[weightIndex] -= deltaWeight;

		weightIndex += 1;
	}
	upstream.ErrorDerivative[upstreamNeuron] = inputError;
}

FeedForwardWeights::~FeedForwardWeights()
{
}
