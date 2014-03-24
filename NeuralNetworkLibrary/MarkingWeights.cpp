#include "stdafx.h"
#include "MarkingWeights.h"
#include "LeNetNetwork.h"

using namespace NeuralNetworkNative;

MarkingWeights::MarkingWeights(LeNetConfiguration &configuration) : Weights(configuration.ClassCount * LeNetNetwork::OutputFeedForwardNeurons),
ClassCount(configuration.ClassCount),
InputLength(LeNetNetwork::OutputFeedForwardNeurons),
ClassStateDefinitions(configuration.ClassDefinitions, configuration.ClassDefinitions + (configuration.ClassCount * LeNetNetwork::OutputFeedForwardNeurons))
{
}

void MarkingWeights::StartPreTrainingCore()
{
}

void MarkingWeights::PreTrainCore(Step &downstream)
{
	int inputIndex = 0;
	for (auto ui = downstream.Upstream.begin(), ue = downstream.Upstream.end(); ui != ue; ++ui)
	{
		Step &upstream = *(*ui);
		assert(inputIndex + upstream.Length <= InputLength);
		for (int i = 0; i < upstream.Length; i++)
		{
			// Error second derivative relative to output is constant, as first derivative is 2.0 * (state - desiredState).
			upstream.ErrorDerivative[inputIndex] = 2.0;

			inputIndex += 1;
		}
	}
}

void MarkingWeights::CompletePreTrainingCore()
{
}

void MarkingWeights::PropogateForwardCore(Step &downstream)
{
	assert(downstream.Upstream.size() == 1);

	for (int o = 0; o < ClassCount; o++)
	{
		PropogateForward(downstream, o);
	}
}

void MarkingWeights::PropogateForward(Step &downstream, int output)
{
	double sumSquaredError = 0;
	int inputIndex = 0;
	int definitionIndex = output * InputLength;
	for (auto ui = downstream.Upstream.begin(), ue = downstream.Upstream.end(); ui != ue; ++ui)
	{
		Step &upstream = *(*ui);
		assert(inputIndex + upstream.Length <= InputLength);
		for (int i = 0; i < upstream.Length; i++)
		{
			double difference = upstream.Output[i] - ClassStateDefinitions[definitionIndex];
			sumSquaredError += difference * difference;

			inputIndex += 1;
			definitionIndex += 1;
		}
	}
	downstream.Output[output] = sumSquaredError;
}

void MarkingWeights::TrainCore(Step &downstream)
{
	int inputIndex = 0;
	int definitionIndex = correctClass * InputLength;
	for (auto ui = downstream.Upstream.begin(), ue = downstream.Upstream.end(); ui != ue; ++ui)
	{
		Step &upstream = *(*ui);
		assert(inputIndex + upstream.Length <= InputLength);
		for (int i = 0; i < upstream.Length; i++)
		{
			double desiredState = ClassStateDefinitions[definitionIndex];

			double firstDerivative = (upstream.Output[inputIndex] - desiredState) *2.0;
			upstream.ErrorDerivative[inputIndex] = firstDerivative;

			inputIndex += 1;
			definitionIndex += 1;
		}
	}
}

int MarkingWeights::getCorrectClass()
{
	return correctClass;
}

void MarkingWeights::setCorrectClass(int value)
{
	correctClass = value;
}

MarkingWeights::~MarkingWeights()
{
}
