#include "stdafx.h"
#include "SubsamplingWeights.h"

using namespace NeuralNetworkNative;

SubsamplingWeights::SubsamplingWeights(int width, int height) : RectangularWeights(width, height, 1)
{
	Weight = Weights::RandomWeight(width * height);
	WeightStepSize = 0.0;
}

void SubsamplingWeights::PropogateForward(RectangularStep &downstream, int mapNumber)
{
	RectangularStep& upstream = *downstream.Upstream[mapNumber];
	int index = 0;
	for (int y = 0; y < downstream.Height; y++)
	{
		for (int x = 0; x < downstream.Width; x++)
		{
			downstream.WeightedInputs[index++] += PropogateForward(upstream, x * Width, y * Height, mapNumber);
		}
	}
}

double SubsamplingWeights::PropogateForward(RectangularStep &upstream, int upstreamX, int upstreamY, int mapNumber)
{
	assert(upstreamX + Width <= upstream.Width);   // Check we are staying within the width limit of the step.
	assert(upstreamY + Height <= upstream.Height);  // Check we are staying within the height limit of the step.

	double result = Bias;

	int upstreamIndex = (upstreamY * upstream.Width) + upstreamX;
	for (int y = 0; y < Height; y++)
	{
		for (int x = 0; x < Width; x++)
		{
			result += upstream.Output[upstreamIndex] * Weight;
			upstreamIndex += 1;
		}
		upstreamIndex += upstream.Width - Width;
	}
	return result;
}

void SubsamplingWeights::StartPreTrainingCore()
{
	RectangularWeights::StartPreTrainingCore();
	WeightStepSize = 0.0;
}

void SubsamplingWeights::PropogateUnitSecondDerivatives(RectangularStep &downstream, int upstreamIndex)
{
	RectangularStep &upstream = *downstream.Upstream[upstreamIndex];

	for (int y = 0; y < downstream.Height; y++)
	{
		for (int x = 0; x < downstream.Width; x++)
		{
			EstimateWeightSecondDerivative(upstream, downstream, x, y);
		}
	}
}

void SubsamplingWeights::EstimateWeightSecondDerivative(RectangularStep &upstream, RectangularStep &downstream, int downstreamX, int downstreamY)
{
	double weight2ndDerivative = 0;

	int downstreamIndex = downstreamY * downstream.Width + downstreamX;
	int upstreamIndex = (downstreamY * Height * upstream.Width) + downstreamX * Width;

	double downstreamSecondDerivative = downstream.ErrorDerivative[downstreamIndex];
	double upstreamSecondDerivative = Weight * Weight * downstreamSecondDerivative;

	// This loop here is equivalent to the sigma in equation 19 in Gradient-Based Learning Applied to Document Recognition.
	for (int y = 0; y < Height; y++)
	{
		for (int x = 0; x < Width; x++)
		{
			double upstreamState = upstream.Output[upstreamIndex];

			// Here we calculate (d^2)Ej/(dWij)^2 by multiplying the 2nd derivative of E with respect to the sum of inputs, Ai
			// by the state of Oi, the upstream unit, squared. Refer to Equation 25 in document.
			// The summing happening here is described by equation 23.
			weight2ndDerivative += downstreamSecondDerivative * upstreamState * upstreamState;

			// This is implementing the last sigma of Equation 27.
			// This propogates error second derivatives back to previous layer, but will need to be multiplied by the second derivative
			// of the activation function at the previous layer.
			upstream.ErrorDerivative[upstreamIndex] = upstreamSecondDerivative;

			upstreamIndex += 1;
		}
		upstreamIndex += upstream.Width - Width;
	}

	WeightStepSize += weight2ndDerivative;
}

void SubsamplingWeights::EstimateBiasSecondDerivative(RectangularStep &downstream)
{
	for (int i = 0; i < downstream.Length; i++)
	{
		// Calculating the sum of: second derivatives of error with respect to the bias weight.
		// Note that the bias is implemented as an always-on Neuron with a (the same) weight to the outputs neurons.
		BiasStepSize += downstream.ErrorDerivative[i] * 1.0 * 1.0;
	}
}

void SubsamplingWeights::CompletePreTrainingCore()
{
	double sampleCount = (double)preTrainingSamples;
	// Divide each of the 2nd derivative sums by the number of samples used to make the estimation, then  convert into step size.

	BiasStepSize /= sampleCount;
	WeightStepSize /= sampleCount;

	BiasStepSize = learningRate / (mu + BiasStepSize);
	WeightStepSize = learningRate / (mu + WeightStepSize);
}

void SubsamplingWeights::PropogateError(RectangularStep &downstream, int mapNumber)
{
	RectangularStep &upstream = *downstream.Upstream[mapNumber];

	for (int y = 0; y < downstream.Height; y++)
	{
		for (int x = 0; x < downstream.Width; x++)
		{
			PropogateError(downstream, upstream, x, y);
		}
	}
}

void SubsamplingWeights::PropogateError(RectangularStep &downstream, RectangularStep &upstream, int downstreamX, int downstreamY)
{
	int downstreamIndex = downstreamY * downstream.Width + downstreamX;
	int upstreamIndex = (downstreamY * Height * upstream.Width) + downstreamX * Width;

	double downstreamErrorDerivative = downstream.ErrorDerivative[downstreamIndex];
	double upstreamError = Weight * downstreamErrorDerivative;
	double weightError = 0.0;

	for (int y = 0; y < Height; y++)
	{
		for (int x = 0; x < Width; x++)
		{
			upstream.ErrorDerivative[upstreamIndex] = upstreamError;

			double weightErrorGradient = downstreamErrorDerivative * upstream.Output[upstreamIndex];
			weightError += weightErrorGradient;

			upstreamIndex += 1;
		}
		upstreamIndex += upstream.Width - Width;
	}
	Weight -= weightError * WeightStepSize;
}


SubsamplingWeights::~SubsamplingWeights()
{
}
