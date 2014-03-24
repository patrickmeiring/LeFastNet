#include "stdafx.h"
#include "RectangularWeights.h"

using namespace NeuralNetworkNative;

RectangularWeights::RectangularWeights(int width, int height, int maps) : BiasedWeights(width * height * maps), Width(width), Height(height), MapCount(maps)
{
	assert(width > 0 && height > 0 && maps > 0);
}

void RectangularWeights::PropogateForwardCore(Step &downstream)
{
	RectangularStep& step = dynamic_cast<RectangularStep&>(downstream);
	assert(MapCount == step.Upstream.size());

	for (int i = 0; i < MapCount; i++)
	{
		PropogateForward(step, i);
	}
}

void RectangularWeights::TrainCore(Step &downstream)
{
	RectangularStep& step = dynamic_cast<RectangularStep&>(downstream);
	assert(MapCount == step.Upstream.size());

	for (int i = 0; i < MapCount; i++)
	{
		PropogateError(step, i);
	}
}

void RectangularWeights::PreTrainCore(Step &downstream)
{
	RectangularStep& step = dynamic_cast<RectangularStep&>(downstream);
	assert(MapCount == step.Upstream.size());

	for (int i = 0; i < MapCount; i++)
	{
		PropogateUnitSecondDerivatives(step, i);
	}
	EstimateBiasSecondDerivative(step);
}

RectangularWeights::~RectangularWeights()
{
}
