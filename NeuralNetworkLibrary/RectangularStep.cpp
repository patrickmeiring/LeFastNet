#include "stdafx.h"
#include "RectangularStep.h"

using namespace NeuralNetworkNative;

RectangularStep::RectangularStep(int width, int height, const std::vector<RectangularStep*> &upstream) 
: Step(width * height, RectangularStep::ToStepVector(upstream)),
	Width(width), 
	Height(height)
{
	assert(width > 0 && height > 0);
	Upstream = upstream;
}

int RectangularStep::WidthOf(const std::vector<RectangularStep*> &upstream)
{
	assert(upstream.size() > 0);
	int width = upstream[0]->Width;
	for (auto vi = upstream.begin(), ve = upstream.end(); vi != ve; ++vi) {
		assert (width == (*vi)->Width);
	}
	return width;
}

int RectangularStep::HeightOf(const std::vector<RectangularStep*> &upstream)
{
	assert(upstream.size() > 0);

	int height = upstream[0]->Height;
	for (auto vi = upstream.begin(), ve = upstream.end(); vi != ve; ++vi) {
		assert (height == (*vi)->Height);
	}
	return height;
}

std::vector<RectangularStep*> RectangularStep::SingleRectangularStepVector(RectangularStep &step)
{
	std::vector<RectangularStep*> result = std::vector<RectangularStep*>(1);
	result[0] = &step;
	return result;
}

std::vector<Step*> RectangularStep::ToStepVector(const std::vector<RectangularStep*> &rectangularSteps)
{
	std::vector<Step*> result;
	for (auto si = rectangularSteps.begin(), se = rectangularSteps.end(); si != se; ++si)
	{
		result.push_back(*si);
	}
	return result;
}

RectangularStep::~RectangularStep()
{
}
