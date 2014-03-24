#include "stdafx.h"
#include "SubsamplingStep.h"

using namespace NeuralNetworkNative;

SubsamplingStep::SubsamplingStep(RectangularStep &upstream, int subsamplingSize) : SubsamplingStep(upstream, subsamplingSize, subsamplingSize)
{

}

SubsamplingStep::SubsamplingStep(const std::vector<RectangularStep*> &upstream, int subsamplingSize) : SubsamplingStep(upstream, subsamplingSize, subsamplingSize)
{

}

SubsamplingStep::SubsamplingStep(RectangularStep &upstream, int subsamplingWidth, int subsamplingHeight)
: SubsamplingStep(RectangularStep::SingleRectangularStepVector(upstream), subsamplingWidth, subsamplingHeight)
{

}

SubsamplingStep::SubsamplingStep(const std::vector<RectangularStep*> &upstream, int subsamplingWidth, int subsamplingHeight)
: RectangularStep(RectangularStep::WidthOf(upstream) / subsamplingWidth, RectangularStep::HeightOf(upstream) / subsamplingHeight, upstream)
{
	assert(RectangularStep::WidthOf(upstream) % subsamplingWidth == 0);
	assert(RectangularStep::HeightOf(upstream) % subsamplingHeight == 0);
	weights = new SubsamplingWeights(subsamplingWidth, subsamplingHeight);
}

Weights* SubsamplingStep::getWeights()
{
	return weights;
}

SubsamplingStep::~SubsamplingStep()
{
	delete weights;
}
