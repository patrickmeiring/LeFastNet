#include "stdafx.h"
#include "ConvolutionStep.h"

using namespace NeuralNetworkNative;

ConvolutionStep::ConvolutionStep(RectangularStep &upstream, int convolutionSize) : ConvolutionStep(upstream, convolutionSize, convolutionSize)
{

}

ConvolutionStep::ConvolutionStep(const std::vector<RectangularStep*> &upstream, int convolutionSize) : ConvolutionStep(upstream, convolutionSize, convolutionSize)
{

}

ConvolutionStep::ConvolutionStep(RectangularStep &upstream, int convolutionWidth, int convolutionHeight) 
	: ConvolutionStep(RectangularStep::SingleRectangularStepVector(upstream), convolutionWidth, convolutionHeight)
{

}

ConvolutionStep::ConvolutionStep(const std::vector<RectangularStep*> &upstream, int convolutionWidth, int convolutionHeight)
	: RectangularStep(RectangularStep::WidthOf(upstream) - convolutionWidth + 1, RectangularStep::HeightOf(upstream) - convolutionHeight + 1, upstream)
{
	weights = new ConvolutionWeights(convolutionWidth, convolutionHeight, upstream.size());
}

Weights* ConvolutionStep::getWeights()
{
	return weights;
}

ConvolutionStep::~ConvolutionStep()
{
	delete weights;
}
