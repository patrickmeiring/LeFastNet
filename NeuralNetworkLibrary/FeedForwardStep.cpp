#include "stdafx.h"
#include "FeedForwardStep.h"

using namespace NeuralNetworkNative;

FeedForwardStep::FeedForwardStep(Step &upstream, int outputs) : FeedForwardStep(FlatStep::SingleStepVector(upstream), outputs)
{
}

FeedForwardStep::FeedForwardStep(const std::vector<Step*> &upstream, int outputs) 
  : FlatStep(outputs, upstream)
{
	weights = new FeedForwardWeights(FlatStep::SizeOf(upstream) * upstream.size(), outputs);
}

Weights* FeedForwardStep::getWeights()
{
	return weights;
}

FeedForwardStep::~FeedForwardStep()
{
	delete weights;
}
