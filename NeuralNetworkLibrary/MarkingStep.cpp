#include "stdafx.h"
#include "MarkingStep.h"
#include "LeNetNetwork.h"

using namespace NeuralNetworkNative;

MarkingStep::MarkingStep(Step &upstream, LeNetConfiguration &configuration) : MarkingStep(FlatStep::SingleStepVector(upstream), configuration)
{
}

MarkingStep::MarkingStep(const std::vector<Step*> &upstream, LeNetConfiguration &configuration)
: FlatStep(configuration.ClassCount, upstream, true)
{
	assert(FlatStep::SizeOf(upstream) * upstream.size() == LeNetNetwork::OutputFeedForwardNeurons);
	weights = new MarkingWeights(configuration);
}

Weights* MarkingStep::getWeights()
{
	return weights;
}

int MarkingStep::getCorrectClass() {
	return weights->getCorrectClass();
}

void MarkingStep::setCorrectClass(int value) {
	weights->setCorrectClass(value);
}

MarkingStep::~MarkingStep()
{
	delete weights;
}
