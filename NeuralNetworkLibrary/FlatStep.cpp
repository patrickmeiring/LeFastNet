#include "stdafx.h"
#include "FlatStep.h"

using namespace NeuralNetworkNative;

FlatStep::FlatStep(int length, const std::vector<Step*> &upstream, bool isFinal) : Step(length, upstream, isFinal)
{
	Upstream = upstream;
}

int FlatStep::SizeOf(const std::vector<Step*>& upstream)
{
	assert(upstream.size() != 0);

	int length = upstream[0]->Length;
	for (auto vi = upstream.begin(), ve = upstream.end(); vi != ve; ++vi) {
		assert((*vi)->Length == length);
	}
	return length;
}


std::vector<Step*> FlatStep::SingleStepVector(Step &step)
{
	std::vector<Step*> result(1, &step);
	return result;
}

FlatStep::~FlatStep()
{
}
