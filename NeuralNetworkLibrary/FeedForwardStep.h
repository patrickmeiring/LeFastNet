#pragma once
#include "Step.h"
#include "FeedForwardWeights.h"

namespace NeuralNetworkNative
{
	class FeedForwardStep :
		public FlatStep
	{
	private:
		FeedForwardWeights* weights;

	public:
		FeedForwardStep(Step &upstream, int outputs);
		FeedForwardStep(const std::vector<Step*> &upstream, int outputs);

		virtual Weights* getWeights() override;

		virtual ~FeedForwardStep();
	};
}