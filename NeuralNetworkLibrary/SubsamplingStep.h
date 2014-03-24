#pragma once
#include "RectangularStep.h"
#include "SubsamplingWeights.h"

namespace NeuralNetworkNative
{
	class SubsamplingStep :
		public RectangularStep
	{
	private:
		SubsamplingWeights* weights;

	public:
		SubsamplingStep(RectangularStep &upstream, int subsamplingSize);
		SubsamplingStep(const std::vector<RectangularStep*> &upstream, int subsamplingSize);
		SubsamplingStep(RectangularStep &upstream, int subsamplingWidth, int subsamplingHeight);
		SubsamplingStep(const std::vector<RectangularStep*> &upstream, int subsamplingWidth, int subsamplingHeight);

		virtual Weights* getWeights() override;

		virtual ~SubsamplingStep();
	};
}