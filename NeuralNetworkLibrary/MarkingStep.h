#pragma once
#include "Step.h"
#include "MarkingWeights.h"

namespace NeuralNetworkNative
{
	class MarkingWeights;
	class MarkingStep :
		public FlatStep
	{
	private:
		MarkingWeights* weights;

	public:
		MarkingStep(Step &upstream, LeNetConfiguration &configuration);
		MarkingStep(const std::vector<Step*> &upstream, LeNetConfiguration &configuration);

		int getCorrectClass();
		void setCorrectClass(int value);

		virtual Weights* getWeights() override;

		virtual ~MarkingStep();

	};
}