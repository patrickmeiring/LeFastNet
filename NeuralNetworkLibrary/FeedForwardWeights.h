#pragma once
#include "BiasedWeights.h"

namespace NeuralNetworkNative
{
	class FeedForwardWeights :
		public BiasedWeights
	{
	protected:
		std::vector<double> Weight;
		std::vector<double> WeightStepSize;

		virtual void PropogateForwardCore(Step &downstream) override;
		void PropogateForward(Step &downstream, Step &upstream, int upstreamNeuron, int inputNeuron);
		virtual void StartPreTrainingCore() override;
		virtual void PreTrainCore(Step &downstream) override;
		void PropogateSecondDerivatives(Step &downstream, Step &upstream, int upstreamNeuron, int inputNeuron);
		void EstimateBiasSecondDerivative(Step &downstream);
		virtual void CompletePreTrainingCore() override;
		virtual void TrainCore(Step &downstream) override;
		void PropogateError(Step &downstream, Step &upstream, int upstreamNeuron, int inputNeuron);

	public:
		const int InputNeurons;
		const int OutputNeurons;
		FeedForwardWeights(int inputLength, int outputLength);
		virtual ~FeedForwardWeights();
	};
}