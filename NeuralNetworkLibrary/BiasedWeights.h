#pragma once
#include "Weights.h"

namespace NeuralNetworkNative
{
	class BiasedWeights :
		public Weights
	{
	private:
		void FinaliseErrorFirstDerivatives(Step &downstream);
		void FinaliseErrorSecondDerivatives(Step &downstream);
		void FinaliseOutputs(Step &downstream);

	protected:
		double Bias;
		double BiasStepSize;

		virtual void Train(Step &downstream) override final;
		virtual void PreTrain(Step &downstream) override final;
		virtual void StartPreTrainingCore() override;

	public:
		BiasedWeights(int size);
		virtual ~BiasedWeights();
		virtual void PropogateForward(Step &downstream) override final;
	};
}