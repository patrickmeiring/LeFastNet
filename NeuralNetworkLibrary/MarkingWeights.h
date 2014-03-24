#pragma once
#include "Weights.h"
#include "LeNetConfiguration.h"

namespace NeuralNetworkNative
{
	class MarkingWeights :
		public Weights
	{
	private:
		int correctClass;
		void PropogateForward(Step &downstream, int output);

	protected:
		virtual void StartPreTrainingCore() override final;
		virtual void CompletePreTrainingCore() override final;
		virtual void PreTrainCore(Step &downstream) override final;
		virtual void PropogateForwardCore(Step &downstream) override final;
		virtual void TrainCore(Step &downstream) override final;

	public:
		const int ClassCount;
		const int InputLength;
		const std::vector<double> ClassStateDefinitions;

		int getCorrectClass();
		void setCorrectClass(int value);

		MarkingWeights(LeNetConfiguration& configuration);
		virtual ~MarkingWeights();
	};
}