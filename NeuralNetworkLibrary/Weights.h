#pragma once
#include "nnlib.h"

namespace NeuralNetworkNative
{
	class Step;
	class Weights
	{
	private:
		static std::mt19937_64 random;

	protected:
		static void Randomise(std::vector<double> &weights, int fanIn);
		static double RandomWeight(int fanIn);
		static void Clear(std::vector<double> &vector);

		Weights(int size);
		int preTrainingSamples;

		virtual void PropogateForwardCore(Step &downstream) = 0;
		virtual void StartPreTrainingCore() = 0;
		virtual void PreTrainCore(Step &downstream) = 0;
		virtual void CompletePreTrainingCore() = 0;
		virtual void TrainCore(Step &downstream) = 0;

		double mu;
		double learningRate;

	public:
		double getLearningRate();
		void setLearningRate(double learningRate);
		double getMu();
		void setMu(double mu);

		const int Size;

		void StartPreTraining();
		virtual void PreTrain(Step &downstream);
		void CompletePreTraining();

		virtual void PropogateForward(Step &downstream);
		virtual void Train(Step &downstream);
		virtual ~Weights();
	};
}