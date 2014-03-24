#pragma once
#include "nnlib.h"

namespace NeuralNetworkNative
{
	class Weights;
	class Step
	{
	public:
		std::vector<double> Output;
		std::vector<double> WeightedInputs;
		std::vector<double> ErrorDerivative;
		const int Length;
		const bool IsFinalLayer;
		std::vector<Step*> Upstream;

		void setPreTraining(bool value);
		bool getPreTraining();
		void PropogateForward();
		void PropogateBackwards();

		virtual Weights* getWeights() = 0;
		virtual double CalculateActivation(double weightedInputs);
		virtual double CalculateActivationDerivative(double weightedInputs);
		virtual ~Step();

	protected:
		Step(int length, const std::vector<Step*> &upstream,  bool isFinalLayer = false);
		void ClearState();
		void ClearError();

	private:
		bool isPreTraining;
	};
}