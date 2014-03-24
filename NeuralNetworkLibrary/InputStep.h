#pragma once
#include "RectangularStep.h"

namespace NeuralNetworkNative
{
	class InputStep :
		public RectangularStep
	{
	public:
		InputStep(int width, int height);

		void setInputs(const double* inputs);

		virtual Weights* getWeights() override;

		virtual ~InputStep();
	};
}