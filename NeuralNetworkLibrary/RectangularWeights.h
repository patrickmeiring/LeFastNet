#pragma once
#include "nnlib.h"
#include "RectangularStep.h"
#include "BiasedWeights.h"

namespace NeuralNetworkNative
{
	class RectangularWeights :
		public BiasedWeights
	{
	protected:
		virtual void PropogateForwardCore(Step &downstream) override final;
		virtual void TrainCore(Step &downstream) override final;
		virtual void PreTrainCore(Step &downstream) override final;

		virtual void PropogateForward(RectangularStep &downstream, int mapNumber) = 0;
		virtual void PropogateUnitSecondDerivatives(RectangularStep &downstream, int mapNumber) = 0;
		virtual void EstimateBiasSecondDerivative(RectangularStep &downstream) = 0;
		virtual void PropogateError(RectangularStep &downstream, int mapNumber) = 0;

	public:
		const int Width;
		const int Height;
		const int MapCount;

		RectangularWeights(int width, int height, int maps);
		virtual ~RectangularWeights();
	};
}