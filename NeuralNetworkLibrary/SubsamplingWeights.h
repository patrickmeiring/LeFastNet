#pragma once
#include "RectangularWeights.h"

namespace NeuralNetworkNative
{
	class SubsamplingWeights :
		public RectangularWeights
	{
	private:
		double PropogateForward(RectangularStep& upstream, int upstreamX, int upstreamY, int mapNumber);
		void EstimateWeightSecondDerivative(RectangularStep& upstream, RectangularStep& downstream, int downstreamX, int downstreamY);
		void PropogateError(RectangularStep& downstream, RectangularStep& upstream, int downstreamX, int downstreamY);

	protected:
		double Weight;
		double WeightStepSize;

		virtual void PropogateForward(RectangularStep& downstream, int mapNumber) override final;

		virtual void StartPreTrainingCore() override final;
		virtual void PropogateUnitSecondDerivatives(RectangularStep& downstream, int upstreamIndex) override final;
		virtual void EstimateBiasSecondDerivative(RectangularStep& downstream) override final;
		virtual void CompletePreTrainingCore() override final;
		virtual void PropogateError(RectangularStep& downstream, int mapNumber) override final;

	public:
		SubsamplingWeights(int width, int height);
		virtual ~SubsamplingWeights();
	};
}