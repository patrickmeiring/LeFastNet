#pragma once
#include "RectangularWeights.h"

namespace NeuralNetworkNative
{
	class ConvolutionWeights :
		public RectangularWeights
	{
	private:
		double PropogateForward(RectangularStep &upstream, int downstreamX, int downstreamY, int mapNumber);
		void PropogateUnitSecondDerivatives(RectangularStep &upstream, RectangularStep &downstream, int weightX, int weightY, int mapNumber);
		void PropogateErrors(RectangularStep &upstream, RectangularStep &downstream, int weightX, int weightY, int mapNumber);

	protected:
		virtual void PropogateForward(RectangularStep &downstream, int mapNumber) override final;
		virtual void PropogateUnitSecondDerivatives(RectangularStep &downstream, int mapNumber) override final;
		virtual void EstimateBiasSecondDerivative(RectangularStep &downstream) override final;
		virtual void PropogateError(RectangularStep &downstream, int mapNumber) override final;
		virtual void StartPreTrainingCore() override final;
		virtual void CompletePreTrainingCore() override final;


		std::vector<double> Weight;
		std::vector<double> WeightStepSize;

	public:
		ConvolutionWeights(int convolutionWidth, int convolutionHeight, int mapCount);
		virtual ~ConvolutionWeights();
	};
}