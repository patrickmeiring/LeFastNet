#pragma once
#include "RectangularStep.h"
#include "ConvolutionWeights.h"
#include <vector>

namespace NeuralNetworkNative
{
	class ConvolutionStep :
		public RectangularStep
	{
	private:
		ConvolutionWeights* weights;

	public:
		ConvolutionStep(RectangularStep &upstream, int convolutionSize);
		ConvolutionStep(const std::vector<RectangularStep*> &upstream, int convolutionSize);
		ConvolutionStep(RectangularStep &upstream, int convolutionWidth, int convolutionHeight);
		ConvolutionStep(const std::vector<RectangularStep*> &upstream, int convolutionWidth, int convolutionHeight);

		virtual Weights* getWeights() override;

		virtual ~ConvolutionStep();
	};
}