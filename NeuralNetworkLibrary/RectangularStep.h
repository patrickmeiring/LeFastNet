#pragma once
#include "Step.h"

namespace NeuralNetworkNative
{
	class RectangularStep :
		public Step
	{
	protected:
		static int WidthOf(const std::vector<RectangularStep*> &upstream);
		static int HeightOf(const std::vector<RectangularStep*> &upstream);
		static std::vector<RectangularStep*> SingleRectangularStepVector(RectangularStep &step);
		static std::vector<Step*> ToStepVector(const std::vector<RectangularStep*> &rectangularSteps);

	public:
		RectangularStep(int width, int height, const std::vector<RectangularStep*> &upstream);
		virtual ~RectangularStep();
		const int Width;
		const int Height;


		std::vector<RectangularStep*> Upstream;
	};
}

