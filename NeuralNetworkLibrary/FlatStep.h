#pragma once
#include "Step.h"

namespace NeuralNetworkNative
{
	class FlatStep :
		public Step
	{
	public:
		std::vector<Step*> Upstream;
		static int SizeOf(const std::vector<Step*>& upstream);
		static std::vector<Step*> SingleStepVector(Step &step);



		FlatStep(int length, const std::vector<Step*> &upstream, bool isFinal = false);
		virtual ~FlatStep();
	};
}