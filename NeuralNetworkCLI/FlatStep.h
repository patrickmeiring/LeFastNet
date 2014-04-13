#pragma once

namespace NeuralNetworkCLI
{
	public ref class FlatStep :
		public Step
	{
	private:
		const NeuralNetworkNative::FlatStep* nativeStep;
	protected:
		virtual const NeuralNetworkNative::Step* getNativeStep() override;

	internal:
		FlatStep(LeNetNetwork^ network, const NeuralNetworkNative::FlatStep* nativeStep);

	};

}