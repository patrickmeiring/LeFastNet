#pragma once

namespace NeuralNetworkCLI
{
	public ref class RectangularStep :
		public Step
	{
	private:
		const NeuralNetworkNative::RectangularStep* nativeStep;

	protected:
		virtual const NeuralNetworkNative::Step* getNativeStep() override;

	internal:
		RectangularStep(LeNetNetwork^ network, const NeuralNetworkNative::RectangularStep* nativeStep);

	public:
		property int Width
		{
			int get();
		}
		property int Height
		{
			int get();
		}
	};

}