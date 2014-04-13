#pragma once

namespace NeuralNetworkCLI
{
	ref class LeNetNetwork;
	public ref class Step abstract
	{
	private:
		LeNetNetwork^ network;
	protected:
		virtual const NeuralNetworkNative::Step* getNativeStep() abstract;
		void ThrowOnDisposed();

	public:
		Step(LeNetNetwork^ network);

		property int Length
		{
			int get();
		}
		property LeNetNetwork^ Network
		{
			LeNetNetwork^ get();
		}
		void CopyOutputs(array<double>^ destination);
	};

}