#pragma once

namespace NeuralNetworkCLI
{
	public ref class LeNetNetwork
	{
	private:
		NeuralNetworkNative::LeNetNetwork* nativeNetwork;

	public:
		LeNetNetwork(LeNetConfiguration^ configuration);

		property double LearningRate {
			double get();
			void set(double value);
		}
		property double Mu {
			double get();
			void set(double value);
		}
		property bool IsPreTraining {
			bool get();
			void set(bool value);
		}

		void PropogateForward(DataSetItem^ item);
		TrainingResults^ Train(DataSetItem^ item);

		static const int OutputFeedForwardNeurons = NeuralNetworkNative::LeNetNetwork::OutputFeedForwardNeurons;

		!LeNetNetwork();
		~LeNetNetwork();
	};

}