#pragma once

namespace NeuralNetworkCLI
{
	public ref class LeNetNetwork
	{
	private:
		NeuralNetworkNative::LeNetNetwork* nativeNetwork;

		RectangularStep^ inputStep;
		System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ firstConvolutions;
		System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ firstSubsampling;
		System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ secondConvolutions;
		System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ secondSubsampling;
		FlatStep^ consolidationStep;
		FlatStep^ outputStep;

		void CreateSteps();
		void CreateInputStep();
		void CreateFirstConvolutions();
		void CreateFirstSubsampling();
		void CreateSecondConvolutions();
		void CreateSecondSubsampling();
		void CreateConsolidationAndOutputSteps();

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
		property bool IsDisposed {
			bool get();
		}
		property RectangularStep^ InputStep
		{
			RectangularStep^ get();
		}
		property System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ FirstConvolutions
		{
			System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ get();
		}
		property System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ FirstSubsampling
		{
			System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ get();
		}
		property System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ SecondConvolutions
		{
			System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ get();
		}
		property System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ SecondSubsampling
		{
			System::Collections::ObjectModel::ReadOnlyCollection<RectangularStep^>^ get();
		}
		property FlatStep^ ConsolidationStep
		{
			FlatStep^ get();
		}
		property FlatStep^ OutputStep
		{
			FlatStep^ get();
		}

		void PropogateForward(DataSetItem^ item);
		TrainingResults^ Train(DataSetItem^ item);

		static const int OutputFeedForwardNeurons = NeuralNetworkNative::LeNetNetwork::OutputFeedForwardNeurons;

		!LeNetNetwork();
		~LeNetNetwork();
	};

}