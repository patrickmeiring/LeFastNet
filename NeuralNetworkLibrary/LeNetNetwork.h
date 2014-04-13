#pragma once
#include "LeNetConfiguration.h"
#include "nnlib.h"
#include "InputStep.h"
#include "ConvolutionStep.h"
#include "SubsamplingStep.h"
#include "FeedForwardStep.h"
#include "MarkingStep.h"
#include "DataSetItem.h"
#include "TrainingResults.h"


namespace NeuralNetworkNative
{
	class __LENETLIB_DLLEXPORT LeNetNetwork
	{
	private:
		void CreateNetwork();
		
		void CreateInputStep();
		void CreateFirstConvolutionStep();
		void CreateFirstSubsamplingStep();
		void CreateSecondConvolutionStep();
		void CreateSecondSubsamplingStep();
		void CreateConsolidationAndOutputSteps();

		bool preTraining;

		int classIndexOf(wchar_t character);

	protected:
		InputStep* inputLayer;
		std::vector<ConvolutionStep*> firstConvolutions;
		std::vector<SubsamplingStep*> firstSubsampling;
		std::vector<ConvolutionStep*> secondConvolutions;
		std::vector<SubsamplingStep*> secondSubsampling;
		FeedForwardStep* consolidation;
		FeedForwardStep* output;
		MarkingStep* marking;

		std::vector<Step*> allSteps;
		LeNetConfiguration configuration;

		double learningRate;
		double mu;

	public:
		static const int FirstConvolutionCount;
		static const int FirstConvolutionSize;
		static const int SecondConvolutionCount;
		static const int SecondConvolutionSize;

		static const int OutputFeedForwardNeurons;
		static const int ConsolidationNeurons;
		static const std::vector<bool> SecondConvolutionConnections;

		double getLearningRate();
		void setLearningRate(double learningRate);
		double getMu();
		void setMu(double mu);


		bool isPreTraining();
		void setPreTraining(bool value);
		void PropogateForward(DataSetItem &inputs);
		TrainingResults Train(DataSetItem &inputs);

		const InputStep* getInputStep();
		const ConvolutionStep *const* getFirstConvolutions();
		const SubsamplingStep *const* getFirstSubsampling();
		const ConvolutionStep *const* getSecondConvolutions();
		const SubsamplingStep *const* getSecondSubsampling();
		const FeedForwardStep * getConsolidationStep();
		const FeedForwardStep * getOutputStep();

		LeNetNetwork(const LeNetConfiguration &configuration);
		virtual ~LeNetNetwork();
	};
}