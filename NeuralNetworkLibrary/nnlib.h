#pragma once
#ifdef __LENETLIB_EXPORT
#define __LENETLIB_DLLEXPORT __declspec(dllexport)
#else
#define __LENETLIB_DLLEXPORT __declspec(dllimport)
#endif

#include <random>
#include <vector>
#include <math.h>
#include "Step.h"
#include "Weights.h"
#include "FlatStep.h"
#include "BiasedWeights.h"
#include "RectangularWeights.h"
#include "RectangularStep.h"
#include "ConvolutionWeights.h"
#include "ConvolutionStep.h"
#include "SubsamplingWeights.h"
#include "SubsamplingStep.h"
#include "FeedForwardWeights.h"
#include "FeedForwardStep.h"
#include "LeNetConfiguration.h"
#include "MarkingWeights.h"
#include "MarkingStep.h"
#include "InputStep.h"
#include "DataSetItem.h"
#include "TrainingResults.h"
#include "LeNetNetwork.h"


//#define NDEBUG
#include <assert.h>