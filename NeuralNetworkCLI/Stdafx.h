// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently,
// but are changed infrequently

#pragma once

#ifdef __LENETLIB_EXPORT
#define __LENETLIB_DLLEXPORT __declspec(dllexport)
#else
#define __LENETLIB_DLLEXPORT __declspec(dllimport)
#endif

#include "nnlib.h"
#include "../NeuralNetworkCLI/LeNetConfiguration.h"
#include "../NeuralNetworkCLI/DataSetItem.h"
#include "../NeuralNetworkCLI/TrainingResults.h"
#include "../NeuralNetworkCLI/Step.h"
#include "../NeuralNetworkCLI/RectangularStep.h"
#include "../NeuralNetworkCLI/FlatStep.h"
#include "../NeuralNetworkCLI/LeNetNetwork.h"