#include "NetDimensions.h"

namespace {
    constexpr int startingPoint = 225; // A rather arbitrary selection
    constexpr int testInputDim = NetInputs<startingPoint>::count;
    constexpr int testOutputDim = NetOutputs<testInputDim>::count;
    static_assert(testOutputDim == startingPoint, "I/O dimension mismatch detected");
}
