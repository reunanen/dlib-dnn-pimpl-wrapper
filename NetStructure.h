#pragma once

#include <dlib/dnn.h>
#include "NetDimensions.h"

// ----------------------------------------------------------------------------------------

// Set default configuration (not in use right now)

#ifndef DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#define DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT (1)
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

#ifndef DLIB_DNN_PIMPL_WRAPPER_LEVEL_DEPTH
#define DLIB_DNN_PIMPL_WRAPPER_LEVEL_DEPTH (2)
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_DEPTH

// ----------------------------------------------------------------------------------------

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
#error TODO: implement grayscale input
#else
typedef dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>> input_layer_type;
#endif

// ----------------------------------------------------------------------------------------

#ifndef __INTELLISENSE__

// Essentially taken from https://github.com/davisking/dlib/blob/master/examples/dnn_mmod_ex.cpp

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3 = dlib::con<num_filters,3,3,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.
// We also use relu and batch normalization in the standard way.
template <typename SUBNET> using bdownsampler = dlib::relu<dlib::bn_con<con5d<32,dlib::relu<dlib::bn_con<con5d<32,dlib::relu<dlib::bn_con<con5d<32,SUBNET>>>>>>>>>;
template <typename SUBNET> using adownsampler = dlib::relu<dlib::affine<con5d<32,dlib::relu<dlib::affine<con5d<32,dlib::relu<dlib::affine<con5d<32,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using brcon3 = dlib::relu<dlib::bn_con<con3<32,SUBNET>>>;
template <typename SUBNET> using arcon3 = dlib::relu<dlib::affine<con3<32,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using bnet_type = dlib::loss_mmod<dlib::con<1,6,6,1,1,brcon3<brcon3<brcon3<bdownsampler<input_layer_type>>>>>>;
using anet_type = dlib::loss_mmod<dlib::con<1,6,6,1,1,arcon3<arcon3<arcon3<adownsampler<input_layer_type>>>>>>;

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------
