#pragma once

#include <dlib/dnn.h>
#include "NetDimensions.h"

// ----------------------------------------------------------------------------------------

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
typedef dlib::input_grayscale_image<dlib::memory_manager_stateless<uint8_t>::kernel_2_3e> input_layer_type;
#else
typedef dlib::input_rgb_image<dlib::memory_manager_stateless<uint8_t>::kernel_2_3e> input_layer_type;
#endif

// ----------------------------------------------------------------------------------------

#ifndef __INTELLISENSE__

constexpr long default_class_count = 2;

// Introduce the building blocks used to define the segmentation network.
// The network first does residual downsampling (similar to the dnn_imagenet_(train_)ex 
// example program), and then residual upsampling. The network could be improved e.g.
// by introducing skip connections from the input image, and/or the first layers, to the
// last layer(s).  (See Long et al., Fully Convolutional Networks for Semantic Segmentation,
// https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block = BN<dlib::con<N,3,3,1,1, dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using blockt = BN<dlib::cont<N,3,3,1,1,dlib::relu<BN<dlib::cont<N,3,3,stride,stride,SUBNET>>>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N,2,2,2,2,dlib::skip1<dlib::tag2<blockt<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, typename SUBNET> using res       = dlib::relu<residual<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = dlib::relu<residual_down<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using res_up    = dlib::relu<residual_up<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_up   = dlib::relu<residual_up<block,N,dlib::affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using res512 = res<512, SUBNET>;
template <typename SUBNET> using res256 = res<256, SUBNET>;
template <typename SUBNET> using res128 = res<128, SUBNET>;
template <typename SUBNET> using res64  = res<64, SUBNET>;
template <typename SUBNET> using ares512 = ares<512, SUBNET>;
template <typename SUBNET> using ares256 = ares<256, SUBNET>;
template <typename SUBNET> using ares128 = ares<128, SUBNET>;
template <typename SUBNET> using ares64  = ares<64, SUBNET>;


template <typename SUBNET> using level1 = dlib::repeat<2,res512,res_down<512,SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<2,res256,res_down<256,SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<2,res128,res_down<128,SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<2,res64,res<64,SUBNET>>;

template <typename SUBNET> using alevel1 = dlib::repeat<2,ares512,ares_down<512,SUBNET>>;
template <typename SUBNET> using alevel2 = dlib::repeat<2,ares256,ares_down<256,SUBNET>>;
template <typename SUBNET> using alevel3 = dlib::repeat<2,ares128,ares_down<128,SUBNET>>;
template <typename SUBNET> using alevel4 = dlib::repeat<2,ares64,ares<64,SUBNET>>;

template <typename SUBNET> using level1t = dlib::repeat<2,res512,res_up<512,SUBNET>>;
template <typename SUBNET> using level2t = dlib::repeat<2,res256,res_up<256,SUBNET>>;
template <typename SUBNET> using level3t = dlib::repeat<2,res128,res_up<128,SUBNET>>;
template <typename SUBNET> using level4t = dlib::repeat<2,res64,res_up<64,SUBNET>>;

template <typename SUBNET> using alevel1t = dlib::repeat<2,ares512,ares_up<512,SUBNET>>;
template <typename SUBNET> using alevel2t = dlib::repeat<2,ares256,ares_up<256,SUBNET>>;
template <typename SUBNET> using alevel3t = dlib::repeat<2,ares128,ares_up<128,SUBNET>>;
template <typename SUBNET> using alevel4t = dlib::repeat<2,ares64,ares_up<64,SUBNET>>;

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
                            dlib::cont<default_class_count,7,7,2,2,
                            level4t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            level3t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            level2t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            level1t<
                            level1<
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            level2<
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            level3<
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            level4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::bn_con<dlib::con<64,7,7,2,2,
                            input_layer_type
                            >>>>>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >>
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
                            dlib::cont<default_class_count,7,7,2,2,
                            alevel4t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            alevel3t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            alevel2t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            alevel1t<
                            alevel1<
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            alevel2<
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            alevel3<
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<64,7,7,2,2,
                            input_layer_type
                            >>>>>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
    >>
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >>>;

#ifndef DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#define DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT (1)
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be greater than or equal to 1.");
static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT <= 4, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be less than or equal to 4.");

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<1,Inputs<1,Inputs<DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT-1,W,3,2>::count,3,2>::count,7,2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT-1,Outputs<1,Outputs<1,W,7,2>::count,3,2>::count,3,2>::count
    };
};

#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
static_assert(NetInputs<1>::count == 67, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
static_assert(NetInputs<1>::count == 35, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
static_assert(NetInputs<1>::count == 19, "Unexpected net input count");
#else // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
static_assert(NetInputs<1>::count == 11, "Unexpected net input count");
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------
