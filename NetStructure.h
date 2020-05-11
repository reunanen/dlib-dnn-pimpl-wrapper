#pragma once

#include <dlib/dnn.h>
#include "NetDimensions.h"

// ----------------------------------------------------------------------------------------

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
typedef dlib::input_grayscale_image<dlib::memory_manager_stateless<uint8_t>::kernel_2_3e> input_layer_type;
#else
typedef dlib::input_rgb_image<dlib::memory_manager_stateless<uint8_t>::kernel_2_3e> input_layer_type;
#endif

#ifndef __INTELLISENSE__

// ----------------------------------------------------------------------------------------

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using res = dlib::relu<residual<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_down = dlib::relu<residual_down<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

// ----------------------------------------------------------------------------------------

#ifndef DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#define DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT (6)
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be greater than or equal to 0.");
static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT <= 6, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be less than or equal to 6.");

template <typename SUBNET> using level1 = res<768,res<768,res_down<768,SUBNET>>>;
template <typename SUBNET> using level2 = res<640,res<640,res_down<640,SUBNET>>>;
template <typename SUBNET> using level3 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level4 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level5 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level6 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<768,ares<768,ares_down<768,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<640,ares<640,ares_down<640,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel4 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel5 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel6 = ares<64,ares<64,ares<64,SUBNET>>>;

constexpr long default_class_count = 2;

// training network type
using net_type = dlib::loss_multiclass_log<dlib::fc<default_class_count,dlib::avg_pool_everything<
                            level1<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            level2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            level3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            level4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            level5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            level6<
#endif
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::bn_con<dlib::con<64,3,3,2,2,
                            input_layer_type
                            >>>>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >
#endif
                            >>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log<dlib::fc<default_class_count,dlib::avg_pool_everything<
                            alevel1<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            alevel2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            alevel3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            alevel4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            alevel5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            alevel6<
#endif
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<64,3,3,2,2,
                            input_layer_type
                            >>>>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >
#endif
                            >>>>;

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------
