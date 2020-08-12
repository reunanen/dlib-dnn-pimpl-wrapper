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

const int output_channel_count = 1;

// ----------------------------------------------------------------------------------------

// Introduce the building blocks used to define the segmentation network.
// The network first does downsampling, and then upsampling, using DenseNet
// blocks. In addition, U-net style skip connections are employed.

template <int N, template <typename> class BN, typename SUBNET>
using dense_block_layer = BN<dlib::mish<dlib::con<N,3,3,1,1,SUBNET>>>;

template <typename SUBNET> using dintag = dlib::add_tag_layer<2000+0,SUBNET>; // input to the dense block
template <typename SUBNET> using dotag0 = dlib::add_tag_layer<2000+1,SUBNET>; // output of the first layer of the dense block
template <typename SUBNET> using dctag0 = dlib::add_tag_layer<2000+2,SUBNET>; // dintag and dotag0 concatenated
template <typename SUBNET> using dotag1 = dlib::add_tag_layer<2000+3,SUBNET>; // output of the second layer of the dense block
template <typename SUBNET> using dctag1 = dlib::add_tag_layer<2000+4,SUBNET>; // dctag0 and dotag1 concatenated
template <typename SUBNET> using dotag2 = dlib::add_tag_layer<2000+5,SUBNET>; // output of the third layer of the dense block

// TODO: can this be used??
//template <template<typename> class TAG, typename SUBNET>
//using concat_prev = dlib::add_layer<dlib::concat<TAG, SUBNET>>;

// The following dense block is modeled after Figure 2 of the tiramisu
// paper (Jegou et al 2017, https://arxiv.org/pdf/1611.09326.pdf).
template <int N, template <typename> class BN, typename SUBNET>
using dense_block4 = dlib::concat3<dotag0,dotag1,dotag2,    dense_block_layer<N,BN,
                            dlib::concat_prev<dctag1,dotag2<dense_block_layer<N,BN,
                     dctag1<dlib::concat_prev<dctag0,dotag1<dense_block_layer<N,BN,
                     dctag0<dlib::concat_prev<dintag,dotag0<dense_block_layer<N,BN,
                     dintag<SUBNET>
                     >>>>>>>>>>>>>;

// More light-weight versions of the above.
// (Note that more heavy-weight versions are rather easy to extrapolate as well.)
template <int N, template <typename> class BN, typename SUBNET>
using dense_block3 = dlib::concat2<dotag0,dotag1,           dense_block_layer<N,BN,
                            dlib::concat_prev<dctag0,dotag1<dense_block_layer<N,BN,
                     dctag0<dlib::concat_prev<dintag,dotag0<dense_block_layer<N,BN,
                     dintag<SUBNET>
                     >>>>>>>>>;

template <int N, template <typename> class BN, typename SUBNET>
using dense_block2 = dlib::concat_prev<dotag0,       dense_block_layer<N,BN,
                     dlib::concat_prev<dintag,dotag0<dense_block_layer<N,BN,
                     dintag<SUBNET>
                     >>>>>;

template <int N, template <typename> class BN, typename SUBNET>
using dense_block1 = dlib::concat_prev<dintag,dense_block_layer<N,BN,
                     dintag<SUBNET>
                     >>;

template <int N, typename SUBNET> using dense1    = dense_block1<N, dlib::bn_con, SUBNET>;
template <int N, typename SUBNET> using adense1   = dense_block1<N, dlib::affine, SUBNET>;
template <int N, typename SUBNET> using dense2    = dense_block2<N, dlib::bn_con, SUBNET>;
template <int N, typename SUBNET> using adense2   = dense_block2<N, dlib::affine, SUBNET>;
template <int N, typename SUBNET> using dense3    = dense_block3<N, dlib::bn_con, SUBNET>;
template <int N, typename SUBNET> using adense3   = dense_block3<N, dlib::affine, SUBNET>;
template <int N, typename SUBNET> using dense4    = dense_block4<N,dlib::bn_con,SUBNET>;
template <int N, typename SUBNET> using adense4   = dense_block4<N,dlib::affine,SUBNET>;

template <int N, template <typename> class BN, typename SUBNET>
using transition_down = BN<dlib::mish<dlib::con<N, 1, 1, 1, 1, dlib::max_pool<3, 3, 2, 2, SUBNET>>>>;

template <int N, typename SUBNET> using down	  = transition_down<N,dlib::bn_con,SUBNET>;
template <int N, typename SUBNET> using adown     = transition_down<N,dlib::affine,SUBNET>;

template <int N, typename SUBNET> using up        = dlib::cont<N,3,3,2,2,SUBNET>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using utag1 = dlib::add_tag_layer<2100+1,SUBNET>; // U-net top layer
template <typename SUBNET> using utag2 = dlib::add_tag_layer<2100+2,SUBNET>;
template <typename SUBNET> using utag3 = dlib::add_tag_layer<2100+3,SUBNET>;
template <typename SUBNET> using utag4 = dlib::add_tag_layer<2100+4,SUBNET>;
template <typename SUBNET> using utag5 = dlib::add_tag_layer<2100+5,SUBNET>;
template <typename SUBNET> using utag6 = dlib::add_tag_layer<2100+6,SUBNET>;

template <typename SUBNET> using concat_utag1 = dlib::concat_prev<utag1,SUBNET>;
template <typename SUBNET> using concat_utag2 = dlib::concat_prev<utag2,SUBNET>;
template <typename SUBNET> using concat_utag3 = dlib::concat_prev<utag3,SUBNET>;
template <typename SUBNET> using concat_utag4 = dlib::concat_prev<utag4,SUBNET>;
template <typename SUBNET> using concat_utag5 = dlib::concat_prev<utag5,SUBNET>;
template <typename SUBNET> using concat_utag6 = dlib::concat_prev<utag6,SUBNET>;

// ----------------------------------------------------------------------------------------

#ifndef DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#define DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT (6)
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 0, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be greater than or equal to 0.");
static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT <= 6, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be less than or equal to 6.");

constexpr int default_level0_feature_count = 16;
constexpr int default_level1_feature_count = 16;
constexpr int default_level2_feature_count = 24;
constexpr int default_level3_feature_count = 32;
constexpr int default_level4_feature_count = 40;
constexpr int default_level5_feature_count = 48;
constexpr int default_level6_feature_count = 56;

#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 6
constexpr int default_deepest_level_feature_count = 64;
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 5
constexpr int default_deepest_level_feature_count = 56;
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 4
constexpr int default_deepest_level_feature_count = 48;
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 3
constexpr int default_deepest_level_feature_count = 40;
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 2
constexpr int default_deepest_level_feature_count = 32;
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 1
constexpr int default_deepest_level_feature_count = 24;
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 0
constexpr int default_deepest_level_feature_count = 16;
#else
#error unexpected DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

template <typename SUBNET> using level0   = dlib::mish<dlib::bn_con<dlib::con <default_level0_feature_count,3,3,1,1,SUBNET>>>;
template <typename SUBNET> using alevel0  = dlib::mish<dlib::affine<dlib::con <default_level0_feature_count,3,3,1,1,SUBNET>>>;
template <typename SUBNET> using level0t  = dlib::mish<dlib::bn_con<dlib::cont<default_level0_feature_count,3,3,1,1,SUBNET>>>;
template <typename SUBNET> using alevel0t = dlib::mish<dlib::affine<dlib::cont<default_level0_feature_count,3,3,1,1,SUBNET>>>;

template <typename SUBNET> using level1 = down<default_level1_feature_count,utag1<dense1<default_level1_feature_count,SUBNET>>>;
template <typename SUBNET> using level2 = down<default_level2_feature_count,utag2<dense2<default_level2_feature_count,SUBNET>>>;
template <typename SUBNET> using level3 = down<default_level3_feature_count,utag3<dense2<default_level3_feature_count,SUBNET>>>;
template <typename SUBNET> using level4 = down<default_level4_feature_count,utag4<dense3<default_level4_feature_count,SUBNET>>>;
template <typename SUBNET> using level5 = down<default_level5_feature_count,utag5<dense3<default_level5_feature_count,SUBNET>>>;
template <typename SUBNET> using level6 = down<default_level6_feature_count,utag6<dense4<default_level6_feature_count,SUBNET>>>;

template <typename SUBNET> using alevel1 = adown<default_level1_feature_count,utag1<adense1<default_level1_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel2 = adown<default_level2_feature_count,utag2<adense2<default_level2_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel3 = adown<default_level3_feature_count,utag3<adense2<default_level3_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel4 = adown<default_level4_feature_count,utag4<adense3<default_level4_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel5 = adown<default_level5_feature_count,utag5<adense3<default_level5_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel6 = adown<default_level6_feature_count,utag6<adense4<default_level6_feature_count,SUBNET>>>;

template <typename SUBNET> using level1t = dense1<default_level1_feature_count,concat_utag1<up<default_level1_feature_count,SUBNET>>>;
template <typename SUBNET> using level2t = dense2<default_level2_feature_count,concat_utag2<up<default_level2_feature_count,SUBNET>>>;
template <typename SUBNET> using level3t = dense2<default_level3_feature_count,concat_utag3<up<default_level3_feature_count,SUBNET>>>;
template <typename SUBNET> using level4t = dense3<default_level4_feature_count,concat_utag4<up<default_level4_feature_count,SUBNET>>>;
template <typename SUBNET> using level5t = dense3<default_level5_feature_count,concat_utag5<up<default_level5_feature_count,SUBNET>>>;
template <typename SUBNET> using level6t = dense4<default_level6_feature_count,concat_utag6<up<default_level6_feature_count,SUBNET>>>;

template <typename SUBNET> using alevel1t = adense1<default_level1_feature_count,concat_utag1<up<default_level1_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel2t = adense2<default_level2_feature_count,concat_utag2<up<default_level2_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel3t = adense2<default_level3_feature_count,concat_utag3<up<default_level3_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel4t = adense3<default_level4_feature_count,concat_utag4<up<default_level4_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel5t = adense3<default_level5_feature_count,concat_utag5<up<default_level5_feature_count,SUBNET>>>;
template <typename SUBNET> using alevel6t = adense4<default_level6_feature_count,concat_utag6<up<default_level6_feature_count,SUBNET>>>;

// ----------------------------------------------------------------------------------------

// training network type
using net_type = dlib::loss_mean_squared_per_pixel<
                            dlib::con<output_channel_count,1,1,1,1,
                            level0t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            level1t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            level2t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            level3t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            level4t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            level5t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            level6t<
#endif
                            dense4<default_deepest_level_feature_count,
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            level6<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            level5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            level4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            level3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            level2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            level1<
#endif
                            level0<
                            input_layer_type
                            >
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            >>
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            > // for the deepest level - let's just put it here
                            >>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_mean_squared_per_pixel<
                            dlib::con<output_channel_count,1,1,1,1,
                            alevel0t<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            alevel1t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            alevel2t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            alevel3t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            alevel4t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            alevel5t<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            alevel6t<
#endif
                            adense4<default_deepest_level_feature_count,
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            alevel6<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            alevel5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            alevel4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            alevel3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            alevel2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            alevel1<
#endif
                            alevel0<
                            input_layer_type
                            >
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            >>
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            >>
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1    
                            > // for the deepest level - let's just put it here
                            >>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT,W,3,2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT,W,3,2>::count
    };
};

#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 6
static_assert(NetInputs<1>::count == 127, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 5
static_assert(NetInputs<1>::count == 63, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 4
static_assert(NetInputs<1>::count == 31, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 3
static_assert(NetInputs<1>::count == 15, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 2
static_assert(NetInputs<1>::count == 7, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 1
static_assert(NetInputs<1>::count == 3, "Unexpected net input count");
#elif DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT == 0
static_assert(NetInputs<1>::count == 1, "Unexpected net input count");
#else
#error unexpected DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------
