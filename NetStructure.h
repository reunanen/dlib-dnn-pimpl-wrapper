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
// The network first does downsampling, and then upsampling. U-net style
// skip connections are used.

template <int N, typename SUBNET> using down                                = dlib::max_pool<3,3,2,2,SUBNET>;
template <int N, template <typename> class BN, typename SUBNET> using up    = dlib::cont<N,3,3,2,2,SUBNET>; // NB: the BN template parameter is not used at the moment!
template <int N, template <typename> class BN, typename SUBNET> using level = dlib::relu<BN<dlib::con<N,3,3,1,1,SUBNET>>>;

// ----------------------------------------------------------------------------------------

#ifndef DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT
#define DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT (4)
#endif // DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT

static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 0, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be greater than or equal to 0.");
static_assert(DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT <= 6, "If defined, DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT must be less than or equal to 6.");

#ifndef DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL
#define DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL (1)
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL

static_assert(DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 1, "If defined, DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL must be greater than or equal to 0.");
static_assert(DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL <= 7, "If defined, DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL must be less than or equal to 7.");

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using utag1 = dlib::add_tag_layer<2100+1,SUBNET>; // U-net top layer
template <typename SUBNET> using utag2 = dlib::add_tag_layer<2100+2,SUBNET>;
template <typename SUBNET> using utag3 = dlib::add_tag_layer<2100+3,SUBNET>;
template <typename SUBNET> using utag4 = dlib::add_tag_layer<2100+4,SUBNET>;
template <typename SUBNET> using utag5 = dlib::add_tag_layer<2100+5,SUBNET>;
template <typename SUBNET> using utag6 = dlib::add_tag_layer<2100+6,SUBNET>;

template <typename SUBNET> using concat_utag1
#if DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 1
    = dlib::concat_prev<utag1,SUBNET>;
#else // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 1
    = SUBNET;
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 1
template <typename SUBNET> using concat_utag2
#if DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 2
    = dlib::concat_prev<utag2,SUBNET>;
#else // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 2
    = SUBNET;
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 2
template <typename SUBNET> using concat_utag3
#if DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 3
    = dlib::concat_prev<utag3,SUBNET>;
#else // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 3
    = SUBNET;
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 3
template <typename SUBNET> using concat_utag4
#if DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 4
    = dlib::concat_prev<utag4,SUBNET>;
#else // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 4
    = SUBNET;
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 4
template <typename SUBNET> using concat_utag5
#if DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 5
    = dlib::concat_prev<utag5,SUBNET>;
#else // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 5
    = SUBNET;
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 5
template <typename SUBNET> using concat_utag6
#if DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 6
    = dlib::concat_prev<utag6, SUBNET>;
#else // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 6
    = SUBNET;
#endif // DLIB_DNN_PIMPL_WRAPPER_UNET_STARTING_LEVEL >= 6

// ----------------------------------------------------------------------------------------

constexpr int default_level0_feature_count = 16;
constexpr int default_level1_feature_count = 16;
constexpr int default_level2_feature_count = 24;
constexpr int default_level3_feature_count = 32;
constexpr int default_level4_feature_count = 40;
constexpr int default_level5_feature_count = 48;
constexpr int default_level6_feature_count = 56;

template <typename SUBNET> using blevel0 = level<default_level0_feature_count, dlib::bn_con, SUBNET>;
template <typename SUBNET> using blevel1 = level<default_level1_feature_count, dlib::bn_con, SUBNET>;
template <typename SUBNET> using blevel2 = level<default_level2_feature_count, dlib::bn_con, SUBNET>;
template <typename SUBNET> using blevel3 = level<default_level3_feature_count, dlib::bn_con, SUBNET>;
template <typename SUBNET> using blevel4 = level<default_level4_feature_count, dlib::bn_con, SUBNET>;
template <typename SUBNET> using blevel5 = level<default_level5_feature_count, dlib::bn_con, SUBNET>;
template <typename SUBNET> using blevel6 = level<default_level6_feature_count, dlib::bn_con, SUBNET>;

template <typename SUBNET> using alevel0 = level<default_level0_feature_count, dlib::affine, SUBNET>;
template <typename SUBNET> using alevel1 = level<default_level1_feature_count, dlib::affine, SUBNET>;
template <typename SUBNET> using alevel2 = level<default_level2_feature_count, dlib::affine, SUBNET>;
template <typename SUBNET> using alevel3 = level<default_level3_feature_count, dlib::affine, SUBNET>;
template <typename SUBNET> using alevel4 = level<default_level4_feature_count, dlib::affine, SUBNET>;
template <typename SUBNET> using alevel5 = level<default_level5_feature_count, dlib::affine, SUBNET>;
template <typename SUBNET> using alevel6 = level<default_level6_feature_count, dlib::affine, SUBNET>;

template <typename SUBNET> using bup1 = blevel1<concat_utag1<up<default_level1_feature_count, dlib::bn_con, SUBNET>>>;
template <typename SUBNET> using bup2 = blevel2<concat_utag2<up<default_level2_feature_count, dlib::bn_con, SUBNET>>>;
template <typename SUBNET> using bup3 = blevel3<concat_utag3<up<default_level3_feature_count, dlib::bn_con, SUBNET>>>;
template <typename SUBNET> using bup4 = blevel4<concat_utag4<up<default_level4_feature_count, dlib::bn_con, SUBNET>>>;
template <typename SUBNET> using bup5 = blevel5<concat_utag5<up<default_level5_feature_count, dlib::bn_con, SUBNET>>>;
template <typename SUBNET> using bup6 = blevel6<concat_utag6<up<default_level6_feature_count, dlib::bn_con, SUBNET>>>;

template <typename SUBNET> using bdown1 = blevel1<down<default_level1_feature_count, utag1<SUBNET>>>;
template <typename SUBNET> using bdown2 = blevel2<down<default_level2_feature_count, utag2<SUBNET>>>;
template <typename SUBNET> using bdown3 = blevel3<down<default_level3_feature_count, utag3<SUBNET>>>;
template <typename SUBNET> using bdown4 = blevel4<down<default_level4_feature_count, utag4<SUBNET>>>;
template <typename SUBNET> using bdown5 = blevel5<down<default_level5_feature_count, utag5<SUBNET>>>;
template <typename SUBNET> using bdown6 = blevel6<down<default_level6_feature_count, utag6<SUBNET>>>;

template <typename SUBNET> using aup1 = alevel1<concat_utag1<up<default_level1_feature_count, dlib::affine, SUBNET>>>;
template <typename SUBNET> using aup2 = alevel2<concat_utag2<up<default_level2_feature_count, dlib::affine, SUBNET>>>;
template <typename SUBNET> using aup3 = alevel3<concat_utag3<up<default_level3_feature_count, dlib::affine, SUBNET>>>;
template <typename SUBNET> using aup4 = alevel4<concat_utag4<up<default_level4_feature_count, dlib::affine, SUBNET>>>;
template <typename SUBNET> using aup5 = alevel5<concat_utag5<up<default_level5_feature_count, dlib::affine, SUBNET>>>;
template <typename SUBNET> using aup6 = alevel6<concat_utag6<up<default_level6_feature_count, dlib::affine, SUBNET>>>;

template <typename SUBNET> using adown1 = alevel1<down<default_level1_feature_count, utag1<SUBNET>>>;
template <typename SUBNET> using adown2 = alevel2<down<default_level2_feature_count, utag2<SUBNET>>>;
template <typename SUBNET> using adown3 = alevel3<down<default_level3_feature_count, utag3<SUBNET>>>;
template <typename SUBNET> using adown4 = alevel4<down<default_level4_feature_count, utag4<SUBNET>>>;
template <typename SUBNET> using adown5 = alevel5<down<default_level5_feature_count, utag5<SUBNET>>>;
template <typename SUBNET> using adown6 = alevel6<down<default_level6_feature_count, utag6<SUBNET>>>;

// ----------------------------------------------------------------------------------------

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
                            dlib::con<default_class_count,1,1,1,1,
                            blevel0<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            bup1<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            bup2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            bup3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            bup4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            bup5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            bup6<
#endif
                            //
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            bdown6<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            bdown5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            bdown4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            bdown3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            bdown2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            bdown1<
#endif
                            input_layer_type
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
                            >>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
                            dlib::con<default_class_count,1,1,1,1,
                            alevel0<
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            aup1<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            aup2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            aup3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            aup4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            aup5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            aup6<
#endif
                            //
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 6
                            adown6<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 5
                            adown5<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 4
                            adown4<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 3
                            adown3<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 2
                            adown2<
#endif
#if DLIB_DNN_PIMPL_WRAPPER_LEVEL_COUNT >= 1
                            adown1<
#endif
                            input_layer_type
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
