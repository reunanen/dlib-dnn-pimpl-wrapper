#pragma once

#include <dlib/dnn.h>

// ----------------------------------------------------------------------------------------

#ifndef __INTELLISENSE__

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using blockt = BN<dlib::cont<N, 3, 3, 1, 1, dlib::relu<BN<dlib::cont<N, 3, 3, stride, stride, SUBNET>>>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N, 2, 2, 2, 2, dlib::skip1<dlib::tag2<blockt<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, typename SUBNET> using res = dlib::relu<residual<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_down = dlib::relu<residual_down<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_up = dlib::relu<residual_up<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_up = dlib::relu<residual_up<block, N, dlib::affine, SUBNET>>;

// ----------------------------------------------------------------------------------------

#if 1
template <typename SUBNET> using level1 = res<512, res_down<512, SUBNET>>;
template <typename SUBNET> using level2 = res<256, res_down<256, SUBNET>>;
template <typename SUBNET> using level3 = res<128, res_down<128, SUBNET>>;
template <typename SUBNET> using level4 = res<64, res<64, SUBNET>>;

template <typename SUBNET> using alevel1 = ares<512, ares_down<512, SUBNET>>;
template <typename SUBNET> using alevel2 = ares<256, ares_down<256, SUBNET>>;
template <typename SUBNET> using alevel3 = ares<128, ares_down<128, SUBNET>>;
template <typename SUBNET> using alevel4 = ares<64, ares<64, SUBNET>>;

template <typename SUBNET> using level1t = res<512, res_up<512, SUBNET>>;
template <typename SUBNET> using level2t = res<256, res_up<256, SUBNET>>;
template <typename SUBNET> using level3t = res<128, res_up<128, SUBNET>>;
template <typename SUBNET> using level4t = res<64, res_up<64, SUBNET>>;

template <typename SUBNET> using alevel1t = ares<512, ares_up<512, SUBNET>>;
template <typename SUBNET> using alevel2t = ares<256, ares_up<256, SUBNET>>;
template <typename SUBNET> using alevel3t = ares<128, ares_up<128, SUBNET>>;
template <typename SUBNET> using alevel4t = ares<64, ares_up<64, SUBNET>>;
#endif

#if 0
template <typename SUBNET> using level1 = res<512, res<512, res_down<512, SUBNET>>>;
template <typename SUBNET> using level2 = res<256, res<256, res<256, res<256, res<256, res_down<256, SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128, res<128, res<128, res_down<128, SUBNET>>>>;
template <typename SUBNET> using level4 = res<64, res<64, res<64, SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512, ares<512, ares_down<512, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256, ares<256, ares<256, ares<256, ares<256, ares_down<256, SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128, ares<128, ares<128, ares_down<128, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64, ares<64, ares<64, SUBNET>>>;

template <typename SUBNET> using level1t = res<512, res<512, res_up<512, SUBNET>>>;
template <typename SUBNET> using level2t = res<256, res<256, res<256, res<256, res<256, res_up<256, SUBNET>>>>>>;
template <typename SUBNET> using level3t = res<128, res<128, res<128, res_up<128, SUBNET>>>>;
template <typename SUBNET> using level4t = res<64, res<64, res_up<64, SUBNET>>>;

template <typename SUBNET> using alevel1t = ares<512, ares<512, ares_up<512, SUBNET>>>;
template <typename SUBNET> using alevel2t = ares<256, ares<256, ares<256, ares<256, ares<256, ares_up<256, SUBNET>>>>>>;
template <typename SUBNET> using alevel3t = ares<128, ares<128, ares<128, ares_up<128, SUBNET>>>>;
template <typename SUBNET> using alevel4t = ares<64, ares<64, ares_up<64, SUBNET>>>;
#endif

constexpr long default_class_count = 2;

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel<
    dlib::bn_con<dlib::cont<default_class_count, 7, 7, 2, 2,
    level4t<level3t<level2t<level1t<
    level1<level2<level3<level4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::bn_con<dlib::con<64, 7, 7, 2, 2,
    dlib::input<dlib::matrix<dlib::rgb_pixel>>
    >>>>>>>>>>>>>>>;

// inference network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel<
    dlib::affine<dlib::cont<default_class_count, 7, 7, 2, 2,
    alevel4t<alevel3t<alevel2t<alevel1t<
    alevel1<alevel2<alevel3<alevel4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    dlib::input<dlib::matrix<dlib::rgb_pixel>>
    >>>>>>>>>>>>>>>;

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------
