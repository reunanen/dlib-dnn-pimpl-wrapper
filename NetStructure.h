#pragma once

#include <dlib/dnn.h>

#ifndef __INTELLISENSE__

#if 0
// ----------------------------------------------------------------------------------------

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <template <int, template<typename>class, int, typename> class blockt, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N, 2, 2, 2, 2, dlib::skip1<dlib::tag2<blockt<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using blockt = BN<dlib::cont<N, 3, 3, 1, 1, dlib::relu<BN<dlib::cont<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using res = dlib::relu<residual<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_down = dlib::relu<residual_down<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_up = dlib::relu<residual_up<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_up = dlib::relu<residual_up<block, N, dlib::affine, SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<512, res<512, res_down<512, SUBNET>>>;
template <typename SUBNET> using level2 = res<256, res<256, res<256, res<256, res<256, res_down<256, SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128, res<128, res<128, res_down<128, SUBNET>>>>;
template <typename SUBNET> using level4 = res<64, res<64, res_down<64, SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512, ares<512, ares_down<512, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256, ares<256, ares<256, ares<256, ares<256, ares_down<256, SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128, ares<128, ares<128, ares_down<128, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64, ares<64, ares_down<64, SUBNET>>>;

template <typename SUBNET> using level1t = res<512, res<512, res_up<512, SUBNET>>>;
template <typename SUBNET> using level2t = res<256, res<256, res<256, res<256, res<256, res_up<256, SUBNET>>>>>>;
template <typename SUBNET> using level3t = res<128, res<128, res<128, res_up<128, SUBNET>>>>;
template <typename SUBNET> using level4t = res<64, res<64, res_up<64, SUBNET>>>;

template <typename SUBNET> using alevel1t = ares<512, ares<512, ares_up<512, SUBNET>>>;
template <typename SUBNET> using alevel2t = ares<256, ares<256, ares<256, ares<256, ares<256, ares_up<256, SUBNET>>>>>>;
template <typename SUBNET> using alevel3t = ares<128, ares<128, ares<128, ares_up<128, SUBNET>>>>;
template <typename SUBNET> using alevel4t = ares<64, ares<64, ares_up<64, SUBNET>>>;
#endif

// training network type
using net_type = dlib::loss_mean_squared_per_pixel<
    dlib::bn_con<dlib::cont<1, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<64, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<128, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<256, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::con<512, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::con<256, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::con<128, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::con<64, 7, 7, 3, 3,
    dlib::input<dlib::matrix<float>>
    >>>>>>>>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_mean_squared_per_pixel<
    dlib::affine<dlib::cont<1, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<64, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<128, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<256, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::con<512, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::con<256, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::con<128, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::con<64, 7, 7, 3, 3,
    dlib::input<dlib::matrix<float>>
    >>>>>>>>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // __INTELLISENSE__
