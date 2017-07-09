#pragma once

#include <dlib/dnn.h>

#ifndef __INTELLISENSE__

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

template <typename SUBNET> using level1 = res<64, /*res<8,*/ res_down<64, SUBNET>/*>*/>;
template <typename SUBNET> using level2 = res<32, /*res<8, res<8, res<8, res<8,*/ res_down<32, SUBNET>/*>>>>*/>;
template <typename SUBNET> using level3 = res<16, /*res<8, res<8,*/ res_down<16, SUBNET>/*>>*/>;
template <typename SUBNET> using level4 = res<8, /*res<8,*/ res_down<8, SUBNET>/*>*/>;

template <typename SUBNET> using alevel1 = ares<64, /*ares<8,*/ ares_down<64, SUBNET>/*>*/>;
template <typename SUBNET> using alevel2 = ares<32, /*ares<8, ares<8, ares<8, ares<8,*/ ares_down<32, SUBNET>/*>>>>*/>;
template <typename SUBNET> using alevel3 = ares<16, /*ares<8, ares<8,*/ ares_down<16, SUBNET>/*>>*/>;
template <typename SUBNET> using alevel4 = ares<8, /*ares<8,*/ ares_down<8, SUBNET>/*>*/>;

template <typename SUBNET> using level1t = res<64, /*res<8,*/ res_up<64, SUBNET>/*>*/>;
template <typename SUBNET> using level2t = res<32, /*res<8, res<8, res<8, res<8,*/ res_up<32, SUBNET>/*>>>>*/>;
template <typename SUBNET> using level3t = res<16, /*res<8, res<8,*/ res_up<16, SUBNET>/*>>*/>;
template <typename SUBNET> using level4t = res<8, /*res<8,*/ res_up<8, SUBNET>/*>*/>;

template <typename SUBNET> using alevel1t = ares<64, /*ares<8,*/ ares_up<64, SUBNET>/*>*/>;
template <typename SUBNET> using alevel2t = ares<32, /*ares<8, ares<8, ares<8, ares<8,*/ ares_up<32, SUBNET>/*>>>>*/>;
template <typename SUBNET> using alevel3t = ares<16, /*ares<8, ares<8,*/ ares_up<16, SUBNET>/*>>*/>;
template <typename SUBNET> using alevel4t = ares<8, /*ares<8,*/ ares_up<8, SUBNET>/*>*/>;

// training network type
using net_type = dlib::loss_mean_squared_per_pixel<
    dlib::bn_con<dlib::cont<1, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<8, 7, 7, 3, 3,
    level4t<level3t<level2t<level1t<
    level1<level2<level3<level4<
    dlib::max_pool<7, 7, 3, 3, dlib::relu<dlib::bn_con<dlib::con<8, 7, 7, 3, 3,
    dlib::input<dlib::matrix<float>>
    >>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_mean_squared_per_pixel<
    dlib::affine<dlib::cont<1, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<8, 7, 7, 3, 3,
    alevel4t<alevel3t<alevel2t<alevel1t<
    alevel1<alevel2<alevel3<alevel4<
    dlib::max_pool<7, 7, 3, 3, dlib::relu<dlib::affine<dlib::con<8, 7, 7, 3, 3,
    dlib::input<dlib::matrix<float>>
    >>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // __INTELLISENSE__
