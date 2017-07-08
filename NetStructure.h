#pragma once

#include <dlib/dnn.h>

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

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

//template <int N, typename SUBNET> using forwardLayer = dlib::relu<block<N, dlib::bn_con, 2, SUBNET>>;
//template <int N, typename SUBNET> using aforwardLayer = dlib::relu<block<N, dlib::affine, 2, SUBNET>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using blockt = BN<dlib::cont<N, 3, 3, 1, 1, dlib::relu<BN<dlib::cont<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using transp = dlib::relu<blockt<N, dlib::bn_con, 2, SUBNET>>;
template <int N, typename SUBNET> using atransp = dlib::relu<blockt<N, dlib::affine, 2, SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<128, /*res<8,*/ res_down<128, SUBNET>/*>*/>;
template <typename SUBNET> using level2 = res<64, /*res<8, res<8, res<8, res<8,*/ res_down<64, SUBNET>/*>>>>*/>;
template <typename SUBNET> using level3 = res<32, /*res<8, res<8,*/ res_down<32, SUBNET>/*>>*/>;
template <typename SUBNET> using level4 = res<16, /*res<8,*/ res_down<16, SUBNET>/*>*/>;

template <typename SUBNET> using alevel1 = ares<128, /*ares<8,*/ ares_down<128, SUBNET>/*>*/>;
template <typename SUBNET> using alevel2 = ares<64, /*ares<8, ares<8, ares<8, ares<8,*/ ares_down<64, SUBNET>/*>>>>*/>;
template <typename SUBNET> using alevel3 = ares<32, /*ares<8, ares<8,*/ ares_down<32, SUBNET>/*>>*/>;
template <typename SUBNET> using alevel4 = ares<16, /*ares<8,*/ ares_down<16, SUBNET>/*>*/>;

//template <typename SUBNET> using level1 = forwardLayer<128, SUBNET>;
//template <typename SUBNET> using level2 = forwardLayer<64, SUBNET>;
//template <typename SUBNET> using level3 = forwardLayer<32, SUBNET>;
//template <typename SUBNET> using level4 = forwardLayer<16, SUBNET>;
//
//template <typename SUBNET> using alevel1 = aforwardLayer<128, SUBNET>;
//template <typename SUBNET> using alevel2 = aforwardLayer<64, SUBNET>;
//template <typename SUBNET> using alevel3 = aforwardLayer<32, SUBNET>;
//template <typename SUBNET> using alevel4 = aforwardLayer<16, SUBNET>;

template <typename SUBNET> using level1t = transp<128, SUBNET>;
template <typename SUBNET> using level2t = transp<64, SUBNET>;
template <typename SUBNET> using level3t = transp<32, SUBNET>;
template <typename SUBNET> using level4t = transp<16, SUBNET>;

template <typename SUBNET> using alevel1t = atransp<128, SUBNET>;
template <typename SUBNET> using alevel2t = atransp<64, SUBNET>;
template <typename SUBNET> using alevel3t = atransp<32, SUBNET>;
template <typename SUBNET> using alevel4t = atransp<16, SUBNET>;

// training network type
using net_type = dlib::loss_mean_squared_per_pixel<
    dlib::bn_con<dlib::cont<1, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<16, 7, 7, 3, 3,
    level4t<level3t<level2t<level1t<
    level1<level2<level3<level4<
    dlib::max_pool<7, 7, 3, 3, dlib::relu<dlib::bn_con<dlib::con<16, 7, 7, 3, 3,
    dlib::input<dlib::matrix<float>>
    >>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_mean_squared_per_pixel<
    dlib::affine<dlib::cont<1, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<16, 7, 7, 3, 3,
    alevel4t<alevel3t<alevel2t<alevel1t<
    alevel1<alevel2<alevel3<alevel4<
    dlib::max_pool<7, 7, 3, 3, dlib::relu<dlib::affine<dlib::con<16, 7, 7, 3, 3,
    dlib::input<dlib::matrix<float>>
    >>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // __INTELLISENSE__
