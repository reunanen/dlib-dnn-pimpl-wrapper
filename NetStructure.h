#pragma once

#include <dlib/dnn.h>
#include "NetDimensions.h"

// ----------------------------------------------------------------------------------------

#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
typedef dlib::input_grayscale_image input_layer_type;
#else
typedef dlib::input_rgb_image input_layer_type;
#endif

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

constexpr long default_class_count = 2;

#if 0

#if 0
template <typename SUBNET> using level1 = res_down<256, SUBNET>;
template <typename SUBNET> using level2 = res_down<128, SUBNET>;
template <typename SUBNET> using level3 = res_down<64, SUBNET>;
template <typename SUBNET> using level4 = res_down<32, SUBNET>;
template <typename SUBNET> using level5 = res<16, SUBNET>;

template <typename SUBNET> using alevel1 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel2 = ares_down<128, SUBNET>;
template <typename SUBNET> using alevel3 = ares_down<64, SUBNET>;
template <typename SUBNET> using alevel4 = ares_down<32, SUBNET>;
template <typename SUBNET> using alevel5 = ares<16, SUBNET>;

template <typename SUBNET> using level1t = res_up<256, SUBNET>;
template <typename SUBNET> using level2t = res_up<128, SUBNET>;
template <typename SUBNET> using level3t = res_up<64, SUBNET>;
template <typename SUBNET> using level4t = res_up<32, SUBNET>;
template <typename SUBNET> using level5t = res_up<16, SUBNET>;

template <typename SUBNET> using alevel1t = ares_up<256, SUBNET>;
template <typename SUBNET> using alevel2t = ares_up<128, SUBNET>;
template <typename SUBNET> using alevel3t = ares_up<64, SUBNET>;
template <typename SUBNET> using alevel4t = ares_up<32, SUBNET>;
template <typename SUBNET> using alevel5t = ares_up<16, SUBNET>;
#endif

#if 1
template <typename SUBNET> using level1 = res<512, res_down<512, SUBNET>>;
template <typename SUBNET> using level2 = res<256, res_down<256, SUBNET>>;
template <typename SUBNET> using level3 = res<128, res_down<128, SUBNET>>;
template <typename SUBNET> using level4 = res<64, res_down<64, SUBNET>>;

template <typename SUBNET> using alevel1 = ares<512, ares_down<512, SUBNET>>;
template <typename SUBNET> using alevel2 = ares<256, ares_down<256, SUBNET>>;
template <typename SUBNET> using alevel3 = ares<128, ares_down<128, SUBNET>>;
template <typename SUBNET> using alevel4 = ares<64, ares_down<64, SUBNET>>;

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

#if 1
// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::bn_con<dlib::cont<default_class_count,5,5,1,1,/*res<64,*/
    level4t<level3t<level2t<level1t<
    level1<level2<level3<level4<
    /*dlib::max_pool<3,3,2,2,dlib::relu<dlib::bn_con<dlib::con<64,5,5,2,2,*/
    input_layer_type
    >>>>>>>>>>>/*>>>>>*/;

// inference network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::affine<dlib::cont<default_class_count,5,5,1,1,/*ares<64,*/
    alevel4t<alevel3t<alevel2t<alevel1t<
    alevel1<alevel2<alevel3<alevel4<
    /*dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<64,5,5,2,2,*/
    input_layer_type
    >>>>>>>>>>>/*>>>>>*/;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        //count = Inputs<1,Inputs<4,W,3,2>::count,5,2>::count
        count = Inputs<4,W,3,2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<4,W,3,2>::count
    };
};

//static_assert(NetInputs<1>::count == 67, "Unexpected net input count");
static_assert(NetInputs<1>::count == 31, "Unexpected net input count");
#endif

#if 0
// this one is quick to compile

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::bn_con<dlib::cont<default_class_count,5,5,1,1,
    level4t<
    level4<    
    input_layer_type
    >>>>>;

// inference network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::affine<dlib::cont<default_class_count,5,5,1,1,
    alevel4t<
    alevel4<    
    input_layer_type
    >>>>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<1,W,3,2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<1,W,3,2>::count
    };
};

//static_assert(NetInputs<1>::count == 11, "Unexpected net input count");
#endif

#endif

#if 0

template <int N, int K, int S, typename SUBNET> using down = dlib::con<N, K, K, S, S, SUBNET>;
template <int N, int K, int S, typename SUBNET> using up = dlib::cont<N, K, K, S, S, SUBNET>;

template <int N, int K, int S, typename SUBNET> using bdown = dlib::bn_con<down<N, K, S, SUBNET>>;
template <int N, int K, int S, typename SUBNET> using adown = dlib::affine<down<N, K, S, SUBNET>>;
template <int N, int K, int S, typename SUBNET> using bdownrelu = dlib::relu<bdown<N, K, S, SUBNET>>;
template <int N, int K, int S, typename SUBNET> using adownrelu = dlib::relu<adown<N, K, S, SUBNET>>;

template <int N, int K, int S, typename SUBNET> using bup = dlib::bn_con<up<N, K, S, SUBNET>>;
template <int N, int K, int S, typename SUBNET> using aup = dlib::affine<up<N, K, S, SUBNET>>;
template <int N, int K, int S, typename SUBNET> using buprelu = dlib::relu<bup<N, K, S, SUBNET>>;
template <int N, int K, int S, typename SUBNET> using auprelu = dlib::relu<aup<N, K, S, SUBNET>>;

using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
                    bup<default_class_count,5,2,buprelu<8,3,2,buprelu<32,7,3,buprelu<64,7,3,buprelu<128,7,3,
                    bdownrelu<128,7,3,bdownrelu<64,7,3,bdownrelu<32,7,3,dlib::max_pool<3,3,2,2,bdownrelu<8,5,2,
                    input_layer_type>>>>>>>>>>>;

using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
                    aup<default_class_count,5,2,auprelu<8,3,2,auprelu<32,7,3,auprelu<64,7,3,auprelu<128,7,3,
                    adownrelu<128,7,3,adownrelu<64,7,3,adownrelu<32,7,3,dlib::max_pool<3,3,2,2,adownrelu<8,5,2,
                    input_layer_type>>>>>>>>>>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<1,Inputs<1,Inputs<3,W,7,3>::count,3,2>::count,5,2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<3,Outputs<1,Outputs<1,W,5,2>::count,3,2>::count,7,3>::count
    };
};

static_assert(NetInputs<1>::count == 321, "Unexpected net input count");

#endif

#if 0

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::bn_con<dlib::cont<default_class_count, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<64, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<128, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::cont<256, 5, 5, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<512, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<2048, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<512, 5, 5, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<256, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::con<128, 7, 7, 3, 3,
    dlib::relu<dlib::bn_con<dlib::con<64, 7, 7, 3, 3,
    input_layer_type
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::affine<dlib::cont<default_class_count, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<64, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<128, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::cont<256, 5, 5, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<512, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<2048, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<512, 5, 5, 2, 2,
    dlib::relu<dlib::affine<dlib::con<256, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::con<128, 7, 7, 3, 3,
    dlib::relu<dlib::affine<dlib::con<64, 7, 7, 3, 3,
    input_layer_type
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<3, Inputs<1, Inputs<1, W, 3, 2>::count, 5, 2>::count, 7, 3>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<1, Outputs<1, Outputs<3, W, 7, 3>::count, 5, 2>::count, 3, 2>::count
    };
};

static_assert(NetInputs<1>::count == 295, "Unexpected net input count");

#endif

#if 1

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::bn_con<dlib::cont<default_class_count, 5, 5, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<96, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<80, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<64, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<48, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<32, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<16, 5, 5, 2, 2,
    input_layer_type
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::affine<dlib::cont<default_class_count, 5, 5, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<96, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<96, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<80, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<64, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<48, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<32, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<16, 5, 5, 2, 2,
    input_layer_type
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<1, Inputs<1, Inputs<5, W, 3, 2>::count, 3, 2>::count, 5, 2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<5, Outputs<1, Outputs<1, W, 5, 2>::count, 3, 2>::count, 3, 2>::count
    };
};

static_assert(NetInputs<1>::count == 257, "Unexpected net input count");

#endif

#if 0

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::bn_con<dlib::cont<default_class_count, 5, 5, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<16, 5, 5, 2, 2,
    input_layer_type
    >>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::affine<dlib::cont<default_class_count, 5, 5, 2, 2,
    dlib::relu<dlib::affine<dlib::con<16, 5, 5, 2, 2,
    input_layer_type
    >>>>>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<1, W, 5, 2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<1, W, 5, 2>::count
    };
};

//static_assert(NetInputs<1>::count == 63, "Unexpected net input count");

#endif

#if 0

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::bn_con<dlib::cont<default_class_count, 7, 7, 2, 2,
    res<16,
    dlib::relu<dlib::bn_con<dlib::cont<32, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<64, 3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::cont<128, 7, 7, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<64, 7, 7, 2, 2,
    res_down<32,
    dlib::max_pool<3, 3, 2, 2,
    dlib::relu<dlib::bn_con<dlib::con<8, 7, 7, 2, 2,
    input_layer_type
    >>>>>>>>>>>>>>>>>>>>>;

// inference network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel_weighted<
    dlib::affine<dlib::cont<default_class_count, 7, 7, 2, 2,
    ares<16,
    dlib::relu<dlib::affine<dlib::cont<32, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<64, 3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::cont<128, 7, 7, 2, 2,
    dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    ares_down<32,
    dlib::max_pool<3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<8, 7, 7, 2, 2,
    input_layer_type
    >>>>>>>>>>>>>>>>>>>>>;

// The definitions below need to match the network architecture above
template<int W>
struct NetInputs {
    enum {
        count = Inputs<1, Inputs<2, Inputs<1, W, 7, 2>::count, 3, 2>::count, 7, 2>::count
    };
};
template<int W>
struct NetOutputs {
    enum {
        count = Outputs<1, Outputs<2, Outputs<1, W, 7, 2>::count, 3, 2>::count, 7, 2>::count
    };
};

static_assert(NetInputs<1>::count == 67, "Unexpected net input count");

#endif

#endif // __INTELLISENSE__

// ----------------------------------------------------------------------------------------
