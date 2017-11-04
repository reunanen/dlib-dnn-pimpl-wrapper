#ifndef NET_DIMENSIONS_H
#define NET_DIMENSIONS_H

template<int N, int W, int K, int S, int P = (S != 1 ? 0 : W / 2)>
struct Outputs {
    static_assert((W - K + 2 * P) % S == 0, "Output size must be integer");
    enum {
        count = (Outputs<N - 1, W, K, S, P>::count - K + 2 * P) / S + 1        
    };
};
template<int W, int K, int S, int P>
struct Outputs<0, W, K, S, P> {
    enum {
        count = W
    };
};

template<int N, int W, int K, int S, int P = (S != 1 ? 0 : W / 2)>
struct Inputs {
    enum {
        count = (Inputs<N - 1, W, K, S, P>::count - 1) * S - 2 * P + K
    };
};
template<int W, int K, int S, int P>
struct Inputs<0, W, K, S, P> {
    enum {
        count = W
    };
};

#endif // NET_DIMENSIONS_H