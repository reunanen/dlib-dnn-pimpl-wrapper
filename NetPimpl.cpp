#include "NetPimpl.h"
#include "NetStructure.h"

namespace NetPimpl {

struct TrainingNet::Impl
{
    std::unique_ptr<net_type> net;
    std::unique_ptr<dlib::dnn_trainer<net_type, solver_type>> trainer;
};

struct RuntimeNet::Impl
{
    anet_type anet;
    dlib::resizable_tensor temp_input;
};

TrainingNet::TrainingNet()
{
    pimpl = new TrainingNet::Impl();
}

TrainingNet::~TrainingNet()
{
    delete pimpl;
}

void TrainingNet::Initialize(const solver_type& solver, const std::vector<int> extraDevices, std::shared_ptr<ThreadPools> threadPools)
{
    if (pimpl->trainer) {
        pimpl->trainer->get_net(dlib::force_flush_to_disk::no); // may block
    }
    pimpl->net = std::make_unique<net_type>();
    pimpl->trainer = std::make_unique<dlib::dnn_trainer<net_type, solver_type>>(*pimpl->net, solver, extraDevices, threadPools);
}

void TrainingNet::SetLearningRate(double learningRate)
{
    pimpl->trainer->set_learning_rate(learningRate);
}

void TrainingNet::SetIterationsWithoutProgressThreshold(unsigned long threshold)
{
    pimpl->trainer->set_iterations_without_progress_threshold(threshold);
}

void TrainingNet::SetPreviousLossValuesDumpAmount(unsigned long dump_amount)
{
    pimpl->trainer->set_previous_loss_values_dump_amount(dump_amount);
}

void TrainingNet::SetAllBatchNormalizationRunningStatsWindowSizes(unsigned long window_size)
{
    dlib::set_all_bn_running_stats_window_sizes(*pimpl->net, window_size);
}

void TrainingNet::SetLearningRateShrinkFactor(double learningRateShrinkFactor)
{
    pimpl->trainer->set_learning_rate_shrink_factor(learningRateShrinkFactor);
}

void TrainingNet::SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs)
{
    pimpl->trainer->set_synchronization_file(filename, time_between_syncs);
}

class SetNetWidthVisitor
{
public:
    SetNetWidthVisitor(double scaler, int minFilterCount)
        : scaler(scaler)
        , minFilterCount(minFilterCount)
    {}

    template <typename T>
    void SetNetWidth(T&) const
    {
        // ignore other layer detail types
    }

    template <long num_filters, long nr, long nc, int stride_y, int stride_x, int padding_y, int padding_x>
    void SetNetWidth(dlib::con_<num_filters, nr, nc, stride_y, stride_x, padding_y, padding_x>& l) const
    {
        SetFilterCount(l);
    }

    template <long num_filters, long nr, long nc, int stride_y, int stride_x, int padding_y, int padding_x>
    void SetNetWidth(dlib::cont_<num_filters, nr, nc, stride_y, stride_x, padding_y, padding_x>& l) const
    {
        SetFilterCount(l);
    }

    template <typename L>
    void SetFilterCount(L& l) const
    {
        l.set_num_filters(GetNewFilterCount(l.num_filters()));
    }

    int GetNewFilterCount(int currentFilterCount) const
    {
        return std::max(minFilterCount, static_cast<int>(std::round(scaler * currentFilterCount)));
    }

    template<typename input_layer_type>
    void operator()(size_t, input_layer_type&) const
    {
        // ignore other layers
    }

    template <typename T, typename U, typename E>
    void operator()(size_t, dlib::add_layer<T, U, E>& l) const
    {
        SetNetWidth(l.layer_details());
    }

private:
    double scaler;
    int minFilterCount;
};

void TrainingNet::SetNetWidth(double scaler, int minFilterCount)
{
    dlib::visit_layers(*pimpl->net, SetNetWidthVisitor(scaler, minFilterCount));

    // Revert the number of filters in the output layer
    pimpl->net->subnet().layer_details().set_num_filters(1);
}

void TrainingNet::BeVerbose()
{
    pimpl->trainer->be_verbose();
}

int TrainingNet::GetRequiredInputDimension()
{
    constexpr int startingPoint = 225; // A rather arbitrary selection
    constexpr int testInputDim = NetInputs<startingPoint>::count;
    constexpr int testOutputDim = NetOutputs<testInputDim>::count;
    static_assert(testOutputDim == startingPoint, "I/O dimension mismatch detected");

    constexpr int inputDim = NetInputs<1>::count;
    return inputDim;
}

void TrainingNet::StartTraining(const std::vector<input_type>& inputs, const std::vector<training_label_type>& training_labels)
{
    pimpl->trainer->train_one_step(inputs, training_labels);
}

double TrainingNet::GetLearningRate() const
{
    return pimpl->trainer->get_learning_rate();
}

RuntimeNet TrainingNet::GetRuntimeNet() const
{
    RuntimeNet runtimeNet;
    runtimeNet = *this; // there's only assignment operator - no copy constructor (at least for now)
    return runtimeNet;
}

void TrainingNet::Serialize(std::ostream& out) const
{
    dlib::serialize(*pimpl->net, out);
}

void TrainingNet::Deserialize(std::istream& in)
{
    dlib::deserialize(*pimpl->net, in);
}

void TrainingNet::Serialize(const std::string& filename) const
{
    std::ofstream ofs(filename, std::ios::binary);
    Serialize(ofs);
}

void TrainingNet::Deserialize(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    Deserialize(ifs);
}

std::string TrainingNet::GetNetDescription() const
{
    std::ostringstream oss;
    pimpl->trainer->get_net(dlib::force_flush_to_disk::no); // may block
    oss << *pimpl->net;
    return oss.str();
}

RuntimeNet::RuntimeNet()
{
    pimpl = new RuntimeNet::Impl();
    //std::cout << pimpl->anet << std::endl;

#if 0
    pimpl->anet(dlib::matrix<float>(561, 561));
    const auto& output6 = dlib::layer<6>(pimpl->anet).get_output();
    if (output6.num_samples() == 1 && output6.nr() == 1 && output6.nc() == 1 && output6.k() == 64) {
        ; // ok!
    }
    else {
        std::ostringstream oss;
        oss << "Unexpected output size from layer 6:" << std::endl
            << " - num_samples = " << output6.num_samples() << std::endl
            << " - nr          = " << output6.nr() << std::endl
            << " - nc          = " << output6.nc() << std::endl
            << " - k           = " << output6.k() << std::endl;
    }
#endif
}

RuntimeNet::~RuntimeNet()
{
    delete pimpl;
}

RuntimeNet::RuntimeNet(const RuntimeNet& that)
{
    pimpl = new RuntimeNet::Impl();
    operator= (that);
}

RuntimeNet& RuntimeNet::operator= (const RuntimeNet& that)
{
    pimpl->anet = that.pimpl->anet;
    return *this;
}

RuntimeNet& RuntimeNet::operator= (const TrainingNet& trainingNet)
{
    auto& net = trainingNet.pimpl->trainer->get_net(dlib::force_flush_to_disk::no); // may block
    net.clean();
    pimpl->anet = net;
    return *this;
}

output_type RuntimeNet::operator() (const input_type& input) const
{
    return pimpl->anet.process(input);
}

const dlib::tensor& RuntimeNet::GetOutput() const
{
    return pimpl->anet.subnet().get_output();
}

// see: https://stackoverflow.com/a/3499919/19254

const int MAX_OUTPUT_COUNT_FOR_CALCULATING_RECOMMENDED_INPUT_DIMENSION = 500;

template <int N, int OutputCount = N - 1>
class OutputDimensionToInputDimension : public OutputDimensionToInputDimension<N, OutputCount - 1>
{
public:
    static const int dummy;
};

template <int N>
class OutputDimensionToInputDimension<N, 0>
{
public:
    static const int dummy = 0;
    static int array[N];
};

template <int N, int OutputCount>
const int OutputDimensionToInputDimension<N, OutputCount>::dummy = OutputDimensionToInputDimension<N, 0>::array[OutputCount] = NetInputs<OutputCount>::count + 0 * OutputDimensionToInputDimension<N, OutputCount - 1>::dummy;

template <int N>
int OutputDimensionToInputDimension<N, 0>::array[N];

template class OutputDimensionToInputDimension<MAX_OUTPUT_COUNT_FOR_CALCULATING_RECOMMENDED_INPUT_DIMENSION>;

int RuntimeNet::GetRecommendedInputDimension(int minimumInputDimension)
{
    const int *outputDimensionToInputDimension = OutputDimensionToInputDimension<MAX_OUTPUT_COUNT_FOR_CALCULATING_RECOMMENDED_INPUT_DIMENSION>::array;

    for (size_t outputCount = 1; outputCount < MAX_OUTPUT_COUNT_FOR_CALCULATING_RECOMMENDED_INPUT_DIMENSION; ++outputCount) {
        int inputDimension = outputDimensionToInputDimension[outputCount];
        if (inputDimension >= minimumInputDimension) {
            return inputDimension;
        }
    }
    
    std::ostringstream error;
    error << "Requested minimum input dimension " << minimumInputDimension << " is too large (the largest supported is " << outputDimensionToInputDimension[MAX_OUTPUT_COUNT_FOR_CALCULATING_RECOMMENDED_INPUT_DIMENSION - 1] << ")";
    throw std::runtime_error(error.str());
}

output_type RuntimeNet::Process(const input_type& input) const
{
    return pimpl->anet.process(input);
}

const dlib::tensor& RuntimeNet::Forward(const input_type& input) const
{
    auto& subnet = pimpl->anet.subnet();
    subnet.to_tensor(&input, &input + 1, pimpl->temp_input);
    subnet.forward(pimpl->temp_input);
    return subnet.get_output();
}

void RuntimeNet::Serialize(std::ostream& out) const
{
    dlib::serialize(pimpl->anet, out);
}

void RuntimeNet::Deserialize(std::istream& in)
{
    dlib::deserialize(pimpl->anet, in);
}

void RuntimeNet::Serialize(const std::string& filename) const
{
    std::ofstream ofs(filename, std::ios::binary);
    Serialize(ofs);
}

void RuntimeNet::Deserialize(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    Deserialize(ifs);
}

std::string RuntimeNet::GetNetDescription() const
{
    std::ostringstream oss;
    oss << pimpl->anet;
    return oss.str();
}

}
