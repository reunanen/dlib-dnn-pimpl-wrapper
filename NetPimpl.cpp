#include "NetPimpl.h"
#include "NetStructure.h"
#include "NetDimensions.h"

namespace NetPimpl {

struct TrainingNet::Impl
{
    std::unique_ptr<net_type> net;
    std::unique_ptr<dlib::dnn_trainer<net_type, solver_type>> trainer;
};

struct RuntimeNet::Impl
{
    anet_type anet;
};

TrainingNet::TrainingNet()
{
    pimpl = new TrainingNet::Impl();
}

TrainingNet::~TrainingNet()
{
    delete pimpl;
}

void TrainingNet::Initialize(const solver_type& solver)
{
    if (pimpl->trainer) {
        pimpl->trainer->get_net(dlib::force_flush_to_disk::no); // may block
    }
    pimpl->net = std::make_unique<net_type>();
    pimpl->trainer = std::make_unique<dlib::dnn_trainer<net_type, solver_type>>(*pimpl->net, solver);
}

void TrainingNet::SetClassCount(unsigned short classCount)
{
    DLIB_CASSERT(classCount < dlib::loss_multiclass_log_per_pixel_::label_to_ignore);
    pimpl->net->subnet().subnet().layer_details().set_num_filters(classCount);
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
    pimpl->trainer->set_min_learning_rate(learningRateShrinkFactor);
}

void TrainingNet::SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs)
{
    pimpl->trainer->set_synchronization_file(filename, time_between_syncs);
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
    Serialize(std::ofstream(filename, std::ios::binary));
}

void TrainingNet::Deserialize(const std::string& filename)
{
    Deserialize(std::ifstream(filename, std::ios::binary));
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
    pimpl->anet = trainingNet.pimpl->trainer->get_net(dlib::force_flush_to_disk::no); // may block
    return *this;
}

output_type RuntimeNet::operator() (const input_type& input) const
{
    return pimpl->anet(input);
}

output_type RuntimeNet::Process(const input_type& input) const
{
    return pimpl->anet(input);
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
    Serialize(std::ofstream(filename, std::ios::binary));
}

void RuntimeNet::Deserialize(const std::string& filename)
{
    Deserialize(std::ifstream(filename, std::ios::binary));
}

}