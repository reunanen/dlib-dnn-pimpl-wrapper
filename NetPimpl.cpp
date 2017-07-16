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
    pimpl->net = std::make_unique<net_type>();
    pimpl->trainer = std::make_unique<dlib::dnn_trainer<net_type, solver_type>>(*pimpl->net, solver);

    //const std::vector<dlib::matrix<float>> sample{ dlib::matrix<float>(561, 561) };
    //pimpl->trainer->train_one_step(sample, sample);
}

void TrainingNet::SetLearningRate(double learningRate)
{
    pimpl->trainer->set_learning_rate(learningRate);
}

void TrainingNet::SetIterationsWithoutProgressThreshold(unsigned long threshold)
{
    pimpl->trainer->set_iterations_without_progress_threshold(threshold);
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

void TrainingNet::StartTraining(const std::vector<input_type>& inputs, const std::vector<training_label_type>& training_labels)
{
    pimpl->trainer->train_one_step(inputs, training_labels);
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
    pimpl->anet = trainingNet.pimpl->trainer->get_net(); // may block
    return *this;
}

output_type RuntimeNet::operator() (const input_type& input) const
{
    return pimpl->anet(input);
}

std::pair<output_type, std::vector<float>> RuntimeNet::Process(const input_type& input) const
{
    std::pair<output_type, std::vector<float>> result;

    result.first = pimpl->anet(input);

    const auto& bottleneckOutput = dlib::layer<12>(pimpl->anet).get_output();
    if (bottleneckOutput.k() != 256 || bottleneckOutput.num_samples() != 1 || bottleneckOutput.nr() != 1 || bottleneckOutput.nc() != 1) {
        std::ostringstream oss;
        oss << "Unexpected bottleneck output size:" << std::endl
            << " - num_samples = " << bottleneckOutput.num_samples() << std::endl
            << " - k           = " << bottleneckOutput.k() << std::endl
            << " - nr          = " << bottleneckOutput.nr() << std::endl
            << " - nc          = " << bottleneckOutput.nc();
        throw std::runtime_error(oss.str());
    }

    const float* out = bottleneckOutput.host();
    result.second.resize(bottleneckOutput.k());

    for (long k = 0, end = bottleneckOutput.k(); k < end; ++k) {
        result.second[k] = out[k];
    }

    return result;
}

void RuntimeNet::Serialize(std::ostream& out) const
{
    dlib::serialize(pimpl->anet, out);
}

void RuntimeNet::Deserialize(std::istream& in)
{
    dlib::deserialize(pimpl->anet, in);
}

}