#include "NetPimpl.h"
#include "NetStructure.h"

struct NetPimpl::Impl
{
    std::unique_ptr<net_type> net;
    std::unique_ptr<dlib::dnn_trainer<net_type>> trainer;

    std::unique_ptr<net_type> cleanNet;
    std::unique_ptr<anet_type> anet;
};

NetPimpl::NetPimpl()
{
    pimpl = new NetPimpl::Impl();
}

NetPimpl::~NetPimpl()
{
    delete pimpl;
}

void NetPimpl::InitializeForTraining(const solver_type& solver)
{
    pimpl->net = std::make_unique<net_type>();
    pimpl->trainer = std::make_unique<dlib::dnn_trainer<net_type>>(*pimpl->net, solver);
    pimpl->trainer->be_verbose(); // TODO: remove
}

void NetPimpl::SetLearningRate(double learningRate)
{
    pimpl->trainer->set_learning_rate(learningRate);
}

void NetPimpl::SetMinLearningRate(double minLearningRate)
{
    pimpl->trainer->set_min_learning_rate(minLearningRate);
}

void NetPimpl::SetIterationsWithoutProgressThreshold(unsigned long threshold)
{
    pimpl->trainer->set_iterations_without_progress_threshold(threshold);
}

void NetPimpl::SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs)
{
    pimpl->trainer->set_synchronization_file(filename, time_between_syncs);
}

void NetPimpl::StartTraining(const std::vector<NetPimpl::input_type>& inputs, const std::vector<NetPimpl::training_label_type>& training_labels)
{
    pimpl->trainer->train_one_step(inputs, training_labels);
}

void NetPimpl::GetNet()
{
    pimpl->trainer->get_net();
    if (!pimpl->cleanNet.get()) {
        pimpl->cleanNet = std::make_unique<net_type>();
    }
    *pimpl->cleanNet = *pimpl->net;
    pimpl->cleanNet->clean();
    if (!pimpl->anet.get()) {
        pimpl->anet = std::make_unique<anet_type>();
    }
    *pimpl->anet = *pimpl->cleanNet;
}

NetPimpl::output_type NetPimpl::operator() (const NetPimpl::input_type& input) const
{
    if (!pimpl->anet.get()) {
        pimpl->anet = std::make_unique<anet_type>();
    }
    return (*pimpl->anet)(input);
}

void NetPimpl::Serialize(std::ostream& out) const
{
    if (!pimpl->anet.get()) {
        pimpl->anet = std::make_unique<anet_type>();
    }
    dlib::serialize(*pimpl->anet, out);
}

void NetPimpl::Deserialize(std::istream& in) const
{
    if (!pimpl->anet.get()) {
        pimpl->anet = std::make_unique<anet_type>();
    }
    dlib::deserialize(*pimpl->anet, in);
}