#include "NetPimpl.h"
#include "NetStructure.h"

struct NetPimpl::Impl
{
    Impl(const NetPimpl::solver_type& solver)
        : trainer(net, solver)
    {}

    net_type net;
    dlib::dnn_trainer<net_type> trainer;

    anet_type anet;

    bool dirty = false; // Has training been started, but the updated net not copied to anet yet?
};

NetPimpl::NetPimpl(const NetPimpl::solver_type& solver)
{
    pimpl = new NetPimpl::Impl(solver);
    pimpl->trainer.be_verbose(); // TODO: remove
}

NetPimpl::~NetPimpl()
{
    delete pimpl;
}

void NetPimpl::SetLearningRate(double learningRate)
{
    pimpl->trainer.set_learning_rate(learningRate);
}

void NetPimpl::SetMinLearningRate(double minLearningRate)
{
    pimpl->trainer.set_min_learning_rate(minLearningRate);
}

void NetPimpl::SetIterationsWithoutProgressThreshold(unsigned long threshold)
{
    pimpl->trainer.set_iterations_without_progress_threshold(threshold);
}

void NetPimpl::SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs)
{
    pimpl->trainer.set_synchronization_file(filename, time_between_syncs);
}

void NetPimpl::StartTraining(const std::vector<NetPimpl::input_type>& inputs, const std::vector<NetPimpl::training_label_type>& training_labels)
{
    assert(!pimpl->dirty);
    pimpl->trainer.train_one_step(inputs, training_labels);
    pimpl->dirty = true;
}

bool NetPimpl::IsTrainingStarted() const
{
    return pimpl->dirty;
}

bool NetPimpl::IsStillTraining() const
{
    assert(pimpl->dirty);
    return pimpl->trainer.is_training();
}

bool NetPimpl::IsTraining() const
{
    return pimpl->dirty && pimpl->trainer.is_training();
}

void NetPimpl::WaitForTrainingToFinishAndUseNet()
{
    if (pimpl->dirty) {
        pimpl->trainer.get_net();
        pimpl->anet = pimpl->net;
        pimpl->dirty = false;
    }
}

NetPimpl::output_type NetPimpl::operator() (const NetPimpl::input_type& input) const
{
    return pimpl->anet(input);
}
