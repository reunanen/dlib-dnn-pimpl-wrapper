#pragma once

#include <dlib/dnn.h>

class NetPimpl
{
public:
    typedef dlib::matrix<float> input_type;
    typedef dlib::matrix<float> training_label_type;
    typedef dlib::matrix<float> output_type;
    typedef dlib::sgd solver_type;

    NetPimpl();
    virtual ~NetPimpl();

    void InitializeForTraining(const solver_type& solver = dlib::sgd(0.0001, 0.9));

    void SetLearningRate(double learningRate);
    void SetMinLearningRate(double minLearningRate);
    void SetIterationsWithoutProgressThreshold(unsigned long threshold);
    void SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs = std::chrono::minutes(15));

    void StartTraining(const std::vector<input_type>& inputs, const std::vector<training_label_type>& training_labels);
    void GetNet();

    output_type operator() (const input_type& input) const;

    void Serialize(std::ostream& out) const;
    void Deserialize(std::istream& in) const;

private:
    struct Impl;
    Impl* pimpl;
};
