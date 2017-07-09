#pragma once

#include <dlib/dnn.h>

namespace NetPimpl
{
    typedef dlib::matrix<float> input_type;
    typedef dlib::matrix<float> training_label_type;
    typedef dlib::matrix<float> output_type;
    typedef dlib::sgd solver_type;

    class RuntimeNet;

    class TrainingNet {
    public:
        TrainingNet();
        virtual ~TrainingNet();

        void Initialize(const solver_type& solver = dlib::sgd(0.0001, 0.9));

        void SetLearningRate(double learningRate);
        void SetMinLearningRate(double minLearningRate);
        void SetIterationsWithoutProgressThreshold(unsigned long threshold);
        void SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs = std::chrono::minutes(15));
        void BeVerbose();

        void StartTraining(const std::vector<input_type>& inputs, const std::vector<training_label_type>& training_labels);

        RuntimeNet GetRuntimeNet() const; // may block

    private:
        TrainingNet(const TrainingNet&) = delete;
        TrainingNet& operator= (const TrainingNet&) = delete;

        struct Impl;
        Impl* pimpl;

        friend class RuntimeNet;
    };

    class RuntimeNet {
    public:
        RuntimeNet();
        virtual ~RuntimeNet();

        RuntimeNet(const RuntimeNet&);
        RuntimeNet& operator= (const RuntimeNet&);

        RuntimeNet& operator= (const TrainingNet& trainingNet); // may block

        output_type operator() (const input_type& input) const;

        void Serialize(std::ostream& out) const;
        void Deserialize(std::istream& in) const;

    private:
        struct Impl;
        Impl* pimpl;
    };
};
