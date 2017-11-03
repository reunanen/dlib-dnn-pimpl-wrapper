#pragma once

#include <dlib/dnn.h>
#include <vector>

namespace NetPimpl
{
    typedef dlib::matrix<uint8_t> input_type;
    typedef dlib::matrix<uint16_t> training_label_type;
    typedef dlib::matrix<uint16_t> output_type;
#if 0
    typedef dlib::adam solver_type;
    const auto GetDefaultSolver = []() { return dlib::adam(0.001, 0.9, 0.999); };
#else
    typedef dlib::sgd solver_type;
    const auto GetDefaultSolver = []() { return dlib::sgd(0.001, 0.9); };
#endif

    class RuntimeNet;

    class TrainingNet {
    public:
        TrainingNet();
        virtual ~TrainingNet();

        void Initialize(const solver_type& solver = GetDefaultSolver());

        void SetClassCount(unsigned short classCount);
        void SetLearningRate(double learningRate);
        void SetIterationsWithoutProgressThreshold(unsigned long threshold);
        void SetPreviousLossValuesDumpAmount(unsigned long dump_amount);
        void SetAllBatchNormalizationRunningStatsWindowSizes(unsigned long window_size);
        void SetLearningRateShrinkFactor(double learningRateShrinkFactor);
        void SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs = std::chrono::minutes(15));
        void BeVerbose();

        static int GetRequiredInputDimension();
        void StartTraining(const std::vector<input_type>& inputs, const std::vector<training_label_type>& training_labels);

        double GetLearningRate() const;

        RuntimeNet GetRuntimeNet() const; // may block

        void Serialize(std::ostream& out) const;
        void Deserialize(std::istream& in);

        void Serialize(const std::string& filename) const;
        void Deserialize(const std::string& filename);

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

        output_type Process(const input_type& input) const;

        void Serialize(std::ostream& out) const;
        void Deserialize(std::istream& in);

        void Serialize(const std::string& filename) const;
        void Deserialize(const std::string& filename);

    private:
        struct Impl;
        Impl* pimpl;
    };
};
