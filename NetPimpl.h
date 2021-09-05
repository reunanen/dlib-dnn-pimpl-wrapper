#pragma once

#include <dlib/dnn.h>
#include <vector>
#include "MemoryManager.h"

namespace NetPimpl
{
#ifdef DLIB_DNN_PIMPL_WRAPPER_GRAYSCALE_INPUT
    // TODO: use definition from MemoryManager.h
    typedef dlib::matrix<uint8_t,0,0,dlib::memory_manager_stateless<uint8_t>::kernel_2_3e> input_type;
#else
    typedef dlib::matrix<dlib::rgb_pixel,0,0,dlib::memory_manager_stateless<uint8_t>::kernel_2_3e> input_type;
#endif
    typedef dlib::loss_multiclass_log_per_pixel_weighted_::training_label_type training_label_type;
    typedef dlib::loss_multiclass_log_per_pixel_weighted_::output_label_type output_type;
#if 0
    typedef dlib::adam solver_type;
    const auto GetDefaultSolver = []() { return dlib::adam(0.001, 0.9, 0.999); };
#else
    typedef dlib::sgd solver_type;
    const auto GetDefaultSolver = []() { return dlib::sgd(0.001, 0.9); };
#endif

    typedef std::vector<std::shared_ptr<dlib::thread_pool>> ThreadPools;

    class RuntimeNet;

    class TrainingNet {
    public:
        TrainingNet();
        virtual ~TrainingNet();

        void Initialize(
            const solver_type& solver = GetDefaultSolver(),
            const std::vector<int> extraDevices = std::vector<int>(),
            std::shared_ptr<ThreadPools> threadPools = std::shared_ptr<ThreadPools>()
        );

        void SetClassCount(unsigned short classCount);
        void SetLearningRate(double learningRate);
        void SetIterationsWithoutProgressThreshold(unsigned long threshold);
        void SetPreviousLossValuesDumpAmount(unsigned long dump_amount);
        void SetAllBatchNormalizationRunningStatsWindowSizes(unsigned long window_size);
        void SetLearningRateShrinkFactor(double learningRateShrinkFactor);
        void SetSynchronizationFile(const std::string& filename, std::chrono::seconds time_between_syncs = std::chrono::minutes(15));
        void SetNetWidth(double scaler, int minFilterCount);
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

        output_type operator() (const input_type& input, const std::vector<double>& gainFactors = std::vector<double>()) const;

        const dlib::tensor& GetOutput() const;

        static int GetRecommendedInputDimension(int minimumInputDimension);

        output_type Process(const input_type& input, const std::vector<double>& gainFactors = std::vector<double>()) const;

        const dlib::tensor& Forward(const input_type& input) const;

        void Serialize(std::ostream& out) const;
        void Deserialize(std::istream& in);

        void Serialize(const std::string& filename) const;
        void Deserialize(const std::string& filename);

    private:
        struct Impl;
        Impl* pimpl;
    };

    int GetLevelCount();
};
