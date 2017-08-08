#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>

REGISTER_OP("DenseBinaryParser")
    .Input("records: string")
    .Output("labels: float32")
    .Output("features: float32")
    .Attr("feature_dimension: int");


using namespace tensorflow;

class DenseBinaryParserOp : public OpKernel {
  public:

  explicit DenseBinaryParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_dimension", &feature_dimension_));
  }

  void Compute(OpKernelContext* ctx) override {
    const tensorflow::Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    auto records_t = records->flat<std::string>();

    std::vector<std::vector<float>> features;
    std::vector<std::vector<float>> labels;


    for (size_t i = 0; i < records_t.size(); ++i) {
      std::vector<float> label;
      std::vector<float> feature(feature_dimension_, 0.0);
      const std::string& in_str = records_t(i);
      ExtractCurrRecord(ctx, in_str, label, feature);
      features.push_back(feature);
      labels.push_back(label);
    }

    AllocateTensorFor2DVector<float>(ctx, "labels", labels);
    AllocateTensorFor2DVector<float>(ctx, "features", features);
  }

 private:
  int32 feature_dimension_;

  void ExtractCurrRecord(OpKernelContext* ctx, const string& str, std::vector<float>& label, 
		  std::vector<float>& feature)
  {
    char* p = (char*)str.c_str();
    tensorflow::uint64* u64p = (tensorflow::uint64*)p;
    tensorflow::uint64 header = *u64p;
    //OP_REQUIRES_OK(ctx, header == 0x7FFFFFFF7FFFFFFF);
    p += 8;
    tensorflow::uint32* u32p = (tensorflow::uint32*)p;
    tensorflow::uint32 feature_len = *u32p;
    p += 4;
    u32p = (tensorflow::uint32*)p;
    tensorflow::uint32 label_len = *u32p;
    p += 4;
    for (int i=0; i< feature_len; i++)
    {
      u64p =(tensorflow::uint64*) p;
      tensorflow::uint64 fid = *u64p;
      //OP_REQUIRES_OK(ctx, fid<feature_dimension_);
      p += 8;
      float* p_val = (float*)p;
      p += 4;
      feature[(tensorflow::uint32)fid] = *p_val;
    }

    for (int i=0; i< label_len; i++)
    {
      float* p_val = (float*)p;
      p += 4;
      label.push_back(*p_val);
    }
  } 

  template<typename T>
  void AllocateTensorForVector(OpKernelContext* ctx, const string& name, const std::vector<T>& data) {
    tensorflow::Tensor* tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(name, tensorflow::TensorShape({static_cast<tensorflow::int64>(data.size())}), &tensor));
    auto tensor_data = tensor->flat<T>();
    for (size_t i = 0; i < data.size(); ++i) {
      tensor_data(i) = data[i];
    }
  }

  template<typename T>
  void AllocateTensorFor2DVector(OpKernelContext* ctx, const string& name, const std::vector<std::vector<T>>& data) {
    tensorflow::Tensor* tensor;
    size_t m = data.size();
    size_t n = 0;
    if (m>0)
    {	    
    	n = data[0].size();
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(name, tensorflow::TensorShape({(tensorflow::int64)m, (tensorflow::int64)n}), &tensor));
    auto tensor_data = tensor->flat<T>();
    int c = 0;
    for (int i=0; i<m; i++)
    {
      for (int j=0; j<n; j++)
      {
        tensor_data(c++) = data[i][j];
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseBinaryParser").Device(tensorflow::DEVICE_CPU), DenseBinaryParserOp);
