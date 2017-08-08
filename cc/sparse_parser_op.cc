#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>

REGISTER_OP("SparseBinaryParser")
    .Input("records: string")
    .Output("labels: float32")
    .Output("indices: int64")
    .Output("fvals: float32")
    .Output("fids: int64")
    .Output("dense_shape: int64");


using namespace tensorflow;

class SparseBinaryParserOp : public OpKernel {
  public:

  explicit SparseBinaryParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const tensorflow::Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    auto records_t = records->flat<std::string>();

    std::vector<std::vector<int64>> indices;
    std::vector<float> fvals;
    std::vector<int64> fids;
    std::vector<int64> dense_shape;
    std::vector<std::vector<float>> labels;
    int max_feat_count = 0;
    for (size_t i = 0; i < records_t.size(); ++i) {
      std::vector<float> label;
      const std::string& in_str = records_t(i);
      int feature_count = ExtractCurrRecord(ctx, in_str, label, fvals, fids, indices, i);
      max_feat_count = feature_count > max_feat_count ? feature_count : max_feat_count;
      labels.push_back(label);
    }
    dense_shape.push_back(records_t.size());
    dense_shape.push_back(max_feat_count);
    AllocateTensorFor2DVector<float>(ctx, "labels", labels);
    AllocateTensorFor2DVector<int64>(ctx, "indices", indices);
    AllocateTensorForVector<float>(ctx, "fvals", fvals);
    AllocateTensorForVector<int64>(ctx, "fids", fids);
    AllocateTensorForVector<int64>(ctx, "dense_shape", dense_shape);
  }

 private:

  int ExtractCurrRecord(OpKernelContext* ctx, const string& str, std::vector<float>& label, std::vector<float>& fvals, 
		  std::vector<int64>& fids, std::vector<std::vector<int64>>& indices, int sample_idx) {
    char* p = (char*)str.c_str();
    tensorflow::int64* int64p = (tensorflow::int64*)p;
    tensorflow::int64 header = *int64p;
    //OP_REQUIRES_OK(ctx, header == 0x7FFFFFFF7FFFFFFF);
    p += 8;
    tensorflow::uint32* u32p = (tensorflow::uint32*)p;
    tensorflow::uint32 feature_len = *u32p;
    p += 4;
    u32p = (tensorflow::uint32*)p;
    tensorflow::uint32 label_len = *u32p;
    p += 4;
    std::vector<int64> curr_idx;
    curr_idx.push_back(sample_idx);
    curr_idx.push_back(0);
    for (int i=0; i< feature_len; i++)
    {
      int64p =(tensorflow::int64*) p;
      tensorflow::int64 fid = *int64p;
      fids.push_back(fid);
      
      curr_idx[1] = i;
      indices.push_back(curr_idx);
      //OP_REQUIRES_OK(ctx, fid<feature_dimension_);
      p += 8;
      float* p_val = (float*)p;
      p += 4;
      fvals.push_back(*p_val);    
    }

    for (int i=0; i< label_len; i++)
    {
      float* p_val = (float*)p;
      p += 4;
      label.push_back(*p_val);
    }
    return (int)feature_len;
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

REGISTER_KERNEL_BUILDER(Name("SparseBinaryParser").Device(tensorflow::DEVICE_CPU), SparseBinaryParserOp);
