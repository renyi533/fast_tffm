#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>
#include <unordered_map>

REGISTER_OP("FmSparseTensorParser")
    .Input("segment_ids: int32")
    .Input("feature_ids: int64")
    .Output("ori_ids: int64")
    .Output("local_feature_ids: int32")
    .Output("feature_poses: int32")
    .Attr("vocab_size: int64");

using namespace tensorflow;

class FmSparseTensorParserOp : public OpKernel {
 public:

  explicit FmSparseTensorParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* segment_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("segment_ids", &segment_id_tensor));
    auto segment_ids = segment_id_tensor->flat<int32>();

    const Tensor* feature_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_ids", &feature_id_tensor));
    auto feature_ids = feature_id_tensor->flat<int64>();


    std::unordered_map<int64, int32> ori_id_map;
    std::vector<int32> local_feature_ids;
    std::vector<int32> feature_poses;

    int32 curr_seg_id = -1;
    for (int32 i = 0; i < segment_ids.size(); ++i) {
      if (curr_seg_id != segment_ids(i)) {
        feature_poses.push_back(i);
        curr_seg_id =  segment_ids(i);
      }
    }

    for (size_t i = 0; i < feature_ids.size(); ++i) {
      auto ori_id = vocab_size_ > 0 ? (feature_ids(i) % vocab_size_) : feature_ids(i);
      auto iter = ori_id_map.find(ori_id);
      int32 fid = 0;
      if (iter == ori_id_map.end()) {
        fid = ori_id_map.size();
        ori_id_map[ori_id] = fid;
      } else {
        fid = iter->second;
      }
      local_feature_ids.push_back(fid);
    }

    std::vector<int64> ori_ids(ori_id_map.size(), 0);
    for (auto it = ori_id_map.begin(); it != ori_id_map.end(); ++it) {
      ori_ids[it->second] = it->first;
    }

    AllocateTensorForVector<int64>(ctx, "ori_ids", ori_ids);
    AllocateTensorForVector<int32>(ctx, "local_feature_ids", local_feature_ids);
    AllocateTensorForVector<int32>(ctx, "feature_poses", feature_poses);
  }

 private:
  int64 vocab_size_;

  template<typename T>
  void AllocateTensorForVector(OpKernelContext* ctx, const string& name, const std::vector<T>& data) {
    Tensor* tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(name, TensorShape({static_cast<int64>(data.size())}), &tensor));
    auto tensor_data = tensor->flat<T>();
    for (size_t i = 0; i < data.size(); ++i) {
      tensor_data(i) = data[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FmSparseTensorParser").Device(DEVICE_CPU), FmSparseTensorParserOp);
