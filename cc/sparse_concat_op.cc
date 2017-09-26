#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <vector>

REGISTER_OP("SparseConcatenate")
    .Input("indices: N * int64")
    .Input("values: N * int64")
    .Input("shapes: N * int64")
    .Output("output_indices: int64")
    .Output("output_values: int64")
    .Output("output_shape: int64")
    .Attr("N: int >= 2");

using namespace tensorflow;

class SparseConcatOp : public OpKernel {
  public:

  explicit SparseConcatOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    OpInputList inds;
    OP_REQUIRES_OK(context, context->input_list("indices", &inds));
    const int N = inds.size();
    int count = 0;
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(inds[i].shape()),
                  errors::InvalidArgument(
                      "Input indices should be a matrix but received shape ",
                      inds[i].shape().DebugString(), " at position ", i));
      count += inds[i].dim_size(0);
    }

    OpInputList vals;
    OP_REQUIRES_OK(context, context->input_list("values", &vals));
    OP_REQUIRES(context, vals.size() == N,
                errors::InvalidArgument("Expected ", N, " input values, got ",
                                        vals.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(vals[i].shape()),
                  errors::InvalidArgument(
                      "Input values should be a vector but received shape ",
                      vals[i].shape().DebugString(), " at position ", i));

      OP_REQUIRES(context, inds[i].dim_size(0)==vals[i].dim_size(0),
                  errors::InvalidArgument(
                      "indices size:", inds[i].dim_size(0), ", vals size:",
                      vals[i].dim_size(0), " at position ", i));
    }

    OpInputList shapes;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes));
    OP_REQUIRES(context, shapes.size() == N,
                errors::InvalidArgument("Expected ", N, " input shapes, got ",
                                        shapes.size()));
    int shape_dim0 = 0;
    int max_dim1 = 0;
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(shapes[i].shape()),
                  errors::InvalidArgument(
                      "Input shapes should be a vector but received shape ",
                      shapes[i].shape().DebugString(), " at position ", i));
    
      auto curr_shape = shapes[i].vec<int64>();
      const TensorShape input_shape(curr_shape);
      const int input_rank = input_shape.dims();
      OP_REQUIRES(context, input_rank==2,
                  errors::InvalidArgument(
                      "Input shapes should be of dim 2 but received dim ",
                      input_rank, " at position ", i));
      if (shape_dim0 <=0)
      {
        shape_dim0 = curr_shape(0);
      }
      OP_REQUIRES(context, shape_dim0==curr_shape(0),
                  errors::InvalidArgument(
                      "first dim0 ", shape_dim0, "curr dim0",
                      curr_shape(0), " at position ", i));

      max_dim1 = (max_dim1 > curr_shape(1)) ? max_dim1 : curr_shape(1);
    }

    tensorflow::Tensor* output_shape_out;
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({2}),
                                &output_shape_out));
    auto output_shape = output_shape_out->vec<int64>();
    output_shape(0) = shape_dim0 * N;
    output_shape(1) = max_dim1;

    Tensor* output_values_out;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "output_values", TensorShape({count}),
                                &output_values_out));

    auto output_values = output_values_out->vec<int64>();
    
    Tensor* output_indices_out;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "output_indices", TensorShape({count, 2}),
                                &output_indices_out));

    auto output_indices = output_indices_out->flat<int64>();
    
    int c = 0;
    int sample_c = 0;

    for (int j=0; j<N; j++)
    {
      auto indices_vec = inds[j].flat<int64>();
      auto vals_vec = vals[j].vec<int64>();
      int64 curr_index = 0;
      for (int i=0; i<shape_dim0; i++)
      {
        int64 original_index = curr_index;
        while (curr_index < inds[j].dim_size(0) && indices_vec(2*curr_index) == i)
        {
          output_indices(2*c) = sample_c;
          output_indices(2*c+1) = indices_vec(2*curr_index+1);
            
          output_values(c) = vals_vec(curr_index);
          curr_index++;
          c++;
        }
        sample_c++;
        OP_REQUIRES(context, original_index < curr_index,
                    errors::InvalidArgument(
                      "curr_idx:", i, ", element index:", indices_vec(2*curr_index),
                      " , element num:", j, " , curr_index:", curr_index,
                      ", inds[j].dim_size(0):", inds[j].dim_size(0))
                   );
      }
      OP_REQUIRES(context, curr_index == inds[j].dim_size(0),
                    errors::InvalidArgument(
                      "input sp tensor:", j, ", expected cnt:", inds[j].dim_size(0),
                      " , real cnt:", curr_index)
                   );
    }
    
    OP_REQUIRES(context, c == count,
                errors::InvalidArgument(
                      "expected total element: ", count, ", real count:", c));
    OP_REQUIRES(context, sample_c == shape_dim0*N,
                errors::InvalidArgument(
                      "expected total samples: ", shape_dim0*N, ", real sample count:", sample_c));
  }
  

 private:

};

REGISTER_KERNEL_BUILDER(Name("SparseConcatenate").Device(tensorflow::DEVICE_CPU), SparseConcatOp);


