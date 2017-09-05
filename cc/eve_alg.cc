#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <ctime>
#include <algorithm>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include <math.h>
#include <iostream>

REGISTER_OP("EveAlg")
    .Input("f: float32")
    .Input("smoothed_f: float32")
    .Input("d: float32")
    .Input("target: float32")
    .Input("step: int64")
    .Input("decay_rate: float32")
    .Input("f_decay_rate: float32")
    .Input("lower: float32")
    .Input("upper: float32")
    .Output("out_f: float32")
    .Output("out_smoothed_f: float32")
    .Output("out_d: float32");

using namespace tensorflow;

class EveAlgOp : public OpKernel {
 public:
  explicit EveAlgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* f;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f));
    const Tensor* smoothed_f;
    OP_REQUIRES_OK(ctx, ctx->input("smoothed_f", &smoothed_f));
    const Tensor* d;
    OP_REQUIRES_OK(ctx, ctx->input("d", &d));
    const Tensor* target;
    OP_REQUIRES_OK(ctx, ctx->input("target", &target));
    const Tensor* step;
    OP_REQUIRES_OK(ctx, ctx->input("step", &step));
    const Tensor* decay_rate;
    OP_REQUIRES_OK(ctx, ctx->input("decay_rate", &decay_rate));
    const Tensor* f_decay_rate;
    OP_REQUIRES_OK(ctx, ctx->input("f_decay_rate", &f_decay_rate));
    const Tensor* lower;
    OP_REQUIRES_OK(ctx, ctx->input("lower", &lower));
    const Tensor* upper;
    OP_REQUIRES_OK(ctx, ctx->input("upper", &upper));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(f->shape()),
                errors::InvalidArgument("f is not a scalar: ",
                                        f->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(smoothed_f->shape()),
                errors::InvalidArgument("smoothed_f is not a scalar: ",
                                        smoothed_f->shape().DebugString()));
 
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(d->shape()),
                errors::InvalidArgument("d is not a scalar: ",
                                        d->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(target->shape()),
                errors::InvalidArgument("target is not a scalar: ",
                                        target->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(step->shape()),
                errors::InvalidArgument("step is not a scalar: ",
                                        step->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(decay_rate->shape()),
                errors::InvalidArgument("decay_rate is not a scalar: ",
                                        decay_rate->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(f_decay_rate->shape()),
                errors::InvalidArgument("f_decay_rate is not a scalar: ",
                                        f_decay_rate->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lower->shape()),
                errors::InvalidArgument("lower is not a scalar: ",
                                        lower->shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(upper->shape()),
                errors::InvalidArgument("upper is not a scalar: ",
                                        upper->shape().DebugString()));

    float prev_target = f->scalar<float>()();
    float smoothed_target = smoothed_f->scalar<float>()();
    float prev_d = d->scalar<float>()();
    float curr_target = target->scalar<float>()();
    float curr_step = step->scalar<int64>()();
    float beta = decay_rate->scalar<float>()();
    float f_beta = f_decay_rate->scalar<float>()();
    float low = lower->scalar<float>()();
    float up = upper->scalar<float>()();

    Tensor* out_f;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("out_f", TensorShape({}), &out_f));
    Tensor* out_smoothed_f;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("out_smoothed_f", TensorShape({}), &out_smoothed_f));
    Tensor* out_d;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("out_d", TensorShape({}), &out_d));

    auto f_ref = out_f->flat<float>();
    auto d_ref = out_d->flat<float>();
    auto smoothed_f_ref = out_smoothed_f->flat<float>();
    
    if (step == 0 || prev_target<=1e-6) {
      f_ref(0) = curr_target;
      smoothed_f_ref(0) = curr_target;
      d_ref(0) = 1.0;
      return;
    }
    else if (curr_target < 0) {
      std::cout << "Warning: curr_target:"<<curr_target<<std::endl;
      f_ref(0) = prev_target;
      d_ref(0) = prev_d;
      smoothed_f_ref(0) = smoothed_target;
      return;
    }

    float ratio_low = 0;
    float ratio_up = 0;

    if (curr_target > prev_target) {
      ratio_low = low + 1;
      ratio_up = up + 1;
    }
    else {
      ratio_low = 1 / (up + 1);
      ratio_up = 1 / (low + 1);
    }

    float c = std::min(std::max(ratio_low, curr_target/prev_target), ratio_up);
    curr_target = c * prev_target;
    float min_target = std::min(curr_target, prev_target);
    smoothed_target = smoothed_target * f_beta + min_target * (1 - f_beta);
    float r = fabs(curr_target - prev_target) / smoothed_target;
    
    d_ref(0) = beta * prev_d + (1 - beta) * r;
    f_ref(0) = curr_target;
    smoothed_f_ref(0) = smoothed_target;
  }
 private:
};

REGISTER_KERNEL_BUILDER(Name("EveAlg").Device(DEVICE_CPU), EveAlgOp);
