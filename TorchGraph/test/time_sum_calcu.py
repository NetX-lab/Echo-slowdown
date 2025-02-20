data = {
    "x": 0,
    "conv": 0.0006623641401529312,
    "relu": 1.4728307723999023e-05,
    "size": 2.1789222955703735e-06,
    "view": 5.301311612129211e-06,
    "linear": 2.6744902133941653e-05,
    "output": 1.385807991027832e-06,
    "AddmmBackward00": 0.00010909780859947205,
    "AccumulateGrad0": 4.302412271499634e-06,
    "TBackward00": 3.833472728729248e-06,
    "AccumulateGrad1": 0.001196138486266136,
    "ViewBackward00": 4.125982522964477e-06,
    "ReluBackward00": 8.249431848526001e-06,
    "ConvolutionBackward00": 0.0002823498100042343,
    "AccumulateGrad2": 3.868713974952698e-06,
    "AccumulateGrad3": 3.753527998924255e-06,
    "optimizer_zero": 5.2526965737342836e-05,
    "optimizer_step": 2.3669153451919557e-05
}

total_time = sum(data.values())
print(f"total_time:{total_time*1000}ms")