// ============================================================
// fused_ffn_layer.cpp — Fused FFN Layer for NMC4
// ============================================================
// ONE dispatch = entire FFN layer:
//   Input → RMSNorm → MatMul(W1) → ReLU → MatMul(W2) → Add(residual) → Output
//
// All data stays in DDR. No PCI round-trips between ops.
// Uses Q16.16 fixed-point (mymath.h).
//
// OP_FUSED_FFN = 30
// args: [T, D, F, addr_x, addr_g, addr_W1, addr_b1, addr_W2, addr_b2, addr_out]
//   T = sequence length, D = embed dim, F = FFN dim
//   All addr = DDR word addresses containing IEEE 754 floats
// ============================================================

#include "mymath.h"

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define OP_FUSED_FFN     30
#define OP_FUSED_ATTN    31
#define OP_FUSED_FULL    32
#define OP_NOP           0
#define OP_EXIT          255
#define STATUS_ADDR      30
#define WATCHDOG_ADDR    31

// ============================================================
// Fused FFN: RMSNorm → Linear(W1) → ReLU → Linear(W2) → Residual Add
// ============================================================
void op_fused_ffn() {
    unsigned int T   = mem[1];
    unsigned int D   = mem[2];
    unsigned int F   = mem[3];
    unsigned int* x_in  = (unsigned int*)mem[4];   // [T, D] input (IEEE float)
    unsigned int* gamma = (unsigned int*)mem[5];   // [D] RMSNorm weight
    unsigned int* W1    = (unsigned int*)mem[6];   // [D, F] first linear
    unsigned int* b1    = (unsigned int*)mem[7];   // [F] bias
    unsigned int* W2    = (unsigned int*)mem[8];   // [F, D] second linear
    unsigned int* b2    = (unsigned int*)mem[9];   // [D] bias
    unsigned int* x_out = (unsigned int*)mem[10];  // [T, D] output

    unsigned int t_idx = 0;  // incremental index for row t

    for (unsigned int t = 0; t < T; t++) {
        // --- RMSNorm ---
        // rms = sqrt(mean(x^2) + eps)
        fixed32 sum_sq = 0;
        unsigned int x_off = t_idx;  // t * D done incrementally
        for (unsigned int d = 0; d < D; d++) {
            fixed32 val = float_to_fixed(x_in[x_off + d]);
            sum_sq = add_fixed(sum_sq, mul_fixed(val, val));
        }
        fixed32 rms = sqrt_fixed(add_fixed(div_fixed(sum_sq, INT_TO_FIXED(D)), 1));
        fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

        // --- Linear(W1) + ReLU ---
        // h[f] = ReLU(sum_d(norm_x[d] * W1[d,f]) + b1[f])
        // Store in temp area after x_out
        unsigned int* h_temp = x_out + mul_u32(T, D);  // temp [T, F] after output

        unsigned int h_off = mul_u32(t, F);
        for (unsigned int f = 0; f < F; f++) {
            fixed32 acc = float_to_fixed(b1[f]);
            unsigned int w1_col = f;
            for (unsigned int d = 0; d < D; d++) {
                fixed32 xn = mul_fixed(mul_fixed(float_to_fixed(x_in[x_off + d]), inv_rms),
                                       float_to_fixed(gamma[d]));
                fixed32 w = float_to_fixed(W1[mul_u32(d, F) + f]);
                acc = add_fixed(acc, mul_fixed(xn, w));
            }
            // ReLU
            if (acc < 0) acc = 0;
            h_temp[h_off + f] = fixed_to_float(acc);
        }

        // --- Linear(W2) + Residual ---
        // out[d] = x_in[d] + sum_f(h[f] * W2[f,d]) + b2[d]
        for (unsigned int d = 0; d < D; d++) {
            fixed32 acc = add_fixed(float_to_fixed(x_in[x_off + d]),
                                    float_to_fixed(b2[d]));
            for (unsigned int f = 0; f < F; f++) {
                fixed32 hv = float_to_fixed(h_temp[h_off + f]);
                fixed32 w = float_to_fixed(W2[mul_u32(f, D) + d]);
                acc = add_fixed(acc, mul_fixed(hv, w));
            }
            x_out[x_off + d] = fixed_to_float(acc);
        }

        t_idx += D;
    }
}

// ============================================================
// Fused Full Layer: RMSNorm → QKV Proj → Attention → Output Proj → Residual → FFN
// ============================================================
void op_fused_full_layer() {
    // args layout:
    // mem[1]=T, mem[2]=D, mem[3]=H, mem[4]=F
    // mem[5]=addr_x, mem[6]=addr_g1 (attn norm)
    // mem[7]=addr_Wq, mem[8]=addr_Wk, mem[9]=addr_Wv, mem[10]=addr_Wo
    // mem[11]=addr_g2 (ffn norm)
    // mem[12]=addr_W1, mem[13]=addr_b1, mem[14]=addr_W2, mem[15]=addr_b2
    // mem[16]=addr_out, mem[17]=addr_scratch

    unsigned int T  = mem[1];
    unsigned int D  = mem[2];
    unsigned int Heads = mem[3];
    unsigned int F  = mem[4];
    // HD = D / Heads — avoid UDiv32 (broken libgcc)
    // Heads is power of 2: 1,2,4,8. Use shift.
    unsigned int hshift = 0;
    if (Heads == 2) hshift = 1;
    else if (Heads == 4) hshift = 2;
    else if (Heads == 8) hshift = 3;
    unsigned int HD = my_rshift(D, hshift);

    unsigned int* x_in   = (unsigned int*)mem[5];
    unsigned int* g1     = (unsigned int*)mem[6];
    unsigned int* Wq     = (unsigned int*)mem[7];
    unsigned int* Wk     = (unsigned int*)mem[8];
    unsigned int* Wv     = (unsigned int*)mem[9];
    unsigned int* Wo     = (unsigned int*)mem[10];
    unsigned int* g2     = (unsigned int*)mem[11];
    unsigned int* W1     = (unsigned int*)mem[12];
    unsigned int* b1     = (unsigned int*)mem[13];
    unsigned int* W2     = (unsigned int*)mem[14];
    unsigned int* b2     = (unsigned int*)mem[15];
    unsigned int* x_out  = (unsigned int*)mem[16];
    unsigned int* scratch = (unsigned int*)mem[17];

    // scratch layout:
    // [0..T*D-1]        = normalized x
    // [T*D..2*T*D-1]    = Q
    // [2*T*D..3*T*D-1]  = K
    // [3*T*D..4*T*D-1]  = V
    // [4*T*D..5*T*D-1]  = attn_out
    // [5*T*D..5*T*D+T*F-1] = FFN hidden

    unsigned int* xn     = scratch;
    unsigned int* Q      = scratch + mul_u32(T, D);
    unsigned int* K_     = Q + mul_u32(T, D);
    unsigned int* V_     = K_ + mul_u32(T, D);
    unsigned int* attn_o = V_ + mul_u32(T, D);
    unsigned int* h_ffn  = attn_o + mul_u32(T, D);

    // --- Step 1: RMSNorm for attention ---
    for (unsigned int t = 0; t < T; t++) {
        unsigned int off = mul_u32(t, D);
        fixed32 sum_sq = 0;
        for (unsigned int d = 0; d < D; d++) {
            fixed32 v = float_to_fixed(x_in[off + d]);
            sum_sq = add_fixed(sum_sq, mul_fixed(v, v));
        }
        fixed32 inv_rms = div_fixed(FIXED_ONE,
            sqrt_fixed(add_fixed(div_fixed(sum_sq, INT_TO_FIXED(D)), 1)));
        for (unsigned int d = 0; d < D; d++) {
            fixed32 v = mul_fixed(mul_fixed(float_to_fixed(x_in[off + d]), inv_rms),
                                  float_to_fixed(g1[d]));
            xn[off + d] = fixed_to_float(v);
        }
    }

    // --- Step 2: Q, K, V projections ---
    // Q = xn @ Wq, K = xn @ Wk, V = xn @ Wv
    for (unsigned int t = 0; t < T; t++) {
        unsigned int xn_off = mul_u32(t, D);
        unsigned int q_off  = xn_off;
        for (unsigned int d = 0; d < D; d++) {
            fixed32 sq = 0, sk = 0, sv = 0;
            for (unsigned int k = 0; k < D; k++) {
                fixed32 xv = float_to_fixed(xn[xn_off + k]);
                sq = add_fixed(sq, mul_fixed(xv, float_to_fixed(Wq[mul_u32(k, D) + d])));
                sk = add_fixed(sk, mul_fixed(xv, float_to_fixed(Wk[mul_u32(k, D) + d])));
                sv = add_fixed(sv, mul_fixed(xv, float_to_fixed(Wv[mul_u32(k, D) + d])));
            }
            Q[q_off + d]  = fixed_to_float(sq);
            K_[q_off + d] = fixed_to_float(sk);
            V_[q_off + d] = fixed_to_float(sv);
        }
    }

    // --- Step 3: Multi-head Attention (per head) ---
    for (unsigned int h = 0; h < Heads; h++) {
        for (unsigned int t = 0; t < T; t++) {
            // Compute attention scores for this position
            // score[t, t2] = Q[t,h,:] @ K[t2,h,:] / sqrt(HD)
            fixed32 max_score = INT_TO_FIXED(-30000);

            // First pass: find max for numerical stability
            for (unsigned int t2 = 0; t2 <= t; t2++) {  // causal mask
                fixed32 score = 0;
                for (unsigned int d = 0; d < HD; d++) {
                    unsigned int qi = mul_u32(t, D) + mul_u32(h, HD) + d;
                    unsigned int ki = mul_u32(t2, D) + mul_u32(h, HD) + d;
                    score = add_fixed(score,
                        mul_fixed(float_to_fixed(Q[qi]), float_to_fixed(K_[ki])));
                }
                // Divide by sqrt(HD) ≈ multiply by 1/sqrt(HD)
                // For HD=8: 1/sqrt(8) ≈ 0.354 ≈ 23170 in Q16.16
                score = mul_fixed(score, 23170);  // ~1/sqrt(8)
                if (score > max_score) max_score = score;
            }

            // Second pass: softmax
            fixed32 exp_sum = 0;
            // Use scratch for scores (reuse h_ffn area temporarily)
            for (unsigned int t2 = 0; t2 <= t; t2++) {
                fixed32 score = 0;
                for (unsigned int d = 0; d < HD; d++) {
                    unsigned int qi = mul_u32(t, D) + mul_u32(h, HD) + d;
                    unsigned int ki = mul_u32(t2, D) + mul_u32(h, HD) + d;
                    score = add_fixed(score,
                        mul_fixed(float_to_fixed(Q[qi]), float_to_fixed(K_[ki])));
                }
                score = mul_fixed(score, 23170);
                fixed32 e = exp_fixed_lut(sub_fixed(score, max_score));
                h_ffn[t2] = (unsigned int)e;  // store exp temporarily
                exp_sum = add_fixed(exp_sum, e);
            }
            if (exp_sum == 0) exp_sum = 1;

            // Weighted sum of V
            for (unsigned int d = 0; d < HD; d++) {
                fixed32 acc = 0;
                for (unsigned int t2 = 0; t2 <= t; t2++) {
                    fixed32 w = div_fixed((fixed32)h_ffn[t2], exp_sum);
                    unsigned int vi = mul_u32(t2, D) + mul_u32(h, HD) + d;
                    acc = add_fixed(acc, mul_fixed(w, float_to_fixed(V_[vi])));
                }
                unsigned int oi = mul_u32(t, D) + mul_u32(h, HD) + d;
                attn_o[oi] = fixed_to_float(acc);
            }
        }
    }

    // --- Step 4: Output projection + residual ---
    for (unsigned int t = 0; t < T; t++) {
        unsigned int off = mul_u32(t, D);
        for (unsigned int d = 0; d < D; d++) {
            fixed32 acc = float_to_fixed(x_in[off + d]);  // residual
            for (unsigned int k = 0; k < D; k++) {
                acc = add_fixed(acc,
                    mul_fixed(float_to_fixed(attn_o[off + k]),
                              float_to_fixed(Wo[mul_u32(k, D) + d])));
            }
            x_out[off + d] = fixed_to_float(acc);
        }
    }

    // --- Step 5: RMSNorm for FFN ---
    for (unsigned int t = 0; t < T; t++) {
        unsigned int off = mul_u32(t, D);
        fixed32 sum_sq = 0;
        for (unsigned int d = 0; d < D; d++) {
            fixed32 v = float_to_fixed(x_out[off + d]);
            sum_sq = add_fixed(sum_sq, mul_fixed(v, v));
        }
        fixed32 inv_rms = div_fixed(FIXED_ONE,
            sqrt_fixed(add_fixed(div_fixed(sum_sq, INT_TO_FIXED(D)), 1)));
        for (unsigned int d = 0; d < D; d++) {
            xn[off + d] = fixed_to_float(
                mul_fixed(mul_fixed(float_to_fixed(x_out[off + d]), inv_rms),
                          float_to_fixed(g2[d])));
        }
    }

    // --- Step 6: FFN (Linear → ReLU → Linear) + residual ---
    for (unsigned int t = 0; t < T; t++) {
        unsigned int off = mul_u32(t, D);
        unsigned int h_off = mul_u32(t, F);

        // W1 projection + ReLU
        for (unsigned int f = 0; f < F; f++) {
            fixed32 acc = float_to_fixed(b1[f]);
            for (unsigned int d = 0; d < D; d++) {
                acc = add_fixed(acc,
                    mul_fixed(float_to_fixed(xn[off + d]),
                              float_to_fixed(W1[mul_u32(d, F) + f])));
            }
            if (acc < 0) acc = 0;  // ReLU
            h_ffn[h_off + f] = fixed_to_float(acc);
        }

        // W2 projection + residual
        for (unsigned int d = 0; d < D; d++) {
            fixed32 acc = float_to_fixed(x_out[off + d]);  // residual from attention
            acc = add_fixed(acc, float_to_fixed(b2[d]));
            for (unsigned int f = 0; f < F; f++) {
                acc = add_fixed(acc,
                    mul_fixed(float_to_fixed(h_ffn[h_off + f]),
                              float_to_fixed(W2[mul_u32(f, D) + d])));
            }
            x_out[off + d] = fixed_to_float(acc);
        }
    }
}

// ============================================================
// Main dispatcher loop
// ============================================================
int main() {
    mem[STATUS_ADDR] = 0;
    mem[WATCHDOG_ADDR] = 0;
    unsigned int watchdog = 0;

    while (1) {
        watchdog++;
        mem[WATCHDOG_ADDR] = watchdog;

        unsigned int op = mem[0];
        if (op == OP_NOP) continue;
        if (op == OP_EXIT) {
            mem[STATUS_ADDR] = 1;
            mem[0] = OP_NOP;
            break;
        }

        mem[STATUS_ADDR] = 0;

        switch (op) {
            case OP_FUSED_FFN:        op_fused_ffn(); break;
            case OP_FUSED_FULL:       op_fused_full_layer(); break;
            default:
                mem[STATUS_ADDR] = 2;
                mem[0] = OP_NOP;
                continue;
        }

        mem[STATUS_ADDR] = 1;
        mem[0] = OP_NOP;
    }
    return 0;
}
