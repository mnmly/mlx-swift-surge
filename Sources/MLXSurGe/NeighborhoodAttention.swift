// 2D neighborhood attention for the NAD head.
//
// NATTEN's fused kernel has no MLX equivalent, so this reuses the SurGe
// "export-friendly" reimplementation (`coreai/nad_attention_torch.py`): plain
// gathers + softmax over a `kernel×kernel` inner-neighborhood window, validated
// to ~1e-6 vs real NATTEN. The window slides inward at borders so every query
// attends to exactly `kernel` keys per axis.

import MLX
import MLXFast
import MLXNN

// MARK: - Fused Metal kernel

/// Helper run before the main kernel body: the NATTEN inner-neighborhood window
/// start along one axis (mirrors `neighborIndices`).
private let nadKernelHeader = """
inline int nad_window_start(int i, int L, int K, int D) {
    int r = (K - 1) / 2;
    if (D <= 1) {
        return max(i - r, 0) + ((i + r >= L) ? (L - i - r - 1) : 0);
    }
    int a = (L / D) * D;
    int bRem = L - a;
    int imodd = i % D;
    int ni = i - r * D;
    int left = imodd;
    int right_lo = L - bRem + imodd - 2 * r * D;
    int right_hi = a + imodd - K * D;
    int right = (imodd < bRem) ? right_lo : right_hi;
    if (ni < 0) return left;
    if (i + r * D >= L) return right;
    return ni;
}
"""

/// One thread per output (b, i, j, head); loops the KH×KW window with an online
/// softmax over HD channels. `q` is pre-scaled (so scores need no `* scale`).
private let nadKernelSource = """
uint tid = thread_position_in_grid.x;
uint total = (uint)(B * H * W * HEADS);
if (tid >= total) { return; }

int head = (int)(tid % (uint)HEADS);
uint t = tid / (uint)HEADS;
int j = (int)(t % (uint)W);
t /= (uint)W;
int i = (int)(t % (uint)H);
int b = (int)(t / (uint)H);

int start_row = nad_window_start(i, H, KH, DH);
int start_col = nad_window_start(j, W, KW, DW);

uint qoff = (uint)(((((b * H + i) * W + j) * HEADS) + head) * HD);

float m = -1e30f;
float l = 0.0f;
float acc[HD];
for (int d = 0; d < HD; ++d) { acc[d] = 0.0f; }

for (int a = 0; a < KH; ++a) {
    int ii = start_row + a * DH;
    for (int bb = 0; bb < KW; ++bb) {
        int jj = start_col + bb * DW;
        uint koff = (uint)(((((b * H + ii) * W + jj) * HEADS) + head) * HD);
        float score = 0.0f;
        for (int d = 0; d < HD; ++d) { score += q[qoff + d] * k[koff + d]; }
        float new_m = max(m, score);
        float corr = exp(m - new_m);
        float ex = exp(score - new_m);
        l = l * corr + ex;
        for (int d = 0; d < HD; ++d) { acc[d] = acc[d] * corr + ex * v[koff + d]; }
        m = new_m;
    }
}

float inv = 1.0f / l;
for (int d = 0; d < HD; ++d) { out[qoff + d] = acc[d] * inv; }
"""

private let nadFusedKernel = MLXFast.metalKernel(
    name: "nad_neighborhood_attention",
    inputNames: ["q", "k", "v"],
    outputNames: ["out"],
    source: nadKernelSource,
    header: nadKernelHeader)

/// Fused windowed attention (drop-in for ``neighborhoodAttention(q:k:v:...)``).
///
/// Collapses the 2×kernel² gathers of the reference implementation into a single
/// Metal launch with an online softmax. `q`/`k`/`v` are `(B, H, W, heads, hd)`;
/// inputs are cast to float32 for the kernel and the result cast back to `q`'s
/// dtype. Numerically equivalent to ``neighborhoodAttention(q:k:v:kernelH:kernelW:dilationH:dilationW:scale:)``.
func neighborhoodAttentionFused(
    q: MLXArray, k: MLXArray, v: MLXArray,
    kernelH: Int, kernelW: Int, dilationH: Int, dilationW: Int, scale: Float
) -> MLXArray {
    let B = q.dim(0)
    let H = q.dim(1)
    let W = q.dim(2)
    let heads = q.dim(3)
    let hd = q.dim(4)

    // Fold the attention scale into q so the kernel needs no scalar input.
    let qScaled = (q.asType(.float32) * scale)
    let kf = k.asType(.float32)
    let vf = v.asType(.float32)

    let total = B * H * W * heads
    let tg = max(1, min(256, total))

    let out = nadFusedKernel(
        [qScaled, kf, vf],
        template: [
            ("B", B), ("H", H), ("W", W), ("HEADS", heads), ("HD", hd),
            ("KH", kernelH), ("KW", kernelW), ("DH", dilationH), ("DW", dilationW),
        ],
        grid: (total, 1, 1),
        threadGroup: (tg, 1, 1),
        outputShapes: [[B, H, W, heads, hd]],
        outputDTypes: [.float32])[0]

    return out.asType(q.dtype)
}

/// `(length, kernel)` neighbor positions per query along one axis (NATTEN
/// inner-neighborhood rule). Pure integer math, computed on the CPU.
func neighborIndices(length: Int, kernel: Int, dilation: Int) -> [[Int32]] {
    let r = (kernel - 1) / 2
    var result = [[Int32]](repeating: [Int32](repeating: 0, count: kernel), count: length)

    for i in 0..<length {
        let start: Int
        if dilation <= 1 {
            let clampedLeft = max(i - r, 0)
            let overflow = (i + r >= length) ? (length - i - r - 1) : 0
            start = clampedLeft + overflow
        } else {
            let a = (length / dilation) * dilation
            let bRem = length - a
            let imodd = i % dilation
            let ni = i - r * dilation
            let left = imodd
            let rightLo = length - bRem + imodd - 2 * r * dilation
            let rightHi = a + imodd - kernel * dilation
            let right = imodd < bRem ? rightLo : rightHi
            if ni < 0 {
                start = left
            } else if i + r * dilation >= length {
                start = right
            } else {
                start = ni
            }
        }
        for j in 0..<kernel {
            result[i][j] = Int32(start + j * dilation)
        }
    }
    return result
}

/// Plain-MLX equivalent of `neighborhood_attention_generic` for 2D inputs.
///
/// - q, k, v: `(B, H, W, heads, headDim)`.
/// - Returns: `(B, H, W, heads, headDim)`.
func neighborhoodAttention(
    q: MLXArray, k: MLXArray, v: MLXArray,
    kernelH: Int, kernelW: Int, dilationH: Int, dilationW: Int, scale: Float
) -> MLXArray {
    let H = q.dim(1)
    let W = q.dim(2)

    let rowIdx = neighborIndices(length: H, kernel: kernelH, dilation: dilationH) // (H, kh)
    let colIdx = neighborIndices(length: W, kernel: kernelW, dilation: dilationW) // (W, kw)

    // Column vectors of indices per window slot, as MLXArrays.
    func rowColumn(_ a: Int) -> MLXArray {
        MLXArray((0..<H).map { rowIdx[$0][a] })
    }
    func colColumn(_ b: Int) -> MLXArray {
        MLXArray((0..<W).map { colIdx[$0][b] })
    }

    // Pass 1: scores per (a, b) window slot, softmax over the kh*kw window.
    var scores: [MLXArray] = []
    scores.reserveCapacity(kernelH * kernelW)
    for a in 0..<kernelH {
        let kRow = take(k, rowColumn(a), axis: 1) // (B, H, W, heads, hd)
        for b in 0..<kernelW {
            let kAB = take(kRow, colColumn(b), axis: 2)
            scores.append((q * kAB).sum(axis: -1) * scale) // (B, H, W, heads)
        }
    }
    let attn = softmax(stacked(scores, axis: -1), axis: -1) // (B, H, W, heads, kh*kw)

    // Pass 2: weighted sum of value neighbors.
    var out: MLXArray? = nil
    var slot = 0
    for a in 0..<kernelH {
        let vRow = take(v, rowColumn(a), axis: 1)
        for b in 0..<kernelW {
            let vAB = take(vRow, colColumn(b), axis: 2)
            let w = attn[.ellipsis, slot, .newAxis] // (B, H, W, heads, 1)
            let term = w * vAB
            out = (out == nil) ? term : out! + term
            slot += 1
        }
    }
    return out!
}

/// Neighborhood attention module: QKV projection, optional QK LayerNorm, rope,
/// windowed attention, output projection. Mirrors SurGe's
/// `NeighborhoodAttention2d`.
public class NeighborhoodAttention2d: Module {
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    let kernelH: Int
    let kernelW: Int
    let dilationH: Int
    let dilationW: Int
    public let useQKNorm: Bool

    /// When true (the default), use the fused Metal kernel
    /// (``neighborhoodAttentionFused(q:k:v:kernelH:kernelW:dilationH:dilationW:scale:)``);
    /// when false, use the gather reference (the parity oracle, ~5× slower).
    public var useFusedKernel: Bool

    @ModuleInfo(key: "qkv") private var qkv: Linear
    @ModuleInfo(key: "proj") private var proj: Linear
    @ModuleInfo(key: "q_norm") private var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") private var kNorm: LayerNorm?

    public init(
        embedDim: Int, numHeads: Int,
        kernelSize: (Int, Int), dilation: (Int, Int),
        qkvBias: Bool = true, qkNorm: Bool = false,
        useFusedKernel: Bool = true
    ) {
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = 1.0 / Float(self.headDim).squareRoot()
        self.kernelH = kernelSize.0
        self.kernelW = kernelSize.1
        self.dilationH = dilation.0
        self.dilationW = dilation.1
        self.useQKNorm = qkNorm
        self.useFusedKernel = useFusedKernel

        self._qkv.wrappedValue = Linear(embedDim, embedDim * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(embedDim, embedDim)
        if qkNorm {
            self._qNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
            self._kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        }
    }

    public func callAsFunction(_ x: MLXArray, ropeEmbed: MLXArray?) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let c = x.dim(3)

        let qkvT = qkv(x)
            .reshaped([b, h, w, 3, numHeads, headDim])
            .transposed(3, 0, 1, 2, 4, 5) // (3, B, H, W, heads, hd)

        var q = qkvT[0]
        var k = qkvT[1]
        let v = qkvT[2]

        if let qNorm, let kNorm {
            q = qNorm(q).asType(v.dtype)
            k = kNorm(k).asType(v.dtype)
        }

        if let ropeEmbed {
            q = applyRotaryEmbedding(q, ropeEmbed)
            k = applyRotaryEmbedding(k, ropeEmbed)
        }

        let attnFn = useFusedKernel ? neighborhoodAttentionFused : neighborhoodAttention
        var out = attnFn(q, k, v, kernelH, kernelW, dilationH, dilationW, scale)
        out = out.reshaped([b, h, w, c])
        return proj(out)
    }
}
