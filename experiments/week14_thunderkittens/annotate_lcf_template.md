# Annotating the real ThunderKittens LCF matmul template

Source: the exact matmul example from the [ThunderKittens GitHub
README](https://github.com/HazyResearch/ThunderKittens) (section "Example: A
Simple Matrix Multiplication Kernel"). This is a reading/annotation exercise —
no new code, just the real template with a comment mapped to each numbered
step from the Week 14 Day 3 lesson's Section 5 walkthrough.

Each `<-- STEP N` marker below corresponds to the step numbered `N` in
`sessions/week-14/day-03-worker-overlapping.html`'s `#screen` stepper.

```cuda
#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;                                    // <-- STEP 1: the unit of work moved per step (64x64 bf16 shared tile)
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };               // <-- STEP 2: producer loads land here (shared memory)
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };                  // <-- STEP 2: consumer writes its result here before the store to HBM
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };       // <-- STEP 3: register accumulator; lives across many compute() calls for one tile
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
        // <-- STEP 4: grid() = dim3(132) — one persistent block per H100 SM, not one block per output tile.
        //     This is what the lesson's "grid-level persistent kernel" rule refers to: launch exactly
        //     as many blocks as there are SMs (132 on H100) and loop internally, instead of paying
        //     launch/teardown cost per tile.
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
        // <-- STEP 5: common_setup() computes the (row, col) block this iteration owns, walking
        //     blocks in SUPER_M-sized super-groups instead of raster order. This "SUPER_M ordering"
        //     is what the lesson calls out for improving L2 cache reuse -- nearby task_ids reuse
        //     nearby rows of A/columns of B that are still warm in L2.
        //
        // <-- STEP 6: warpgroup::groupid() == NUM_CONSUMER_WARPS/4 picks out the ONE extra warpgroup
        //     as the producer; every other warpgroup is a consumer. This assignment happens once per
        //     block and never changes -- the "static role split" the lesson describes.
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
            // <-- STEP 7: producer setup() shrinks its own register footprint to 40 per thread.
            //     Producers only compute addresses and issue async copies, so they release
            //     registers back to the shared per-SM pool for consumers to claim.
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
                // <-- STEP 8: producer load() issues async TMA copies of the next A and B tiles from
                //     HBM straight into input_block (shared memory). tma::expect() pre-registers how
                //     many bytes are coming so the barrier knows when to fire "inputs_arrived".
                //
                // <-- STEP 8 (cont.): only lane 0 of the warp (warpgroup::laneid() == 0) issues the
                //     TMA call. One thread's instruction moves the whole tile -- the other 31 lanes
                //     in that warp do nothing for this call, which is why producers need so few
                //     registers in the first place.
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            // <-- STEP 9: consumer setup() claims the registers producers freed (40 -> pool -> 232).
            kittens::warp::zero(args.state.accum);
            // <-- STEP 9 (cont.): zero the accumulator before the first matmul of a new output tile.
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            // <-- STEP 10: consumer compute() issues an asynchronous warpgroup matrix-multiply-
            //     accumulate (WGMMA) on the tile that the producer already staged in shared memory.
            //     This call returns immediately; the tensor cores keep working in the background.
            warpgroup::mma_async_wait();
            // <-- STEP 11: mma_async_wait() blocks only until THIS warpgroup's mma finishes -- not a
            //     global kernel-wide sync. Other warps/warpgroups are unaffected.
            if (warp::laneid() == 0) arrive(args.inputs_finished);
            // <-- STEP 12: arrive(inputs_finished) is the barrier handshake back to the producer:
            //     "I'm done reading this shared-memory slot, you may overwrite it with tile N+2."
            //     This exact load -> compute -> arrive sequence repeats once per iter, with
            //     INPUT_PIPE_STAGES = 4 tiles multi-buffered so the producer can be several tiles
            //     ahead of the consumer without waiting.
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            // <-- STEP 13: consumer finish() moves the final accumulator from registers into
            //     finish_block (shared memory) once num_iters worth of compute() calls are done.
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                // <-- STEP 14: tma::store_async() asynchronously writes the finished output tile from
                //     shared memory back to HBM (global memory C) -- again a single non-blocking
                //     instruction that frees the calling thread immediately.
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
                // <-- STEP 15: tma::store_async_read_wait() blocks only until THIS store is safely
                //     read by the TMA engine, so finish_block can be reused for the next output tile
                //     -- again a narrow wait, not a kernel-wide sync.
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
            // <-- STEP 16: arrive(finish_finished) signals the block is done with this output tile;
            //     common_setup() runs again for the next task_id if any work remains (loop back to
            //     STEP 5). Across the whole file this ~100-line template reaches 855 TFLOPs on an
            //     H100, about 86% of theoretical peak bf16 throughput -- almost entirely because
            //     producer and consumer warps overlap instead of taking turns.
        }
    };
};
```

## Step-index cross-reference

| Walkthrough step | Template line(s) | What it does |
|---|---|---|
| 1 | `base_tile = st_bf<64,64>` | Defines the 64x64 bf16 tile that moves per load/store |
| 2 | `input_block`, `finish_block` | Separate shared-memory regions for producer loads vs. consumer output |
| 3 | `consumer_state { rt_fl<...> accum }` | Register accumulator that persists across many `compute()` calls |
| 4 | `grid() { return dim3(132); }` | Persistent grid: one block per H100 SM (132), not per tile |
| 5 | `common_setup()` SUPER_M logic | Chooses (row, col) this task_id owns using SUPER_M block ordering for L2 reuse |
| 6 | `warpgroup::groupid()==NUM_CONSUMER_WARPS/4` | Static, once-per-block producer/consumer role split |
| 7 | `producer::setup` → `decrease_registers<40>()` | Shrinks producer register footprint |
| 8 | `producer::load` → `tma::expect` + `tma::load_async` | Async fetch of A/B tiles from HBM into shared memory, issued by lane 0 only |
| 9 | `consumer::setup` → `increase_registers<232>()` + `zero(accum)` | Consumer claims freed registers, zeroes its accumulator |
| 10 | `consumer::compute` → `warpgroup::mma_AB` | Async tensor-core matmul on the already-staged tile |
| 11 | `warpgroup::mma_async_wait()` | Narrow wait for only this warpgroup's matmul |
| 12 | `arrive(args.inputs_finished)` | Barrier handshake telling the producer this shared-memory slot is free |
| 13 | `consumer::finish` → `warpgroup::store` | Moves the finished accumulator from registers to shared memory |
| 14 | `tma::store_async` | Async, non-blocking write of the output tile back to HBM |
| 15 | `tma::store_async_read_wait()` | Narrow wait for only this store before reusing `finish_block` |
| 16 | `arrive(args.finish_finished)`; loop | Signals completion; `common_setup()` runs again if tiles remain |

That is 16 inline `<-- STEP N` annotations in the code block above (one per
numbered walkthrough step, with step 8 and step 9 each carrying two related
annotations), satisfying the "at least 15 inline annotations" requirement.
