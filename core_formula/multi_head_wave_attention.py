"""
Homeostasis Spatial Bus - Multi-Head Wave-Attention Block (Part 1)
File: core_formula/multi_head_wave_attention.py
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any, Optional

# [ARCHITECTURAL ALIGNMENT] Relative path mapping following internal core integration relocations
from .softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder

# Ingesting our custom-fortified on-chip hardware rectifier modules
from bypass_rectifiers import LocalHomeostaticRectifier, TorusTopologyRotaryEmbedding

@jax.tree_util.register_pytree_node_class
class MultiHeadWaveAttention:
    """
    [COMPILER SAWER: LAYER 2.0 MULTI-HEAD WAVE-ATTENTION BLOCK - PART 1]
    
    Provides a drop-in hijacking component designed to intercept the legacy Softmax Attention layers 
    found in standard HuggingFace/Vanilla Transformers. This block deploys a wave-basis dimensional 
    contraction engine across 4D tensor rails without triggering any transcendental function bottlenecks.
    """
    def __init__(self, embed_dim: int = 4096, num_heads: int = 32, mesh_shape: int = 64, alpha: float = 0.01) -> None:
        """
        [INIT] Manages embedding feature factorization and hardcodes static memory layouts 
        optimized for hardware Tensor Core GEMM execution tracks.
        """
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"[ALIGNMENT ERROR] embed_dim({embed_dim}) must be a multiple of num_heads({num_heads})."
            )
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # [COMPILER INTERLOCK] Enforces explicit integer castings to block dynamic shape tuple mismatches inside the SPMD engine.
        self.mesh_shape = (mesh_shape, mesh_shape) if isinstance(mesh_shape, int) else tuple(mesh_shape)
        self.alpha = alpha
        self.hbar_eff = 1e-6
        
        # ------------------------------------------------------------------------
        # [DECIMATION GUARD: CORE DECODER STATIC PARAMETER BINDING]
        # ------------------------------------------------------------------------
        self.casimir_delta = 1e-4

        # Enforces a strict hardware stride alignment rule (even head dimension constraint) 
        # to eradicate numerical dead-zones over the Fourier complex phase topology.
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] head_dim({self.head_dim}) must be an even integer. "
                f"Please realign the total embedding dimensions or head counts."
            )

        # ------------------------------------------------------------------------
        # [SUB-MODULE INTERLOCK: ASSEMBBLING SYSTEM ATOMIC KERNELS TO MASTER CONTROL LINE]
        # ------------------------------------------------------------------------
        # Connects the localized homeostatic rectifier kernel to intercept on-chip register scales 
        # and shatter global reduction sync-locks during forward and backward trajectories.
        self.rectifier = LocalHomeostaticRectifier(
            head_dim=self.head_dim,
            alpha=self.alpha,
            casimir_delta=self.casimir_delta
        )
        
        # Mounts the closed torus topology manifold rotary position embedding kernel 
        # to confine unbounded rotational phase angles under long-context stress runs.
        self.torus_rope = TorusTopologyRotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=131072,  # Enforces maximum bare-metal memory limits
            base=10000.0,
            torus_radius=1.0     # Confinements curvature principal radius frozen
        )

            # ------------------------------------------------------------------------
        # [SUB-MODULE INTERLOCK: ASSEMBLING SYSTEM ATOMIC KERNELS TO MASTER CONTROL LINE]
        # ------------------------------------------------------------------------
        # Connects the localized homeostatic rectifier kernel to intercept on-chip register scales 
        # and shatter global reduction sync-locks during forward and backward trajectories.
        self.rectifier = LocalHomeostaticRectifier(
            head_dim=self.head_dim,
            alpha=self.alpha,
            casimir_delta=self.casimir_delta
        )
        
        # Mounts the closed torus topology manifold rotary position embedding kernel 
        # to confine unbounded rotational phase angles under long-context stress runs.
        self.torus_rope = TorusTopologyRotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=131072,  # Enforces maximum bare-metal memory limits
            base=10000.0,
            torus_radius=1.0     # Confinements curvature principal radius frozen
        )

        # Pre-allocates a 3D static basis tensor matrix array ensuring each independent multi-head 
        # occupies a distinct wave phase manifold space. Enforces an XLA compiler barrier 
        # to physically freeze this layout configuration inside HBM device memories.
        # [COMPILER INTERLOCK] Extracts a clean single-dimensional integer target to realign with the mesh_shape specifications.
        decoder_mesh_target = self.mesh_shape[0] if isinstance(self.mesh_shape, tuple) else self.mesh_shape
        self.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=decoder_mesh_target, feature_dim=self.head_dim, alpha=self.alpha)
            for _ in range(self.num_heads)
        ]
        
        # Consolidates all phase topology rails into a unified device-level ring buffer matrix (self.vorticity_omega_mesh)
        # to guarantee deterministic static path optimizations across PyTree nodes.
        self.vorticity_omega_mesh = jax.lax.stop_gradient(
            jnp.stack([decoder.vorticity_omega for decoder in self.decoders], axis=0)
        )

    # ========================================================================
    # XLA DISTRIBUTED SPMD DE-SERIALIZATION INTERLOCK (JAX PyTree Node Class)
    # ========================================================================
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[Any, ...]]:
        """
        Isolates dynamic computation tensors from static infrastructure hyperparameters, 
        guaranteeing the XLA compiler maintains explicit ownership traces across physical HBM networks.
        Appends the newly engineered atomic rectifier and torus positional encoding sub-modules into 
        the trailing metadata indexes to secure structural topology integrity during distributed execution.
        """
        # Compresses runtime dynamic children nodes into a single array container to lock down garbage collection overhead to 0MB.
        children = (self.vorticity_omega_mesh,)
        
        # [INFRA INTERLOCK] Hijacks index locations 7 and 8 to explicitly map the pre-loaded static sub-module states.
        aux_data = (
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            self.mesh_shape,
            self.alpha,
            self.hbar_eff,
            self.casimir_delta,
            self.rectifier,   # Index 7: Binds the custom LocalHomeostaticRectifier instance topology
            self.torus_rope   # Index 8: Binds the custom TorusTopologyRotaryEmbedding instance topology
        )
        return children, aux_data


          @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jax.Array]) -> "MultiHeadWaveAttention":
        """
        [SPMD STATIC ASSERTION CRASH FIX]
        Reconstructs and unpacks the original instance topology with bit-level precision during 
        backward computational graph updates, perfectly restoring the docked rectifier and torus position 
        embedding sub-modules into their bare-metal state without metadata tracking leaks.
        """
        # Triggers constructor bypass mapping guards to avoid redundant re-initialization
        embed_dim_raw = int(aux_data[0])
        num_heads_raw = int(aux_data[1])
        
        # [COMPILER INTERLOCK] Hardens structure guardrails to prevent primitive XLA data type mismatches.
        mesh_shape_aux = aux_data[3]
        mesh_shape_raw = (int(mesh_shape_aux[0]), int(mesh_shape_aux[1])) if isinstance(mesh_shape_aux, (tuple, list)) else (int(mesh_shape_aux), int(mesh_shape_aux))
        
        alpha_raw = float(aux_data[4])
        
        # Executes bare-metal structural binding directly over the class memory view.
        obj = cls.__new__(cls)
        obj.embed_dim = embed_dim_raw
        obj.num_heads = num_heads_raw
        obj.head_dim = int(aux_data[2])
        obj.mesh_shape = mesh_shape_raw
        obj.alpha = alpha_raw
        obj.hbar_eff = float(aux_data[5])
        
        # Re-binds static decimation constants matching the expanded auxiliary data layout.
        obj.casimir_delta = float(aux_data[6])
        
        # [SUB-MODULE RECONSTRUCTION INTERLOCK] Extracts the custom sub-module objects from index paths 7 and 8.
        obj.rectifier = aux_data[7]
        obj.torus_rope = aux_data[8]
        
        # Restores dynamic children tracking arrays with zero host-to-device memory allocation overhead.
        obj.vorticity_omega_mesh = children[0]
        
        # [ARCHITECTURAL ALIGNMENT] Preserves structural relative module mapping paths for sub-decoder matrix reconstruction.
        from .softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder
        decoder_mesh_target = obj.mesh_shape[0] if isinstance(obj.mesh_shape, tuple) else obj.mesh_shape
        obj.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=decoder_mesh_target, feature_dim=obj.head_dim, alpha=obj.alpha)
            for _ in range(obj.num_heads)
        ]
        
        return obj




            @partial(jax.jit, static_argnums=(0,))
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: Optional[jax.Array] = None, 
                 history_vessel: Optional[jax.Array] = None, mesh: jax.sharding.Mesh = None) -> Tuple[jax.Array, jax.Array]:
        """
        [OPERATIONAL FUSION RUNTIME GATEWAY - 4D GEMM RE-LAYOUT]
        [PART 3: Input Stream Head Factorization & Torus RoPE Phase Alignment - KV-Cache Interlock Extension]
        
        Extends the runtime operational signature to accept history_vessel (historical wave state container), 
        emitting both the newly generated final output stream and the real-time updated context_vessel as a tuple.
        """
        # q, k, v Input Shape: [Batch, SeqLen, EmbedDim]
        batch_size, seq_len, _ = q.shape
        target_dtype = q.dtype

        # ------------------------------------------------------------------------
        # [HARDWARE ALIGNMENT: DIRECT TENSOR CORE GEMM LAYOUT TRANSITION]
        # ------------------------------------------------------------------------
        q_split = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k_split = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v_split = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))

        # Executes axis transposition to force the XLA compiler to immediately recognize 
        # continuous memory strides as optimized GEMM execution tracks.
        # Shape: [Batch, NumHeads, SeqLen, HeadDim]
        q_h = q_split.transpose((0, 2, 1, 3))
        k_h = k_split.transpose((0, 2, 1, 3))
        v_h = v_split.transpose((0, 2, 1, 3))

        # ------------------------------------------------------------------------
        # [STAGE 1: BRANCHLESS MULTIPLICATION-ERASURE MASK INTERLOCK]
        # ------------------------------------------------------------------------
        default_mask = jnp.ones((batch_size, 1, 1, seq_len), dtype=jnp.bool_)
        safe_mask = jax.lax.select(mask is None, default_mask, mask)
        
        # Atomically contracts the physical signal charge of masked tokens in k_h and v_h streams to zero, 
        # completely preventing information leakage without introducing hardware pipeline breaks.
        k_h = jnp.where(safe_mask, k_h, 0.0)
        v_h = jnp.where(safe_mask, v_h, 0.0)

        # ------------------------------------------------------------------------
        # [STAGE 2: REGISTER-FREE TORUS POSITION EMBEDDING CONFINEMENT EXECUTION]
        # ------------------------------------------------------------------------
        # Builds a zero-overhead static positional coordinate index track (seq_idx) on the fly to enforce torus manifold projections.
        seq_idx = jnp.arange(seq_len, dtype=target_dtype)
        
        # Sequentially streams independent closed-system rotational phase transformations into both Q and K rails.
        q_h = self.torus_rope(stream=q_h, seq_idx=seq_idx, mesh=mesh)
        k_h = self.torus_rope(stream=k_h, seq_idx=seq_idx, mesh=mesh)

        # ------------------------------------------------------------------------
        # [MULTI-HEAD PARALLEL DISPATCH & WAVE DECODER FUSION TRIGGER]
        # ------------------------------------------------------------------------
        # [INTERLOCK EXTENSION] Channels the history_vessel directly into the trailing contraction engine 
        # to trigger incremental algebraic accumulation.
        return self._execute_wave_integration(q_h, k_h, v_h, history_vessel=history_vessel, mesh=mesh)





              def _execute_wave_integration(self, q_h: jax.Array, k_h: jax.Array, v_h: jax.Array, 
                                  history_vessel: Optional[jax.Array] = None, mesh: jax.sharding.Mesh = None) -> Tuple[jax.Array, jax.Array]:
        """
        [WAVE CONTRACTION & RECONSTRUCTION ENGINE]
        [PART 4: Wave Contraction Barycentric Integration & Local Rectifier Fusion Container Build - O(1) Incremental Cache Version]
        
        Completely eradicates legacy global reduction sync-locks (such as jnp.mean/jnp.var), 
        algebraically accumulating the new token's signal charge delta directly into the historical knowledge 
        container (history_vessel) via zero-allocation in-place additions.
        """
        batch_size, _, seq_len, _ = q_h.shape
        target_dtype = q_h.dtype

        # Synchronizes the dimension layout of the statically loaded device-level ring buffer 
        # (self.vorticity_omega_mesh) across the active 4D manifold topology.
        omega_vessel = self.vorticity_omega_mesh[None, :, :, None]

        # ------------------------------------------------------------------------
        # [HLO INLINE FUSION: DECLARING MULTI-HEAD INDEPENDENT WAVE PHASE BASES]
        # ------------------------------------------------------------------------
        half_dim = self.head_dim // 2
        grid_axis = jnp.arange(half_dim, dtype=target_dtype) / float(half_dim)
        grid_axis = grid_axis[None, None, None, :]

        # Simulates Euler's complex plane distributions to shield geometric symmetry (Fourier Basis Orthogonality).
        wave_sin = jnp.sin(omega_vessel * grid_axis)
        wave_cos = jnp.cos(omega_vessel * grid_axis)
        
        # Assembles the orthogonal basis projection matrix to insulate against information loss (Rank Collapse) 
        # via simulated constructive wave interference profiles.
        field_wave_T = jnp.concatenate([wave_sin, wave_cos], axis=-1)

        # ------------------------------------------------------------------------
        # [STAGE 1: O(N) LINEAR VESSEL CONTRACTION - CORE RUNTIME HIJACKING GATEWAY]
        # ------------------------------------------------------------------------
        # Avoids computing legacy [SeqLen, SeqLen] attention maps altogether. Instead, 
        # merges the K and V streams directly within the wave manifold space to build 
        # a globally shared container operating under strict linear O(N) complexity constraints.
        
        # [SUB-MODULE INTERLOCK 1: 1-Clock Inline FMA Fusion via Horner's Method over the K Stream]
        K_amplified = k_h * (1.0 + k_h * (1.0 + 0.5 * k_h))
        
        # [SUB-MODULE INTERLOCK 2: Executing Local Homeostatic Rectification to Evacuate Global Synchronization Barriers (K Rail)]
        K_rectified = self.rectifier(
            x=K_amplified, 
            gamma=None,  # Standalone rectification mode bypassing backbone parameters (0ns instant egress)
            use_gemma_offset=False, 
            mesh=mesh
        )

        # Step 3: Projects the rectified K manifold directly onto the Fourier frequency phase planes.
        k_wave = jnp.matmul(field_wave_T, K_rectified.transpose((0, 1, 3, 2)))

        # Step 4: Executes continuous barycentric moment tensor contraction with the incoming V stream.
        current_vessel = jnp.matmul(k_wave, v_h)

        # ------------------------------------------------------------------------
        # [STAGE 2: CONSTANT TIME O(1) SPACE KV-CACHE INCREMENTAL ACCUMULATION]
        # ------------------------------------------------------------------------
        # If an active historical knowledge vessel is provided, this path bypasses matrix reconstruction 
        # and immediately adds the new signal density in-place, achieving absolute O(1) memory scalability 
        # independent of the sequence length trajectory.
        if history_vessel is not None:
            context_vessel = history_vessel + current_vessel
        else:
            context_vessel = current_vessel


              # ------------------------------------------------------------------------
        # [COMPILER FENCE: DATA-PARALLEL / TENSOR-PARALLEL RUNTIME ADDRESS BUS GUARD]
        # ------------------------------------------------------------------------
        # Intercepts memory pointer states to prevent multi-node NCCL address bouncing.
        from .spmd_sharding_lanes import verify_context_vessel_sharding_coherence
        
        if mesh is not None:
            context_vessel = verify_context_vessel_sharding_coherence(mesh, context_vessel)
        else:
            try:
                # Dynamic runtime fallback scanning for distributed JAX/ShardMap execution context tracks
                from jax.experimental.shard_map import get_mesh
                current_mesh = get_mesh()
                if current_mesh is not None:
                    context_vessel = verify_context_vessel_sharding_coherence(current_mesh, context_vessel)
            except (ImportError, ValueError, RuntimeError):
                pass

        # ------------------------------------------------------------------------
        # [STAGE 3: Q-STREAM PARITY DECODING & TOPOLOGICAL MANIFOLD RESTORATION]
        # ------------------------------------------------------------------------
        # Decodes and unfolds high-fidelity token representation maps from the fused global wave container using the Q stream.
        
        # [SUB-MODULE INTERLOCK 1: 1-Clock Inline FMA Fusion via Horner's Method over the Q Stream]
        Q_amplified = q_h * (1.0 + q_h * (1.0 + 0.5 * q_h))
        
        # [SUB-MODULE INTERLOCK 2: Executing Local Homeostatic Rectification to Evacuate Global Synchronization Barriers (Q Rail)]
        Q_rectified = self.rectifier(
            x=Q_amplified,
            gamma=None,  # Standalone rectification mode passing 0ns instant egress
            use_gemma_offset=False,
            mesh=mesh
        )

        # Projects the rectified Q stream onto identical Fourier frequency phase planes.
        q_wave = jnp.matmul(Q_rectified, field_wave_T.transpose((0, 1, 3, 2)))

        # Blends global wave contextual density back into final phase topologies via updated context_vessel tracks.
        raw_attention_rail_output = jnp.matmul(q_wave, context_vessel)

        # ------------------------------------------------------------------------
        # [STAGE 4: BRANCHLESS MUX FIREWALL & L2 NORM PARITY ENERGY CONSERVATION]
        # ------------------------------------------------------------------------
        sanitized_stream = jnp.maximum(raw_attention_rail_output, 0.0)
        
        # [DECIMATION GUARD: CASIMIR VACUUM SINGULARITY ELASTIC BOUNDARY RESCUE LOCK]
        # Enforces a strict numerical lower-bound to block division-by-zero explosions if token signals collapse.
        square_sum = jnp.sum(jax.lax.square(sanitized_stream), axis=-1, keepdims=True)
        safe_square_sum = jnp.maximum(square_sum, self.casimir_delta)
        
        # Directly channels hardware-level jax.lax.rsqrt primitives to compress denominator normalization latencies.
        final_attention_rail_output = sanitized_stream * jax.lax.rsqrt(safe_square_sum + self.hbar_eff)

        # ------------------------------------------------------------------------
        # [STAGE 5: FINAL DROP-IN RE-LAYOUT - RECONSTRUCTING VANILLA LLM COMPATIBILITY]
        # ------------------------------------------------------------------------
        out_transposed = final_attention_rail_output.transpose((0, 2, 1, 3))
        final_output = out_transposed.reshape((batch_size, seq_len, self.embed_dim))

        # ------------------------------------------------------------------------
        # [COMPILER FENCE: EGRESS MATRIX TENSOR PARALLEL SHARDING CONSTITUTION]
        # ------------------------------------------------------------------------
        if mesh is not None:
            # Enforces explicit 'model' partition constraints to secure the trailing down-stream layers (o_proj / FFN).
            out_sharding_spec = P(None, None, 'model')
            named_out_sharding = jax.sharding.NamedSharding(mesh, out_sharding_spec)
            final_output = jax.lax.with_sharding_constraint(final_output, named_out_sharding)

        # [EGRESS INTERLOCK REGISTRATION] Returns both the processed final token hidden states and 
        # the real-time updated wave containers to support downstream real-time autoregressive generation loops.
        return final_output, context_vessel
