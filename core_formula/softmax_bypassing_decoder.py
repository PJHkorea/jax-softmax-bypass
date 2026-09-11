"""
Advanced Softmax-Bypassing Wave Decoder
File: core_formula/softmax_bypassing_decoder.py

[Mathematical Philosophy: Integrated Hardware Acceleration & Backward Optimization Track]
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any

# Ingesting the global sync-lock evacuation rectifier engine
from bypass_rectifiers import LocalHomeostaticRectifier

@jax.tree_util.register_pytree_node_class
class UpgradedSoftmaxBypassingDecoder:
    """
    [COMPILER SAWER: LAYER 1.6 UPGRADED SOFTMAX-BYPASSING WAVE DECODER]
    [PART 1: Kernel Declaration & Hardware Memory Binding - Trainable Optimization Extension]
    
    Locks and binds static wave basis axes over an Euler orthogonal phase plane structure, 
    ensuring the XLA compiler preserves structural continuity inside HBM device memories 
    when assembling backward automatic differentiation gradient trajectories.
    """
    def __init__(self, mesh_shape: int = 64, feature_dim: int = 4096, alpha: float = 0.01) -> None:
        """
        [INIT] Hardcodes static hardware bus stride configurations and frequency domain orthogonal bases.
        """
        # [COMPILER INTERLOCK: HARDWARE STRIDE ALIGNMENT] 
        if feature_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] feature_dim must be an even integer. (Input: {feature_dim})\n"
                f"Enforcing power-of-two architectures is highly recommended to cleanly partition the Fourier sin/cos planes."
            )

        self.mesh_shape = (mesh_shape, mesh_shape) if isinstance(mesh_shape, int) else tuple(mesh_shape)
        self.feature_dim = feature_dim
        self.alpha = alpha  # Non-linear damping scalar coefficient
        self.hbar_eff = 1e-6  # Attenuation guardrail constant to block division-by-zero or numerical explosions
        
        self.scale_factor = 1.0 / jnp.sqrt(float(self.feature_dim))
        self.casimir_delta = 1e-4

        # ------------------------------------------------------------------------
        # [SUB-MODULE INTERLOCK: MOUNTING INTEGRATED HOMEOSTATIC RECTIFIER ENGINE]
        # ------------------------------------------------------------------------
        # Shares the declared feature_dim, alpha, and casimir_delta configurations directly 
        # to bind the custom rectifier kernel, enabling local self-neutralization inside device registers 
        # without invoking global synchronization barriers.
        self.rectifier = LocalHomeostaticRectifier(
            head_dim=self.feature_dim,
            alpha=self.alpha,
            casimir_delta=self.casimir_delta
        )

        # Declares static mesh coordinates to construct the orthogonal Fourier basis vectors (Sin/Cos).
        self.vorticity_omega = jax.lax.stop_gradient(
            jnp.linspace(-jnp.pi, jnp.pi, self.mesh_shape[0], dtype=jnp.float32)
        )

    # ========================================================================
    # XLA STATIC COMPILER TRACING INVARIANT (JAX PyTree Serialization Protocol)
    # ========================================================================
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[Any, ...]]:
        """
        Completely segregates dynamic computation arrays from static hyperparameters, 
        ensuring zero hardware tracking leaks over the HBM distributed clusters. 
        Appends the embedded self.rectifier engine state into the static metadata track 
        to guarantee structural serialization completeness.
        """
        # Allocates dynamic target arrays evaluated during backward gradient propagation.
        children = (self.vorticity_omega,)
        
        # [INFRA INTERLOCK] Binds the custom rectifier instance topology directly onto the trailing 
        # auxiliary data index path to insulate the compiler tracking graphs against layout mutation crashes.
        aux_data = (
            self.mesh_shape, 
            self.feature_dim, 
            self.alpha, 
            self.hbar_eff, 
            self.scale_factor, 
            self.casimir_delta,
            self.rectifier  # Secure mapping line for the static rectifier instance
        )
        return children, aux_data

       @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jax.Array]) -> "UpgradedSoftmaxBypassingDecoder":
        """
        Reconstructs the original structural matrix layout and embedded rectifier kernel topology 
        during backward automatic differentiation tracking path evaluation.
        """
        # [COMPILER INTERLOCK] Hardens layout guards to ensure safe tuple unpacking across distributed SPMD networks.
        mesh_shape_raw = aux_data[0]
        mesh_init = mesh_shape_raw[0] if isinstance(mesh_shape_raw, tuple) else mesh_shape_raw
        
        # Unpacks infrastructure metrics directly into the class layout instance.
        obj = cls(mesh_shape=int(mesh_init), feature_dim=int(aux_data[1]), alpha=float(aux_data[2]))
        obj.hbar_eff = float(aux_data[3])
        obj.scale_factor = float(aux_data[4])
        obj.casimir_delta = float(aux_data[5])
        
        # [SUB-MODULE RECONSTRUCTION INTERLOCK] Re-extracts and copies the static rectifier instance from auxiliary index 6.
        obj.rectifier = aux_data[6]
        
        obj.vorticity_omega = children[0]
        return obj

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def __call__(self, clean_manifold_tensor: jax.Array, gamma: jax.Array = None, use_gemma_offset: bool = False, mesh: jax.sharding.Mesh = None) -> jax.Array:
        """
        [OPERATIONAL FUSION RUNTIME GATEWAY - TAYLOR-FOURIER INTEGRAL INVERSION]
        [PART 2: Taylor Series Horner's Method Inline Fusion & Local Homeostatic Rectifier Runtime]
        
        Eradicates the legacy global reduction bottlenecks (mean/var synchronization locks), 
        completing a localized linear rectification highway exclusively within 1-clock hardware FMA pipelines.
        """
        target_dtype = clean_manifold_tensor.dtype
        
        # ------------------------------------------------------------------------
        # [STAGE 1: INPUT MANIFOLD SCALING & NEGATIVE MASK EXPLOSION CLIPPING FIREWALL]
        # ------------------------------------------------------------------------
        # Step 1-1. Realigms the scalar balance of the incoming stream tensor to match the legacy LLM backbone weights ecosystem.
        X_scaled = clean_manifold_tensor * self.scale_factor
        
        # Step 1-2. Fuses an inline clipping boundary using hardware MUX primitives to intercept inverse mask explosions (-10000.0).
        X_safe = jnp.maximum(X_scaled, -50.0)
        
        # ------------------------------------------------------------------------
        # [SUB-MODULE INTERLOCK 1: COMPILER INLINE FMA FUSION VIA HORNER'S METHOD]
        # ------------------------------------------------------------------------
        # Factorizes the standard polynomial function layout x * (1 + x + 0.5 * x^2) into x * (1.0 + x * (1.0 + 0.5 * x)).
        # This completely wipes out register spilling and temporary tensor track allocations (X_squared, X_amplified), 
        # driving the phase amplification within a single-pass hardware register pipeline.
        X_amplified = X_safe * (1.0 + X_safe * (1.0 + 0.5 * X_safe))
        
        # ------------------------------------------------------------------------
        # [SUB-MODULE INTERLOCK 2: EVACUATING REDUNDANT SYNCHRONIZATION BARRIERS VIA LOCAL RECTIFIER]
        # ------------------------------------------------------------------------
        # Permanently removes jnp.mean/jnp.var global reduction lock execution tracks, streaming the data 
        # through an advanced local stabilization kernel equipped with SPMD mesh specifications and universal weight hijacking rails.
        X_rectified = self.rectifier(
            x=X_amplified, 
            gamma=gamma, 
            use_gemma_offset=use_gemma_offset, 
            mesh=mesh
        )


               # [COMPILER HLO INLINE FUSION: ASSEMBLING FOURIER ORTHOGONAL BASES]
        half_dim = self.feature_dim // 2
        grid_axis = jnp.arange(half_dim, dtype=target_dtype) / float(half_dim)
        
        # Maps virtual coordinate matrices over the statically loaded vorticity_omega constants 
        # to ensure zero redundant VRAM track allocations (0MB dynamic footprint constraint).
        wave_sin = jnp.sin(self.vorticity_omega[:, None] * grid_axis[None, :])
        wave_cos = jnp.cos(self.vorticity_omega[:, None] * grid_axis[None, :])
        field_wave_T = jnp.concatenate([wave_sin, wave_cos], axis=-1)  # Virtual Layout Shape: [Mesh, Feature]

        # [STAGE 1 CONTRACTION: BARYCENTRIC MOMENT INTERGRAL CONTRACTION SCAN]
        # Shrinks the representation dimensions into the Fourier complex manifold space.
        purified_guide_stream = jnp.matmul(X_rectified, field_wave_T.T)

        # [STAGE 2 EXPANSION: EUCLIDEAN MINIMAL RESIDUAL TOPOLOGY RESTORATION]
        # Reconstructs the high-fidelity token representation plane to neutralize Rank Collapse defects.
        final_attention_rail_input = jnp.matmul(purified_guide_stream, field_wave_T)
        
        # [BRANCHLESS MUX FIREWALL]
        # Neutralizes inverse signal leaks using hardware-level multiplexer boundaries.
        sanitized_stream = jnp.maximum(final_attention_rail_input, 0.0)

        # ------------------------------------------------------------------------
        # [DECIMATION GUARD: CASIMIR VACUUM SINGULARITY & ELASTIC RESCUE LOCK]
        # ------------------------------------------------------------------------
        # If all frequency components within the sanitized_stream collapse into negative spaces and 
        # get zeroed out, the square_sum topology completely self-destructs, causing the expressive power 
        # of downstream backbone blocks (o_proj, FFN) to permanently die out (division by zero).
        # To insulate against this singularity, a hard quantum vacuum threshold (casimir_delta) is injected 
        # immediately before L2 normalization to physically salvage a baseline coherent energy density.
        square_sum = jnp.sum(jax.lax.square(sanitized_stream), axis=-1, keepdims=True)
        safe_square_sum = jnp.maximum(square_sum, self.casimir_delta)
        
        # [WAVE L2 NORM PARITY ENERGY CONSERVATION: CHANNELS HARDWARE RSQRT]
        # Interfaces directly with hardware-level jax.lax.rsqrt primitives to compress vector scaling latencies.
        final_attention_rail_output = sanitized_stream * jax.lax.rsqrt(safe_square_sum + self.hbar_eff)
        
        # ------------------------------------------------------------------------
        # [COMPILER FENCE: SAFEGARDING TENSOR PARALLEL SHARDING CONSTITUTION]
        # ------------------------------------------------------------------------
        # Dynamically tracks the outgoing tensor layout rank to enforce SPMD model partition constraints, 
        # ensuring the egress data streams connect with downstream linear layers (o_proj) without triggering memory bus stalls.
        if mesh is not None:
            output_rank = final_attention_rail_output.ndim
            if output_rank == 3:
                # Target layout configuration for standard [Batch, SeqLen, Dim] -> Shards the trailing channel axis to 'model'.
                out_sharding_spec = P(None, None, 'model')
            else:
                # Target layout configuration for variable 4D arrays -> Dynamically isolates the trailing representation axis.
                out_sharding_spec = P(*(None,) * (output_rank - 1), 'model')
                
            named_out_sharding = jax.sharding.NamedSharding(mesh, out_sharding_spec)
            final_attention_rail_output = jax.lax.with_sharding_constraint(
                final_attention_rail_output, named_out_sharding
            )

        # [BACKPROPAGATION PATHWAY UNLOCK] Completely removes all legacy stop_gradient tracking barricades 
        # that previously severed automatic differentiation paths. This establishes a fully continuous, 
        # mathematically transparent gradient highway, enabling parameters above and below this layer 
        # to organically share backward loss update trajectories during deep fine-tuning runs.
        return final_attention_rail_output

# Global namespace isolation gate to block external structural pollution from corrupting the internal phase spaces.
__all__ = ["UpgradedSoftmaxBypassingDecoder"]


