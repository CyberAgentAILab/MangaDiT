import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora
import torch.nn as nn
import math

def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
    timestep: Optional[float] = 0.0,
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    # Input hidden_states have two types, with or without encoder_hidden_states
    # If encoder_hidden_states==None, hidden_states [1, 512+1024, 3072], else [1, 1024, 3072]

    # 1. Calculate original attention
    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if condition_latents is not None:
        # Split condition sequence
        cond_seq_len = cond_query.shape[2] // 2
        
        # Original condition attention computation
        orig_cond_query = cond_query[:, :, :cond_seq_len, :]
        orig_cond_key = cond_key[:, :, :cond_seq_len, :]
        orig_cond_value = cond_value[:, :, :cond_seq_len, :]
        
        # Pooled condition attention computation
        pool_cond_query = cond_query[:, :, cond_seq_len:, :]
        pool_cond_key = cond_key[:, :, cond_seq_len:, :]
        pool_cond_value = cond_value[:, :, cond_seq_len:, :]

        # Concatenate with main sequence for joint attention
        query = torch.cat([query, orig_cond_query, pool_cond_query], dim=2)
        key = torch.cat([key, orig_cond_key, pool_cond_key], dim=2)
        value = torch.cat([value, orig_cond_value, pool_cond_value], dim=2)

    if not model_config.get("union_cond_attn", True):
        # If we don't want to use the union condition attention, we need to mask the attention
        # between the hidden states and the condition latents
        attention_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
        )
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False
        attention_mask[:-condition_n, -condition_n:] = False
    elif model_config.get("independent_condition", False):
        attention_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
        )
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False
    if hasattr(attn, "c_factor"):
        attention_mask = torch.zeros(
            query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
        )
        condition_n = cond_query.shape[2]
        bias = torch.log(attn.c_factor[0])
        attention_mask[-condition_n:, :-condition_n] = bias
        attention_mask[:-condition_n, -condition_n:] = bias

    # 2. Calculate original attention
    original_hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )

    # Q: add new attention with dual pooled condition
    if condition_latents is not None:
        # 3.1 Apply pooling to hidden_states
        # Check if encoder_hidden_states is None
        if encoder_hidden_states is None:
            # If None, hidden_states already contains encoder_hidden_states
            # Need to extract the last 1024 part for pooling
            hidden_states_for_pooling = hidden_states[:, -1024:, :]
        else:
            # If not None, use hidden_states directly
            hidden_states_for_pooling = hidden_states
        
        # Reshape to spatial form
        hidden_seq_len = hidden_states_for_pooling.shape[1]
        hidden_height = int(hidden_seq_len ** 0.5)  # Assume square
        hidden_width = hidden_height

        # Reshape to spatial form
        spatial_hidden = hidden_states_for_pooling.reshape(batch_size, hidden_height, hidden_width, -1)
        
        # Randomly select pooling region size
        pool_sizes = [2, 4, 8]
        pool_size = pool_sizes[torch.randint(0, len(pool_sizes), (1,)).item()]
        
        # Calculate max value of random size region
        pooled_hidden = F.max_pool2d(
            spatial_hidden.permute(0, 3, 1, 2),  # [B, C, H, W]
            kernel_size=pool_size,
            stride=pool_size
        )
        
        # Copy mean value to corresponding region
        upsampled_hidden = F.interpolate(
            pooled_hidden,
            scale_factor=pool_size,
            mode='nearest'
        )
        
        # Convert back to sequence form
        processed_hidden = upsampled_hidden.permute(0, 2, 3, 1).reshape(
            batch_size, hidden_seq_len, -1
        )
        
        # If encoder_hidden_states is None, need to concatenate processed result with front part
        if encoder_hidden_states is None:
            processed_hidden = torch.cat([hidden_states[:, :512, :], processed_hidden], dim=1)
        
        # 3.2 Apply pooling to condition_latents
        # Assume condition_latents contains two 1024 images
        cond_seq_len = condition_latents.shape[1]
        cond_seq_len_per_image = cond_seq_len // 2  # Sequence length per image
        
        # Split two images
        cond_image1 = condition_latents[:, :cond_seq_len_per_image, :]
        cond_image2 = condition_latents[:, cond_seq_len_per_image:, :]
        
        # Apply pooling to first image
        cond_height1 = int(cond_seq_len_per_image ** 0.5)  # Assume square
        cond_width1 = cond_height1
        
        # Reshape to spatial form
        spatial_cond1 = cond_image1.reshape(batch_size, cond_height1, cond_width1, -1)
        
        # Randomly select pooling region size
        pool_size = pool_sizes[torch.randint(0, len(pool_sizes), (1,)).item()]
        
        # Calculate mean value of random size region
        pooled_cond1 = F.max_pool2d(
            spatial_cond1.permute(0, 3, 1, 2),  # [B, C, H, W]
            kernel_size=pool_size,
            stride=pool_size
        )
        
        # Copy mean value to corresponding region
        upsampled_cond1 = F.interpolate(
            pooled_cond1,
            scale_factor=pool_size,
            mode='nearest'
        )
        
        # Convert back to sequence form
        processed_condition1 = upsampled_cond1.permute(0, 2, 3, 1).reshape(
            batch_size, cond_seq_len_per_image, -1
        )
        
        # Apply pooling to second image
        cond_height2 = int(cond_seq_len_per_image ** 0.5)  # Assume square
        cond_width2 = cond_height2
        
        # Reshape to spatial form
        spatial_cond2 = cond_image2.reshape(batch_size, cond_height2, cond_width2, -1)
        
        # Randomly select pooling region size
        pool_size = pool_sizes[torch.randint(0, len(pool_sizes), (1,)).item()]
        
        # Calculate mean value of random size region
        pooled_cond2 = F.max_pool2d(
            spatial_cond2.permute(0, 3, 1, 2),  # [B, C, H, W]
            kernel_size=pool_size,
            stride=pool_size
        )
        
        # Copy mean value to corresponding region
        upsampled_cond2 = F.interpolate(
            pooled_cond2,
            scale_factor=pool_size,
            mode='nearest'
        )
        
        # Convert back to sequence form
        processed_condition2 = upsampled_cond2.permute(0, 2, 3, 1).reshape(
            batch_size, cond_seq_len_per_image, -1
        )
        
        # Combine two processed images back
        processed_condition = torch.cat([processed_condition1, processed_condition2], dim=1)
        
        # 3.3 Calculate new attention
        with enable_lora(
            (attn.to_q, attn.to_k, attn.to_v), 
            model_config.get("latent_lora", False)
        ):
            # Project processed hidden_states
            spatial_query = attn.to_q(processed_hidden)
            spatial_key = attn.to_k(processed_hidden)
            
            # Project processed condition_latents
            spatial_cond_query = attn.to_q(processed_condition)
            spatial_cond_key = attn.to_k(processed_condition)
            
            # Reshape to multi-head form
            spatial_query = spatial_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            spatial_key = spatial_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            spatial_cond_query = spatial_cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            spatial_cond_key = spatial_cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Apply same normalization
            if attn.norm_q is not None:
                spatial_query = attn.norm_q(spatial_query)
                spatial_cond_query = attn.norm_q(spatial_cond_query)
            if attn.norm_k is not None:
                spatial_key = attn.norm_k(spatial_key)
                spatial_cond_key = attn.norm_k(spatial_cond_key)
                
            # encoder_hidden_states from text context
            if encoder_hidden_states is not None:
                spatial_query = torch.cat([encoder_hidden_states_query_proj, spatial_query], dim=2)
                spatial_key = torch.cat([encoder_hidden_states_key_proj, spatial_key], dim=2)

            if image_rotary_emb is not None:
                spatial_query = apply_rotary_emb(spatial_query, image_rotary_emb)
                spatial_key = apply_rotary_emb(spatial_key, image_rotary_emb)

            if cond_rotary_emb is not None:
                spatial_cond_query = apply_rotary_emb(spatial_cond_query, cond_rotary_emb)
                spatial_cond_key = apply_rotary_emb(spatial_cond_key, cond_rotary_emb)
            
            # Concatenate query, key, keep value unchanged
            spatial_query = torch.cat([spatial_query, spatial_cond_query], dim=2)
            spatial_key = torch.cat([spatial_key, spatial_cond_key], dim=2)
            
            # Calculate new attention
            spatial_hidden_states = F.scaled_dot_product_attention(
                spatial_query, spatial_key, value,
                dropout_p=0.0, is_causal=False,
                attn_mask=attention_mask  # Use same attention mask
            )

        # 3.4 Mix two attention maps
        
        # Use cosine schedule, gradually introduce pooled information in early-mid stage, weaken in late stage
        base_weight = model_config.get("hier_weight", False)
        
        # Convert timestep back to 0-1 range
        if isinstance(timestep, torch.Tensor):
            t = timestep.item() / 1000.0  # Convert to scalar and divide by 1000
        else:
            t = timestep / 1000.0        
        weight = base_weight * 0.5 * (1 - math.cos(math.pi * t))
        hidden_states = original_hidden_states + weight * spatial_hidden_states
        
    else:
        hidden_states = original_hidden_states

    # 4. Reshape and type conversion
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    # 5. Process output
    if encoder_hidden_states is not None:
        if condition_latents is not None:
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            condition_latents = attn.to_out[0](condition_latents)
            condition_latents = attn.to_out[1](condition_latents)

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        # if there are condition_latents, we need to separate the hidden_states and the condition_latents
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
    timestep: Optional[float] = 0.0,
):
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    if use_cond:
        (
            norm_condition_latents,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = self.norm1(condition_latents, emb=cond_temb)

    # Attention.
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        timestep=timestep,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        condition_latents = condition_latents + cond_attn_output
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = self.norm2(condition_latents)
        norm_condition_latents = (
            norm_condition_latents * (1 + cond_scale_mlp[:, None])
            + cond_shift_mlp[:, None]
        )

    # Feed-forward.
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        # 1. hidden_states
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    # 3. condition_latents
    if use_cond:
        cond_ff_output = self.ff(norm_condition_latents)
        cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    cond_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):

    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora(
        (
            self.norm.linear,
            self.proj_mlp,
        ),
        model_config.get("latent_lora", False),
    ):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = self.norm(condition_latents, emb=cond_temb)
        mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
        cond_gate = cond_gate.unsqueeze(1)
        condition_latents = cond_gate * self.proj_out(condition_latents)
        condition_latents = residual_cond + condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)
