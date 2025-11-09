import torch
import torch.nn as nn
from einops import pack, rearrange, unpack

from .common_blocks import (
    QKV,
    ActionPositionEmb,
    AdaLayerNormZero,
    FeedForward,
    FinalLayer,
    JointAttention,
    PatchVideo,
    PatchVideoTempMask,
    SinusoidalPosEmb,
    VideoLearnedPositionEmb,
    VideoPositionEmb,
    gate,
    interleave_masks_1d,
    interleave_masks_2d,
    modulate,
)


class MMDiTBlock(nn.Module):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        super().__init__()
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = True

        self.time_scale_shift = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=3
        )
        # notice how elementwise_affine=False because we are using AdaLNZero blocks

        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_fa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)

        self.joint_attn = JointAttention(
            token_dim,
            num_heads=self.num_heads,
        )

        self.ff_cv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            self.ff_pv = FeedForward(token_dim, act=self.act)
            self.ff_ca = FeedForward(token_dim, act=self.act)
            self.ff_pa = FeedForward(token_dim, act=self.act)

        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()

    def forward(self, fv, pv, fa, pa, timesteps, video_pos_embed, action_pos_embed):
        """
        fv - future video
        pv - past video
        fa - future actions
        pa - past actions
        """
        h, w = fv.shape[-2], fv.shape[-1]

        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
            pv_pre_attn_gamma,
            pv_post_attn_gamma,
            pv_pre_ff_gamma,
            pv_post_ff_gamma,
            pv_pre_attn_beta,
            pv_pre_ff_beta,
            fa_pre_attn_gamma,
            fa_post_attn_gamma,
            fa_pre_ff_gamma,
            fa_post_ff_gamma,
            fa_pre_attn_beta,
            fa_pre_ff_beta,
            pa_pre_attn_gamma,
            pa_post_attn_gamma,
            pa_pre_ff_gamma,
            pa_post_ff_gamma,
            pa_pre_attn_beta,
            pa_pre_ff_beta,
        ) = self.time_scale_shift(timesteps)

        fv = rearrange(fv, "b d t h w -> b (t h w) d")
        pv = rearrange(pv, "b d t h w -> b (t h w) d")

        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta)
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta)
        fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta)
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta)

        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_pv(pv_res)
        q_fa, k_fa, v_fa = self.qkv_fa(fa_res)
        q_pa, k_pa, v_pa = self.qkv_pa(pa_res)

        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        fv_res, pv_res, fa_res, pa_res = self.joint_attn(
            [
                (q_fv, k_fv, v_fv),
                (q_pv, k_pv, v_pv),
                (q_fa, k_fa, v_fa),
                (q_pa, k_pa, v_pa),
            ]
        )

        fv = fv + gate(fv_res, fv_post_attn_gamma)
        pv = pv + gate(pv_res, pv_post_attn_gamma)
        fa = fa + gate(fa_res, fa_post_attn_gamma)
        pa = pa + gate(pa_res, pa_post_attn_gamma)

        fv_res = modulate(self.ff_norm_cv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_cv(fv_res)
        fv = fv + gate(fv_res, fv_post_ff_gamma)
        fv = rearrange(fv, "b (t h w) d -> b d t h w", h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta)
            fa_res = modulate(self.ff_norm_ca(fa), fa_pre_ff_gamma, fa_pre_ff_beta)
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta)

            pv_res = self.ff_pv(pv_res)
            fa_res = self.ff_ca(fa_res)
            pa_res = self.ff_pa(pa_res)

            pv = pv + gate(pv_res, pv_post_ff_gamma)
            fa = fa + gate(fa_res, fa_post_ff_gamma)
            pa = pa + gate(pa_res, pa_post_ff_gamma)

        pv = rearrange(pv, "b (t h w) d -> b d t h w", h=h, w=w)
        return fv, pv, fa, pa

    def pos_embed_pf(self, p, f, pos_embedder):
        a, pack_info = pack([p, f], "b h * d")
        a = pos_embedder(a)
        p, f = unpack(a, pack_info, "b h * d")
        return p, f

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.qkv_fv.apply(_basic_init)
        self.qkv_pv.apply(_basic_init)
        self.qkv_fa.apply(_basic_init)
        self.qkv_pa.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_cv.apply(_basic_init)
            self.ff_pv.apply(_basic_init)
            self.ff_ca.apply(_basic_init)
            self.ff_pa.apply(_basic_init)


class VideoDiTModel(nn.Module):
    """
    Standard MMDiT style video model, predicts future video sequence.
    Each of the four streams use their own MLPs,
        but communicate with a shared joint attention block.
    """

    def __init__(
        self,
        dim_C,
        dim_T_past,
        dim_T_future,
        dim_L_past,
        dim_L_future,
        dim_W,
        dim_h,
        dim_act,
        dim_hidden,
        patch_lw,
        n_layers,
        n_head,
        cfg_prob,
        patch_t=1,
        device="cuda",
        add_temp_mask=False,
    ):
        super().__init__()
        self.dim_Cf = dim_C
        self.n_layers = n_layers
        self.n_head = n_head
        self.patch_lw = patch_lw
        self.patch_t = patch_t

        self.dim_C, self.dim_Tf, self.dim_Tp, self.dim_H, self.dim_W = (
            dim_C,
            dim_T_future,
            dim_T_past,
            dim_h,
            dim_W,
        )
        self.dim_Lp = dim_L_past
        self.dim_Lf = dim_L_future

        self.dim_act = dim_act
        self.dim_hidden = dim_hidden
        self.dim_head = self.dim_hidden // self.n_head
        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(self.dim_hidden, theta=10000),
            nn.Linear(self.dim_hidden, self.dim_hidden * 4),
            nn.SiLU(),
            nn.Linear(self.dim_hidden * 4, self.dim_hidden),
        )

        self.add_temp_mask = add_temp_mask
        if add_temp_mask:
            self.action_embedder = nn.Sequential(
                nn.Linear(self.dim_act + 1, self.dim_hidden * 4),
                nn.SiLU(),
                nn.Linear(self.dim_hidden * 4, self.dim_hidden),
            )
            self.patcher_f = PatchVideoTempMask(
                dim_c=self.dim_C,
                dim_t=self.dim_Lf,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )
            self.patcher_p = PatchVideoTempMask(
                dim_c=self.dim_C,
                dim_t=self.dim_Lp,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )
        else:
            self.action_embedder = nn.Sequential(
                nn.Linear(self.dim_act, self.dim_hidden * 4),
                nn.SiLU(),
                nn.Linear(self.dim_hidden * 4, self.dim_hidden),
            )
            self.patcher_f = PatchVideo(
                dim_c=self.dim_Cf,
                dim_t=self.dim_Lf,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )
            self.patcher_p = PatchVideo(
                dim_c=self.dim_C,
                dim_t=self.dim_Lp,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )

        self.action_pos_embed = ActionPositionEmb(
            self.dim_Tp + self.dim_Tf, self.dim_head, theta=10000.0
        )  # both future and past tokens simultaneously
        self.video_pos_embed = VideoPositionEmb(
            head_dim=self.dim_head,
            len_h=self.dim_H // self.patch_lw,
            len_w=self.dim_W // self.patch_lw,
            len_t=self.dim_Lp
            + self.dim_Lf,  # notice how we embed both future and past tokens simultaneously
            theta=10000.0,
            device=device,
        )
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = None
            if i == self.n_layers - 1:
                block = MMDiTBlock(
                    self.dim_hidden,
                    self.dim_hidden,
                    num_heads=self.n_head,
                    skip_context_ff=True,
                )
            else:
                block = MMDiTBlock(
                    self.dim_hidden,
                    self.dim_hidden,
                    num_heads=self.n_head,
                )
            self.blocks.append(block)
        self.final_layer = FinalLayer(
            self.dim_hidden,
            patch_lw=self.patch_lw,
            patch_t=self.patch_t,
            out_channels=self.dim_Cf,
        )

        self.register_buffer(
            "empty_past_frames_emb",
            torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)),
        )

        self.register_buffer(
            "empty_past_actions_emb", torch.zeros((self.dim_Tp, self.dim_act))
        )

        self.register_buffer(
            "empty_future_actions_emb", torch.zeros((self.dim_Tf, self.dim_act))
        )

        self.cfg_prob = cfg_prob
        self.initialize_weights()

    def context_drop(self, batch, use_cfg, device, force_drop_context=False):
        """USING TORCH.CONTEXT
        Drops labels to enable classifier-free guidance.
        """
        b = batch["noisy_latents"].shape[0]
        if force_drop_context == False and use_cfg:
            drop_ids = torch.rand(b, device=device) < self.cfg_prob
            batch["past_latents"][drop_ids, :] = self.empty_past_frames_emb.to(device)
            batch["past_actions"][drop_ids, :] = self.empty_past_actions_emb.to(device)
            batch["future_actions"][drop_ids, :] = self.empty_future_actions_emb.to(
                device
            )
        elif force_drop_context == False and use_cfg == False:
            pass
        elif force_drop_context == True:
            batch["past_latents"] = self.empty_past_frames_emb.repeat(b, 1, 1, 1, 1).to(
                device
            )
            batch["past_actions"] = self.empty_past_actions_emb.repeat(b, 1, 1).to(
                device
            )
            b = batch["noisy_latents"].shape[0]
            batch["future_actions"] = self.empty_future_actions_emb.repeat(b, 1, 1).to(
                device
            )
        return batch

    def forward(
        self, batch, time, device="cuda:1", force_drop_context=False, use_cfg=False
    ):
        device = batch["noisy_latents"].device

        batch = self.context_drop(
            batch, use_cfg, device, force_drop_context=force_drop_context
        )
        fv = batch["noisy_latents"]
        pv = batch["past_latents"]
        pa = batch["past_actions"]
        fa = batch["future_actions"]

        if self.add_temp_mask:
            b = batch["noisy_latents"].shape[0]
            fv = interleave_masks_2d(
                batch["noisy_latents"], torch.zeros((b, self.dim_Lf))
            )
            pv = interleave_masks_2d(
                batch["past_latents"], torch.ones((b, self.dim_Lp))
            )
            pa = interleave_masks_1d(
                batch["past_actions"], torch.ones((b, self.dim_Tp))
            )
            fa = interleave_masks_1d(
                batch["future_actions"], torch.zeros((b, self.dim_Tf))
            )

        pa = self.action_embedder(pa)
        fa = self.action_embedder(fa)
        fv = self.patcher_f(fv)
        pv = self.patcher_p(pv)
        time = self.time_embedder(time)

        for i in range(self.n_layers):
            fv, pv, fa, pa = self.blocks[i](
                fv, pv, fa, pa, time, self.video_pos_embed, self.action_pos_embed
            )

        fv = self.final_layer(fv, time)
        fv = self.patcher_f.unpatchify(fv)
        return fv

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.action_embedder.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder[1].weight, std=0.02)
        nn.init.normal_(self.time_embedder[3].weight, std=0.02)


class MMDiTBlockFutureFrame(nn.Module):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        super().__init__()
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = True

        self.time_scale_shift = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=2
        )
        # notice how elementwise_affine=False because we are using AdaLNZero blocks

        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_pa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)

        self.joint_attn = JointAttention(
            token_dim,
            num_heads=self.num_heads,
        )

        self.ff_cv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            self.ff_pv = FeedForward(token_dim, act=self.act)
            self.ff_pa = FeedForward(token_dim, act=self.act)

        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.initialize_weights()

    def forward(self, fv, pv, pa, timesteps, video_pos_embed, action_pos_embed):
        """
        fv - future video
        pv - past video
        fa - future actions
        pa - past actions
        """
        h, w = fv.shape[-2], fv.shape[-1]

        (
            fv_pre_attn_gamma,
            fv_post_attn_gamma,
            fv_pre_ff_gamma,
            fv_post_ff_gamma,
            fv_pre_attn_beta,
            fv_pre_ff_beta,
            pv_pre_attn_gamma,
            pv_post_attn_gamma,
            pv_pre_ff_gamma,
            pv_post_ff_gamma,
            pv_pre_attn_beta,
            pv_pre_ff_beta,
            pa_pre_attn_gamma,
            pa_post_attn_gamma,
            pa_pre_ff_gamma,
            pa_post_ff_gamma,
            pa_pre_attn_beta,
            pa_pre_ff_beta,
        ) = self.time_scale_shift(timesteps)

        fv = rearrange(fv, "b d t h w -> b (t h w) d")
        pv = rearrange(pv, "b d t h w -> b (t h w) d")

        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta)
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta)
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta)

        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_pv(pv_res)
        q_pa, k_pa, v_pa = self.qkv_pa(pa_res)

        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa = action_pos_embed(q_pa)
        k_pa = action_pos_embed(k_pa)

        fv_res, pv_res, pa_res = self.joint_attn(
            [
                (q_fv, k_fv, v_fv),
                (q_pv, k_pv, v_pv),
                (q_pa, k_pa, v_pa),
            ]
        )

        fv = fv + gate(fv_res, fv_post_attn_gamma)
        pv = pv + gate(pv_res, pv_post_attn_gamma)
        pa = pa + gate(pa_res, pa_post_attn_gamma)

        fv_res = modulate(self.ff_norm_cv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_cv(fv_res)
        fv = fv + gate(fv_res, fv_post_ff_gamma)
        fv = rearrange(fv, "b (t h w) d -> b d t h w", h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta)
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta)

            pv_res = self.ff_pv(pv_res)
            pa_res = self.ff_pa(pa_res)

            pv = pv + gate(pv_res, pv_post_ff_gamma)
            pa = pa + gate(pa_res, pa_post_ff_gamma)

        pv = rearrange(pv, "b (t h w) d -> b d t h w", h=h, w=w)
        return fv, pv, pa

    def pos_embed_pf(self, p, f, pos_embedder):
        a, pack_info = pack([p, f], "b h * d")
        a = pos_embedder(a)
        p, f = unpack(a, pack_info, "b h * d")
        return p, f

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.qkv_fv.apply(_basic_init)
        self.qkv_pv.apply(_basic_init)
        self.qkv_pa.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_cv.apply(_basic_init)
            self.ff_pv.apply(_basic_init)
            self.ff_pa.apply(_basic_init)


class VideoDiTFutureFrameModel(VideoDiTModel):
    """
    Standard MMDiT style video model, predicts a single future frame.
    Each of the four streams use their own MLPs,
        but communicate with a shared joint attention block.
    """

    def __init__(
        self,
        dim_C,
        dim_T_past,
        dim_T_future,
        dim_L_past,
        dim_L_future,
        dim_W,
        dim_h,
        dim_act,
        dim_hidden,
        patch_lw,
        n_layers,
        n_head,
        cfg_prob,
        dim_intermediate=59,
        discrete_time=True,
        patch_t=1,
        device="cuda",
        add_temp_mask=False,
    ):
        nn.Module.__init__(self)
        self.dim_Cf = dim_C
        self.n_layers = n_layers
        self.n_head = n_head
        self.patch_lw = patch_lw
        self.patch_t = patch_t

        self.dim_C, self.dim_Tf, self.dim_Tp, self.dim_H, self.dim_W = (
            dim_C,
            dim_T_future,
            dim_T_past,
            dim_h,
            dim_W,
        )
        self.dim_Lp = dim_L_past
        self.dim_Lf = dim_L_future

        self.dim_act = dim_act
        self.dim_hidden = dim_hidden
        self.dim_head = self.dim_hidden // self.n_head
        self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(self.dim_hidden, theta=10000),
            nn.Linear(self.dim_hidden, self.dim_hidden * 4),
            nn.SiLU(),
            nn.Linear(self.dim_hidden * 4, self.dim_hidden),
        )

        self.add_temp_mask = add_temp_mask
        if add_temp_mask:
            self.action_embedder = nn.Sequential(
                nn.Linear(self.dim_act + 1, self.dim_hidden * 4),
                nn.SiLU(),
                nn.Linear(self.dim_hidden * 4, self.dim_hidden),
            )
            self.patcher_f = PatchVideoTempMask(
                dim_c=self.dim_C,
                dim_t=self.dim_Lf,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )
            self.patcher_p = PatchVideoTempMask(
                dim_c=self.dim_C,
                dim_t=self.dim_Lp,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )
        else:
            self.action_embedder = nn.Sequential(
                nn.Linear(self.dim_act, self.dim_hidden * 4),
                nn.SiLU(),
                nn.Linear(self.dim_hidden * 4, self.dim_hidden),
            )
            self.patcher_f = PatchVideo(
                dim_c=self.dim_Cf,
                dim_t=self.dim_Lf,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )
            self.patcher_p = PatchVideo(
                dim_c=self.dim_C,
                dim_t=self.dim_Lp,
                dim_h=self.dim_H,
                dim_w=self.dim_W,
                dim_hidden=self.dim_hidden,
                patch_s=self.patch_lw,
                patch_t=self.patch_t,
            )

        self.action_pos_embed = ActionPositionEmb(
            self.dim_Tp + dim_intermediate + self.dim_Tf, self.dim_head, theta=10000.0
        )  # both future and past tokens simultaneously
        self.video_pos_embed = VideoLearnedPositionEmb(
            head_dim=self.dim_head,
            len_h=self.dim_H // self.patch_lw,
            len_w=self.dim_W // self.patch_lw,
            len_t=self.dim_Lp
            + self.dim_Lf,  # notice how we embed both future and past tokens simultaneously
            theta=10000.0,
            device=device,
        )
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = None
            if i == self.n_layers - 1:
                block = MMDiTBlockFutureFrame(
                    self.dim_hidden,
                    self.dim_hidden,
                    num_heads=self.n_head,
                    skip_context_ff=True,
                )
            else:
                block = MMDiTBlockFutureFrame(
                    self.dim_hidden,
                    self.dim_hidden,
                    num_heads=self.n_head,
                )
            self.blocks.append(block)
        self.final_layer = FinalLayer(
            self.dim_hidden,
            patch_lw=self.patch_lw,
            patch_t=self.patch_t,
            out_channels=self.dim_Cf,
        )

        self.register_buffer(
            "empty_past_frames_emb",
            torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)),
        )
        # self.empty_past_frames_emb = nn.Parameter(torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)))

        self.register_buffer(
            "empty_past_actions_emb",
            torch.zeros((self.dim_Tp + dim_intermediate + self.dim_Tf, self.dim_act)),
        )
        # self.empty_past_actions_emb = nn.Parameter(torch.zeros((self.dim_Tp, self.dim_act)))

        self.cfg_prob = cfg_prob
        # self.conditioning_manager = conditioning_manager
        # self.conditioning = conditioning
        self.initialize_weights()

    def context_drop(self, batch, use_cfg, device, force_drop_context=False):
        batch = batch.copy()
        b = batch["noisy_latents"].shape[0]
        if force_drop_context == False and use_cfg:
            drop_ids = torch.rand(b, device=device) < self.cfg_prob
            batch["past_latents"][drop_ids, :] = self.empty_past_frames_emb.to(device)
            batch["past_actions"][drop_ids, :] = self.empty_past_actions_emb.to(device)
        elif force_drop_context == False and use_cfg == False:
            pass
        elif force_drop_context == True:
            # batch['past_latents'] = self.empty_past_frames_emb.repeat(b,1,1,h,w).to(device)
            # batch['past_actions'] = self.empty_past_actions_emb.repeat(b,tp,1).to(device)
            # b, _, tf, _, _ = batch['noisy_latents'].shape
            # batch['future_actions'] = self.empty_future_actions_emb.repeat(b,tf,1).to(device)
            batch["past_latents"] = self.empty_past_frames_emb.repeat(b, 1, 1, 1, 1).to(
                device
            )
            batch["past_actions"] = self.empty_past_actions_emb.repeat(b, 1, 1).to(
                device
            )
            b = batch["noisy_latents"].shape[0]
        return batch

    def forward(
        self, batch, time, device="cuda:1", force_drop_context=False, use_cfg=False
    ):
        device = batch["noisy_latents"].device

        batch = self.context_drop(
            batch, use_cfg, device, force_drop_context=force_drop_context
        )
        fv = batch["noisy_latents"]
        pv = batch["past_latents"]
        pa = batch["past_actions"].to(torch.float32)

        if self.add_temp_mask:
            b = batch["noisy_latents"].shape[0]
            fv = interleave_masks_2d(
                batch["noisy_latents"], torch.zeros((b, self.dim_Lf))
            )
            pv = interleave_masks_2d(
                batch["past_latents"], torch.ones((b, self.dim_Lp))
            )
            pa = interleave_masks_1d(
                batch["past_actions"], torch.ones((b, self.dim_Tp))
            )

        pa = self.action_embedder(pa)
        fv = self.patcher_f(fv)
        pv = self.patcher_p(pv)
        time = self.time_embedder(time)

        for i in range(self.n_layers):
            fv, pv, pa = self.blocks[i](
                fv, pv, pa, time, self.video_pos_embed, self.action_pos_embed
            )

        fv = self.final_layer(fv, time)
        fv = self.patcher_f.unpatchify(fv)
        return fv

    def initialize_weights(self):
        # return
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.action_embedder.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder[1].weight, std=0.02)
        nn.init.normal_(self.time_embedder[3].weight, std=0.02)

        # Zero-out output layers:
        # nn.init.xavier_uniform_(self.final_layer.linear.weight, gain=.5)
        # nn.init.xavier_uniform_(self.final_layer.linear.bias)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.normal_(self.final_layer.linear.weight, 0.0, 0.02)
        # nn.init.constant_(self.final_layer.linear.bias, 0)
