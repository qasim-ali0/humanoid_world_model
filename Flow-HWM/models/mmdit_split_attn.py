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
    VideoPositionEmb,
    gate,
    modulate,
)
from .mmdit import MMDiTBlock, VideoDiTModel


class MMDiTSplitAttentionBlock(MMDiTBlock):
    def __init__(self, token_dim, time_dim, num_heads, skip_context_ff=False):
        nn.Module.__init__(self)
        self.token_dim = token_dim
        self.act = nn.GELU()
        self.num_heads = num_heads
        self.qk_norm = True

        self.time_scale_shift_v = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=0
        )
        self.time_scale_shift_a = AdaLayerNormZero(
            time_dim, token_dim, param_factor=6, n_context=0
        )
        self.time_scale_shift_cross = AdaLayerNormZero(
            time_dim, token_dim, param_factor=3, n_context=0
        )
        # notice how elementwise_affine=False because we are using AdaLNZero blocks

        self.attn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.qkv_fv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        # self.qkv_pv = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.qkv_fa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        # self.qkv_pa = QKV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)

        self.crossattn_norm_cv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.crossattn_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.crossattn_norm_ca = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.crossattn_norm_pa = nn.LayerNorm(token_dim, elementwise_affine=False)

        self.q_fv = Q(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.kv_pv = KV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.kv_fa = KV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)
        self.kv_pa = KV(token_dim, num_heads=self.num_heads, qk_norm=self.qk_norm)

        self.joint_attns = nn.ModuleList(
            [
                JointAttention(
                    token_dim,
                    num_heads=self.num_heads,
                )
                for _ in range(2)
            ]
        )
        self.cross_attn = JointAttention(
            token_dim,
            num_heads=self.num_heads,
        )

        self.ff_fv = FeedForward(token_dim, act=self.act)
        self.skip_context_ff = skip_context_ff
        if not skip_context_ff:
            # self.ff_pv = FeedForward(token_dim, act=self.act)
            self.ff_fa = FeedForward(token_dim, act=self.act)
            # self.ff_pa = FeedForward(token_dim, act=self.act)

        # notice how elementwise_affine=False because we are using AdaLNZero blocks
        self.ff_norm_fv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_pv = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.ff_norm_fa = nn.LayerNorm(token_dim, elementwise_affine=False)
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
        ) = self.time_scale_shift_v(timesteps)

        pv_pre_attn_gamma = fv_pre_attn_gamma
        pv_post_attn_gamma = fv_post_attn_gamma
        pv_pre_ff_gamma = fv_pre_ff_gamma
        pv_post_ff_gamma = fv_post_ff_gamma
        pv_pre_attn_beta = fv_pre_attn_beta
        pv_pre_ff_beta = fv_pre_ff_beta

        (
            fa_pre_attn_gamma,
            fa_post_attn_gamma,
            fa_pre_ff_gamma,
            fa_post_ff_gamma,
            fa_pre_attn_beta,
            fa_pre_ff_beta,
        ) = self.time_scale_shift_a(timesteps)

        pa_pre_attn_gamma = fa_pre_attn_gamma
        pa_post_attn_gamma = fa_post_attn_gamma
        pa_pre_ff_gamma = fa_pre_ff_gamma
        pa_post_ff_gamma = fa_post_ff_gamma
        pa_pre_attn_beta = fa_pre_attn_beta
        pa_pre_ff_beta = fa_pre_ff_beta

        (fv_pre_crossattn_gamma, fv_post_crossattn_gamma, fv_pre_crossattn_beta) = (
            self.time_scale_shift_cross(timesteps)
        )
        fv = rearrange(fv, "b d t h w -> b (t h w) d")
        pv = rearrange(pv, "b d t h w -> b (t h w) d")

        fv_res = modulate(self.attn_norm_cv(fv), fv_pre_attn_gamma, fv_pre_attn_beta)
        pv_res = modulate(self.attn_norm_pv(pv), pv_pre_attn_gamma, pv_pre_attn_beta)
        fa_res = modulate(self.attn_norm_ca(fa), fa_pre_attn_gamma, fa_pre_attn_beta)
        pa_res = modulate(self.attn_norm_pa(pa), pa_pre_attn_gamma, pa_pre_attn_beta)

        q_fv, k_fv, v_fv = self.qkv_fv(fv_res)
        q_pv, k_pv, v_pv = self.qkv_fv(pv_res)
        q_fa, k_fa, v_fa = self.qkv_fa(fa_res)
        q_pa, k_pa, v_pa = self.qkv_fa(pa_res)

        q_pv, q_fv = self.pos_embed_pf(q_pv, q_fv, video_pos_embed)
        k_pv, k_fv = self.pos_embed_pf(k_pv, k_fv, video_pos_embed)

        q_pa, q_fa = self.pos_embed_pf(q_pa, q_fa, action_pos_embed)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        fv_res = self.joint_attns[0]([(q_fv, k_fv, v_fv)])[0]
        pv_res = self.joint_attns[0]([(q_pv, k_pv, v_pv)])[0]
        fa_res = self.joint_attns[1]([(q_fa, k_fa, v_fa)])[0]
        pa_res = self.joint_attns[1]([(q_pa, k_pa, v_pa)])[0]

        fv = fv + gate(fv_res, fv_post_attn_gamma)
        pv = pv + gate(pv_res, pv_post_attn_gamma)
        fa = fa + gate(fa_res, fa_post_attn_gamma)
        pa = pa + gate(pa_res, pa_post_attn_gamma)

        ##### Cross attention branch
        fv_res = modulate(
            self.crossattn_norm_cv(fv), fv_pre_crossattn_gamma, fv_pre_crossattn_beta
        )
        pv_res = self.crossattn_norm_pv(pv)
        fa_res = self.crossattn_norm_ca(fa)
        pa_res = self.crossattn_norm_pa(pa)

        q_fv = self.q_fv(fv_res)
        k_pv, v_pv = self.kv_pv(pv_res)
        k_fa, v_fa = self.kv_fa(fa_res)
        k_pa, v_pa = self.kv_pa(pa_res)

        q_fv = video_pos_embed(q_fv)
        k_pv = video_pos_embed(k_pv)
        k_pa, k_fa = self.pos_embed_pf(k_pa, k_fa, action_pos_embed)

        k = torch.cat((k_pv, k_fa, k_pa), 2)
        v = torch.cat((v_pv, v_fa, v_pa), 2)
        fv_res = self.cross_attn([(q_fv, k, v)])[0]

        fv = fv + gate(fv_res, fv_post_crossattn_gamma)
        #####

        fv_res = modulate(self.ff_norm_fv(fv), fv_pre_ff_gamma, fv_pre_ff_beta)
        fv_res = self.ff_fv(fv_res)
        fv = fv + gate(fv_res, fv_post_ff_gamma)
        fv = rearrange(fv, "b (t h w) d -> b d t h w", h=h, w=w)

        if not self.skip_context_ff:
            pv_res = modulate(self.ff_norm_pv(pv), pv_pre_ff_gamma, pv_pre_ff_beta)
            fa_res = modulate(self.ff_norm_fa(fa), fa_pre_ff_gamma, fa_pre_ff_beta)
            pa_res = modulate(self.ff_norm_pa(pa), pa_pre_ff_gamma, pa_pre_ff_beta)

            pv_res = self.ff_fv(pv_res)
            fa_res = self.ff_fa(fa_res)
            pa_res = self.ff_fa(pa_res)

            pv = pv + gate(pv_res, pv_post_ff_gamma)
            fa = fa + gate(fa_res, fa_post_ff_gamma)
            pa = pa + gate(pa_res, pa_post_ff_gamma)

        pv = rearrange(pv, "b (t h w) d -> b d t h w", h=h, w=w)
        return fv, pv, fa, pa

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.qkv_fv.apply(_basic_init)
        self.qkv_fa.apply(_basic_init)
        if not self.skip_context_ff:
            self.ff_fv.apply(_basic_init)
            self.ff_fa.apply(_basic_init)


class VideoDiTSplitAttnModel(VideoDiTModel):
    """
    Variant of MMDiT, predicts future video sequence.
    Rather the joint self-attention between all the tokens,
        attention is factorized into spatial and temporal.
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
        discrete_time=True,
        patch_t=1,
        device="cuda",
        add_temp_mask=False,
    ):
        nn.Module.__init__(self)
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
                dim_c=self.dim_C,
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
            len_h=self.dim_H,  # // self.patch_lw,
            len_w=self.dim_W,  # // self.patch_lw,
            len_t=self.dim_Lp
            + self.dim_Lf,  # notice how we embed both future and past tokens simultaneously
            theta=10000.0,
            device=device,
        )
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            block = None
            if i == self.n_layers - 1:
                block = MMDiTSplitAttentionBlock(
                    self.dim_hidden,
                    self.dim_hidden,
                    num_heads=self.n_head,
                    skip_context_ff=True,
                )
            else:
                block = MMDiTSplitAttentionBlock(
                    self.dim_hidden,
                    self.dim_hidden,
                    num_heads=self.n_head,
                )
            self.blocks.append(block)
        self.final_layer = FinalLayer(
            self.dim_hidden,
            patch_lw=self.patch_lw,
            patch_t=self.patch_t,
            out_channels=self.dim_C,
        )

        self.register_buffer(
            "empty_past_frames_emb",
            torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)),
        )
        # self.empty_past_frames_emb = nn.Parameter(torch.zeros((self.dim_C, self.dim_Lp, self.dim_H, self.dim_W)))

        self.register_buffer(
            "empty_past_actions_emb", torch.zeros((self.dim_Tp, self.dim_act))
        )
        # self.empty_past_actions_emb = nn.Parameter(torch.zeros((self.dim_Tp, self.dim_act)))

        self.register_buffer(
            "empty_future_actions_emb", torch.zeros((self.dim_Tf, self.dim_act))
        )
        # self.empty_future_actions_emb = nn.Parameter(torch.zeros((self.dim_Tf, self.dim_act)))

        self.cfg_prob = cfg_prob
        # self.conditioning_manager = conditioning_manager
        # self.conditioning = conditioning
        self.initialize_weights()
