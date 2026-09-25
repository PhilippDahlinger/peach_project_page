


def get_tokenizer(tokenizer_config, example_batch):
    if tokenizer_config.name == "pointnet":
        from pc_mango.network.pc_encoder.tokenizer.pointnet_tokenizer import PointnetTokenizer
        from torch_geometric.nn import MLP
        """Build the PPRL tokenizer from config."""

        mlp_1_dims = list(tokenizer_config.mlp_1_dims)
        if example_batch["pc_color"].shape[-1] > 1:
            mlp_1_dims[0] += example_batch["pc_color"].shape[-1] - 1  # add color feature dim to input of mlp_1
        mlp_1 = MLP(list(mlp_1_dims))
        mlp_2 = MLP(list(tokenizer_config.mlp_2_dims))
        return PointnetTokenizer(
            mlp_1=mlp_1,
            mlp_2=mlp_2,
            group_size=tokenizer_config.group_size,
            sampling_ratio=tokenizer_config.sampling_ratio,
            point_dim=3,  # Always 3D points
            embed_dim=tokenizer_config.embed_dim,
            random_start=tokenizer_config.random_start,
            padding_value=tokenizer_config.padding_value,
        )
    elif tokenizer_config.name == "instant_policy":
        from pc_mango.network.pc_encoder.tokenizer.instant_policy_tokenizer import InstantPolicyTokenizer
        """Build the Instant Policy tokenizer from config."""
        return InstantPolicyTokenizer(config=tokenizer_config)
    else:
        raise ValueError(f"Unknown tokenizer name: {tokenizer_config.name}")