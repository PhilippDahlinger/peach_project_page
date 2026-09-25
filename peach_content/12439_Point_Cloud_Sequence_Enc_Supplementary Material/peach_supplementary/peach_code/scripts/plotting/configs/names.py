metric_names = {
    "mat_prop_loss": "Material Properties MSE",
    "ml_loss": "Full Rollout MSE"
}

method_names = {
    "dummy_encoder_mango_decoder": "No Context Encoding",
    "pstnet_encoder_mango_decoder": "PSTNet Encoder",
    "gnn_encoder_mango_decoder": "GNN Encoder",
    "pointpatch_encoder_mango_decoder": "PEACH (ours)",
    "oracle_encoder_mango_decoder": "Oracle",
    "mango_encoder_mango_decoder": "MaNGO",
    "mgn": "MGN",
    "mgn_oracle": "MGN Oracle",
    "peach_only_ml": "PEACH (No aux. loss)",
    "peach_only_ml_and_matprop": "PEACH (only aux. MatProp Loss)",
    "peach_only_ml_and_sdf": "PEACH (only aux. SDF Loss)",
    "test_opt": "AdaptiGraph Test Optimization"

}

env_names = {
    "db": "Deformable Block",
    "sd": "Sheet Deformation",
    "trampoline": "Trampoline",
    "bbv": "Bending Beam",
}
