import h5py
import torch
from tqdm import tqdm

# dataset_type = "sd_v1"
dataset_type = "trampoline_v4"

if dataset_type == "sd_v1":
    hdf5_path = "../datasets/pc_mango/sd_v1.hdf5"
elif dataset_type == "db_v4":
    hdf5_path = "../datasets/pc_mangodb_v4.hdf5"
elif dataset_type == "trampoline_v4":
    hdf5_path = "../datasets/pc_mango/trampoline_v4.hdf5"
elif dataset_type == "bbv_v3":
    hdf5_path = "../datasets/pc_mango/bbv_v3.hdf5"

with h5py.File(hdf5_path, "r") as f:
    print("stop")
    all_params = []
    yms = []
    tts = []
    vgs = []
    vtaus = []
    for key in tqdm(f.keys()):
        if key.startswith("task_"):
            task_dict = f[key]
            params = task_dict["params"]
            if dataset_type == "sd_v1":
                ym = torch.tensor(params["youngs_modulus"][()])
                # print("original young's modulus:", ym.item())
                ym = ym / 500
                # take the log of the youngs modulus to make it easier to learn
                ym = torch.log(ym + 1e-8)
                all_params.append(ym)
            elif dataset_type == "db_v4":
                youngs_modulus = torch.tensor(params["youngs_modulus"][()])
                poisson_ratio = torch.tensor(params["poisson_ratio"][()])
                youngs_modulus = youngs_modulus / 10000
                all_params.append(torch.tensor([youngs_modulus, poisson_ratio]))
            elif dataset_type == "trampoline_v4":
                normalized_ball_diameter = torch.tensor(
                    [params["ball_diameter"][()]]) / 60.0  # max diameter is 60, so this should be in [0, 1]
                ball_mass = torch.tensor([params["ball_mass"][()]])
                youngs_modulus = torch.tensor([params["youngs_modulus"][()]])
                tape_thickness = torch.tensor([params["tape_thickness"][()]])
                # ym and thickness are very dependent on each other, since they both control the stiffness of the trampoline, so we normalize them together by their product, which is proportional to the bending stiffness of the trampoline
                Et = youngs_modulus * tape_thickness
                log_Et = torch.log(Et)
                log_m_over_Et = torch.log(ball_mass / Et)
                viscous_g = torch.tensor([params["viscous"]["g"][()]])
                # viscous_k = torch.tensor([task_dict["params"]["viscous"]["k"]])
                log_g = torch.log(viscous_g)
                viscous_tau = torch.tensor([params["viscous"]["tau"][()]])
                log_tau = torch.log(viscous_tau)
                material_properties = torch.cat([normalized_ball_diameter, log_Et, log_m_over_Et, log_g, log_tau],
                                                dim=-1)
                all_params.append(material_properties)
                yms.append(youngs_modulus)
                tts.append(tape_thickness)
                vgs.append(viscous_g)
                vtaus.append(viscous_tau)
            elif dataset_type == "bbv_v3":
                poisson = torch.log(torch.tensor(params["poisson"][()]))
                viscosity = torch.log(torch.tensor(params["viscosity"][()]))
                youngs_modulus = torch.log(torch.tensor(params["youngs_modulus"][()]))
                material_properties = torch.stack([poisson, viscosity, youngs_modulus])[None, :]
                all_params.append(material_properties)
            else:
                raise NotImplementedError
    all_params = torch.stack(all_params, dim=0)
    yms = torch.stack(yms, dim=0)
    tts = torch.stack(tts, dim=0)
    vgs = torch.stack(vgs, dim=0)
    vtaus = torch.stack(vtaus, dim=0)
    print(f"Viscous g statistics: mean={vgs.mean().item()}, std={vgs.std().item()}, min={vgs.min().item()}, max={vgs.max().item()}")
    print(f"Viscous tau statistics: mean={vtaus.mean().item()}, std={vtaus.std().item()}, min={vtaus.min().item()}, max={vtaus.max().item()}")
    print(f"Tape thickness statistics: mean={tts.mean().item()}, std={tts.std().item()}, min={tts.min().item()}, max={tts.max().item()}")
    print(f"Young's modulus statistics: mean={yms.mean().item()}, std={yms.std().item()}, min={yms.min().item()}, max={yms.max().item()}")
    print(f"Mat Prop statistics: mean={all_params.mean(dim=0)}, std={all_params.std(dim=0)}, min={all_params.min(dim=0)}, max={all_params.max(dim=0)}")