import mujoco

model = mujoco.MjModel.from_xml_path("assets/panda_tray/panda_tray_cylinder.xml")

print("njnt", model.njnt, "nq", model.nq, "nv", model.nv)
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    qpos_adr = int(model.jnt_qposadr[j])
    dof_adr  = int(model.jnt_dofadr[j])
    # estimate qpos size (next qpos adr - this) or remaining
    next_addrs = [int(model.jnt_qposadr[k]) for k in range(model.njnt) if int(model.jnt_qposadr[k])>qpos_adr]
    qpos_size = (min(next_addrs)-qpos_adr) if next_addrs else (model.nq - qpos_adr)
    print(f"joint {j:2d}: name='{name:20s}' qpos_adr={qpos_adr:2d} qpos_size={qpos_size:1d} dof_adr={dof_adr:2d}")