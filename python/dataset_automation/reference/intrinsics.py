from dataset_automation.camera.intrinsics import Intrinsics

# GoPro Hero 3 Silver, from the Mermaid dataset description.
MERMAID_INTRINSICS = Intrinsics(
    width_px=3840,
    height_px=2880,
    focal_px=2334.29,
    cx_offset_px=-12.752,
    cy_offset_px=-16.6962,
    k1=-0.222446,
    k2=0.310621,
    k3=-0.0835057,
    p1=-0.000995472,
    p2=-7.8498e-05,
)
