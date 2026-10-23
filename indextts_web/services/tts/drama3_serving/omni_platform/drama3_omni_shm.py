"""SGLang-Omni CUDA platform whose stages exchange tensors through host SHM.

Runs inside the Drama 3 venv, loaded by SGLANG_OMNI_PLATFORM_SPEC, only when
the WebUI sleeps Omni on NVIDIA drivers 580-609. Those checkpoint CUDA
processes but not CUDA IPC (cuIpcGetMemHandle), which the stock platform uses
for every same-GPU stage edge. Import only packages from that venv here; this
directory is deliberately not a package of the WebUI.
"""

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.platforms.cuda import CUDAOmniPlatform


class ShmCudaPlatform(CUDAOmniPlatform):
    def get_intra_node_transport(self) -> TransportKind:
        # Also disables the direct same-GPU payload path, which checks this value.
        return TransportKind.SHM
