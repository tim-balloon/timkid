import os

xilinx_ip = os.getenv("XILINX_IP", "192.168.1.99")
xilinx_sshport = 22
xilinx_username = os.getenv("XILINX_USERNAME", "xilinx")
xilinx_password = os.getenv("XILINX_PASSWORD", "xilinx")
xilinx_key_path = os.getenv("XILINX_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519"))
xilinx_git_path = os.getenv("XILINX_GIT_PATH", "/home/xilinx/tim_readout/")
