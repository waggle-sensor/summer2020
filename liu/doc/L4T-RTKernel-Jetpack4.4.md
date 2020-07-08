## Buid RT Kernel on Nvidia AGX Xavier / TX2 / Nano

A Intel (x64) host machine is needed to cross compile the Linux4Tegra system. 

### Step 1: Install Jetpack SDK on Host Machine
Download link: https://developer.nvidia.com/embedded/jetpack

### Step 2: Make your own Kernel
Create a directory in your home directory:
```
mkdir nvidia-rt
cd nvidiart
```
Clone the jetson-agx-build repo into it.
```
git clone https://github.com/jtagxhub/jetpack-agx-build.git build
```
Add Xavier-4.4-DP into config:
```
cd config
vim Xavier-4.4
```
```
# JetPack 4.4-DP

source $CONFIG_DIR/common/TX2_Xavier-4.4

# OUT
OUT_ROOT=$TOP/Xavier
```
Under common folder, add TX2_Xavier-4.4-DP, which is for the builng of L4T supporting jetPack 4.4-DP:
```
# JetPack 4.4-DP

## Download Links
KERNEL_TOOLCHAIN_LINK=https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit
BSP_TOOLCHAIN_LINK=$KERNEL_TOOLCHAIN_LINK
SOURCES_LINK=https://developer.nvidia.com/embedded/L4T/r32_Release_v4.2/Sources/T186/public_sources.tbz2
BSP_LINK=https://developer.nvidia.com/embedded/L4T/r32_Release_v4.2/t186ref_release_aarch64/Tegra186_Linux_R32.4.2_aarch64.tbz2
ROOTFS_LINK=https://developer.nvidia.com/embedded/L4T/r32_Release_v4.2/t186ref_release_aarch64/Tegra_Linux_Sample-Root-Filesystem_R32.4.2_aarch64.tbz2

# Toolchain
KERNEL_TOOLCHAIN=$KERNEL_TOOLCHAIN_ROOT/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-
BSP_TOOLCHAIN=$KERNEL_TOOLCHAIN
BSP_TOOLCHAIN_ROOT=$KERNEL_TOOLCHAIN_ROOT

# DOWNLOAD
KERNEL_TOOLCHAIN_PACKAGE=$DOANLOAD_ROOT/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
BSP_TOOLCHAIN_PACKAGE=$KERNEL_TOOLCHAIN_PACKAGE
BSP_PACKAGE=$DOANLOAD_ROOT/Tegra186_Linux_R32.4.2_aarch64.tbz2
ROOTFS_PACKAGE=$DOANLOAD_ROOT/Tegra_Linux_Sample-Root-Filesystem_R32.4.2_aarch64.tbz2

# Kernel
KERNEL_VERSION=4.9
TARGET_KERNEL_CONFIG=tegra_defconfig

# Override
SOURCE_UNPACK=$DOANLOAD_ROOT/Linux_for_Tegra/source/public
KERNEL_PACKAGE=$SOURCE_UNPACK/kernel_src.tbz2
CBOOT_PACKAGE=$SOURCE_UNPACK/cboot_src_t19x.tbz2
```
Then set up the environment path:
```
source build/envsetup.sh
```
Several attributes including target board, your release, your user and password for your board, and your device IP are needed.
 * Target board: Xavier or TX2
 * Release: Jetpack 4.4
 * user: nvidia
 * password: nvidia
 * device IP: 192.168.55.1 (this is the IP address the Jetson assigns to itself in the Ethernet-over-USB mini-network)

![image](https://github.com/waggle-sensor/summer2020/blob/master/liu/image/Screenshot%20from%202020-06-08%2011-01-25.png)

Setup Xavier/Linux_for_Tegra, toolchain, and kernel source:
```
l4tout_setup
bspsetup
```
Get into kernel source and apply RT kernel patch:
```
cd sources/kernel/kernel/kernel-4.9
./scripts/rt-patch.sh apply-patches
```
Your kernel is now realtime! Do some quick config and build it:
```
kdefconfig
kmenuconfig     # You don't HAVE to change anything
ksavedefconfig
kbuild -a
```
### Step 3: Flash your Jetson

Make Jetson boards gets into recovery mode. Enter recovery mode by holding down the Recovery switch (location varies by board) and then holding down the Power button. Release both after a couple seconds.

Back on your host, run:
```
flash
```
### Step 4: Install Ubuntu
The guaranteed way you know your flash works is the that the Jetson will power up, and you’ll see an install screen on its monitor. Connect a keyboard and do all the things.

When it prompts you for username and password, be sure to use the same ones you specified in step 2 after running envsetup.sh.

When installation is complete, and your Jetson restarted, login and run uname -a to see what kernel you have. If it has PREEMPT RT in it, congratulations! Your Jetson is running a fully preemptive realtime kernel!

### Step 5: Install SDK components on Jetson

On host machine, start up SDK manager and start to flash the Xavier board. Remember to leave Jetson OS unchecked, since we already flashed it.

### Reference:
 - [Real-Time Linux for Jetson TX2](https://github.com/kozyilmaz/nvidia-jetson-rt/blob/master/docs/README.03-realtime.md#copy-binaries-to-l4t-for-deployment)
 - [Kernel Customization](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fkernel_custom.html)
 - https://github.com/jtagxhub/jetpack-agx-build
 - https://forums.developer.nvidia.com/t/preempt-rt-patch-on-jetson-kernel/65766/9
 - https://orenbell.com/?p=436
