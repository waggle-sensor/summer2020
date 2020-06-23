## RT kernel test

Generic kernel:
```
nvidia@nvidia-desktop:~$ uname -a
Linux nvidia-desktop 4.9.140-tegra #1 SMP PREEMPT Wed Apr 8 18:15:20 PDT 2020 aarch64 aarch64 aarch64 GNU/Linux
nvidia@nvidia-desktop:~$ uname -r
4.9.140-tegra
```
top output:
```
    1 root      20   0  161500   8880   6000 S   0.0  0.0   0:02.10 systemd     
    2 root      20   0       0      0      0 S   0.0  0.0   0:00.01 kthreadd    
    3 root      20   0       0      0      0 S   0.0  0.0   0:00.01 ksoftirqd/0 
    4 root      20   0       0      0      0 S   0.0  0.0   0:00.00 kworker/0:0 
    5 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/0:+ 
    6 root      20   0       0      0      0 S   0.0  0.0   0:00.05 kworker/u1+ 
    7 root      20   0       0      0      0 S   0.0  0.0   0:00.02 rcu_preempt 
    8 root      20   0       0      0      0 S   0.0  0.0   0:00.01 rcu_sched   
    9 root      20   0       0      0      0 S   0.0  0.0   0:00.00 rcu_bh      
   10 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 migration/0 
   11 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 lru-add-dr+ 
   12 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 watchdog/0  
   13 root      20   0       0      0      0 S   0.0  0.0   0:00.00 cpuhp/0     
   14 root      20   0       0      0      0 S   0.0  0.0   0:00.00 cpuhp/1     
   15 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 watchdog/1  
   16 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 migration/1 
   17 root      20   0       0      0      0 S   0.0  0.0   0:00.00 ksoftirqd/1 
   18 root      20   0       0      0      0 S   0.0  0.0   0:00.02 kworker/1:0 
   19 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/1:+ 
   20 root      20   0       0      0      0 S   0.0  0.0   0:00.00 cpuhp/2     
   21 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 watchdog/2  
   22 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 migration/2 
   23 root      20   0       0      0      0 S   0.0  0.0   0:00.00 ksoftirqd/2
```
RT kernel:
```
nvidia@nvidia-xavier-rt:~$ uname -a
Linux nvidia-xavier-rt 4.9.140-rt93-tegra #1 SMP PREEMPT RT Sun Jun 7 23:08:52 EDT 2020 aarch64 aarch64 aarch64 GNU/Linux
nvidia@nvidia-xavier-rt:~$ uname -r
4.9.140-rt93-tegra
```
top:
```
  1 root      20   0  161656   8776   5820 S   0.0  0.0   0:20.63 systemd     
    2 root      20   0       0      0      0 S   0.0  0.0   0:00.04 kthreadd    
    3 root      20   0       0      0      0 S   0.0  0.0   0:05.30 ksoftirqd/0 
    4 root      -2   0       0      0      0 S   0.0  0.0   0:02.02 ktimersoft+ 
    6 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/0:+ 
    8 root      -2   0       0      0      0 S   0.0  0.0   0:03.92 rcu_preempt 
    9 root      -2   0       0      0      0 S   0.0  0.0   0:01.64 rcu_sched   
   10 root      -2   0       0      0      0 S   0.0  0.0   0:00.00 rcub/0      
   11 root      -2   0       0      0      0 S   0.0  0.0   0:01.85 rcuc/0      
   12 root      20   0       0      0      0 S   0.0  0.0   0:00.00 kswork      
   13 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 posixcputm+ 
   14 root      rt   0       0      0      0 S   0.0  0.0   0:00.08 migration/0 
   15 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 watchdog/0  
   16 root      20   0       0      0      0 S   0.0  0.0   0:00.00 cpuhp/0     
   17 root      20   0       0      0      0 S   0.0  0.0   0:00.00 cpuhp/1     
   18 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 watchdog/1  
   19 root      rt   0       0      0      0 S   0.0  0.0   0:00.08 migration/1 
   20 root      -2   0       0      0      0 S   0.0  0.0   0:01.81 rcuc/1      
   21 root      -2   0       0      0      0 S   0.0  0.0   0:01.97 ktimersoft+ 
   22 root      20   0       0      0      0 S   0.0  0.0   0:03.90 ksoftirqd/1 
   24 root       0 -20       0      0      0 S   0.0  0.0   0:00.00 kworker/1:+ 
   25 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 posixcputm+ 
   26 root      20   0       0      0      0 S   0.0  0.0   0:00.00 cpuhp/2     
   27 root      rt   0       0      0      0 S   0.0  0.0   0:00.00 watchdog/2  
   28 root      rt   0       0      0      0 S   0.0  0.0   0:00.08 migration/2 
   29 root      -2   0       0      0      0 S   0.0  0.0   0:01.71 rcuc/2      
   30 root      -2   0       0      0      0 S   0.0  0.0   0:01.98 ktimersoft+ 
```

### Test setup
```
sudo apt-get install build-essential libnuma-dev
git clone git://git.kernel.org/pub/scm/utils/rt-tests/rt-tests.git
cd rt-tests
git checkout stable/v1.0
make all
sudo make install
```
The output of the generic kernel test:
```
nvidia@nvidia-xavier-rt:~/projects/rt-tests$ sudo ./cyclictest --mlockall --smp --priority=80 --interval=200 --distance=0
# /dev/cpu_dma_latency set to 0us
policy: fifo: loadavg: 1.04 0.74 0.61 3/1212 8147           

T: 0 ( 8137) P:80 I:200 C:  73055 Min:      7 Act:   13 Avg:   15 Max:     151
T: 1 ( 8138) P:80 I:200 C:  73040 Min:      7 Act:   18 Avg:   15 Max:     125
T: 2 ( 8139) P:80 I:200 C:  73004 Min:      7 Act:   39 Avg:   14 Max:    1076
T: 3 ( 8140) P:80 I:200 C:  72989 Min:      6 Act:   48 Avg:   14 Max:     141
```
Config RT kernel to work:
```
sudo /usr/sbin/nvpmodel -m 0
sudo /usr/bin/jetson_clocks
```
The output of the RT kernel test:
```
nvidia@nvidia-xavier-rt:~/projects/rt-tests$ uname -a
Linux nvidia-xavier-rt 4.9.140-rt93-tegra #1 SMP PREEMPT RT Wed Jun 3 16:58:57 EDT 2020 aarch64 aarch64 aarch64 GNU/Linux
nvidia@nvidia-xavier-rt:~/projects/rt-tests$ sudo ./cyclictest --mlockall --smp --priority=80 --interval=200 --distance=0
# /dev/cpu_dma_latency set to 0us
policy: fifo: loadavg: 0.34 0.61 0.60 1/1238 8573           

T: 0 ( 8565) P:80 I:200 C:  18697 Min:      4 Act:    7 Avg:    8 Max:     116
T: 1 ( 8566) P:80 I:200 C:  18669 Min:      4 Act:    9 Avg:    7 Max:      49
T: 2 ( 8567) P:80 I:200 C:  18673 Min:      3 Act:    7 Avg:    7 Max:     109
T: 3 ( 8568) P:80 I:200 C:  18660 Min:      4 Act:    6 Avg:    7 Max:      45
T: 4 ( 8569) P:80 I:200 C:  18648 Min:      4 Act:    5 Avg:    8 Max:      83
T: 5 ( 8570) P:80 I:200 C:  18635 Min:      4 Act:    7 Avg:    7 Max:      46
T: 6 ( 8571) P:80 I:200 C:  18623 Min:      4 Act:    6 Avg:    7 Max:      43
T: 7 ( 8572) P:80 I:200 C:  18611 Min:      4 Act:    6 Avg:    7 Max:      50
```

### Reference
 * https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/rt-tests.
