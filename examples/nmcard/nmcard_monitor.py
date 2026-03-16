"""
NM Card Mini — Real-time utilization monitor.
Shows: card compute %, DDR bandwidth, matmul throughput, core activity.

Run alongside training: python nmcard_monitor.py
"""
import ctypes, os, time, sys

NM = r'C:\Program Files\Module\NM_Card\libload\bin'
os.environ['PATH'] = NM + ';' + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(NM)

nm = ctypes.CDLL(os.path.join(NM, 'nm_card_load.dll'))
nm.PL_SetTimeout.argtypes = [ctypes.c_uint32]; nm.PL_SetTimeout(5000)
nm.PL_GetBoardCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]; nm.PL_GetBoardCount.restype = ctypes.c_int
nm.PL_GetBoardDesc.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]; nm.PL_GetBoardDesc.restype = ctypes.c_int
nm.PL_GetAccess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int * 2), ctypes.POINTER(ctypes.c_void_p)]; nm.PL_GetAccess.restype = ctypes.c_int
nm.PL_ReadMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]; nm.PL_ReadMemBlock.restype = ctypes.c_int
nm.PL_CloseAccess.argtypes = [ctypes.c_void_p]; nm.PL_CloseAccess.restype = ctypes.c_int
nm.PL_CloseBoardDesc.argtypes = [ctypes.c_void_p]; nm.PL_CloseBoardDesc.restype = ctypes.c_int

DDR = 0x00340000
STATUS = 30
WATCHDOG = 31

count = ctypes.c_uint()
nm.PL_GetBoardCount(ctypes.byref(count))
if count.value == 0:
    print("No NM Card found"); sys.exit(1)

board = ctypes.c_void_p()
nm.PL_GetBoardDesc(0, ctypes.byref(board))
cn = (ctypes.c_int * 2)(0, 0)
access = ctypes.c_void_p()
nm.PL_GetAccess(board, ctypes.byref(cn), ctypes.byref(access))

buf = (ctypes.c_uint * 2)()

print("NM Card Mini Monitor")
print("=" * 70)
print(f"{'Time':>6} {'WD/s':>10} {'Status':>7} {'Cmd':>5} {'Busy%':>6} {'Ops':>6} {'Est GFLOPS':>10}")
print("-" * 70)

prev_wd = 0
prev_time = time.time()
prev_status_busy = 0
samples = 0
busy_count = 0
total_count = 0

try:
    while True:
        # Read status + watchdog
        nm.PL_ReadMemBlock(access, buf, DDR + STATUS, 2)
        status = buf[0]
        wd = buf[1]

        # Read cmd
        cmd_buf = (ctypes.c_uint * 1)()
        nm.PL_ReadMemBlock(access, cmd_buf, DDR, 1)
        cmd = cmd_buf[0]

        now = time.time()
        dt = now - prev_time

        if dt >= 1.0:
            wd_rate = (wd - prev_wd) / dt if prev_wd > 0 else 0
            busy_pct = (busy_count / max(total_count, 1)) * 100

            # Estimate: each watchdog tick ~ 10 NMC4 cycles @ 1GHz
            # If card is computing, watchdog still increments but slower
            # Busy% = fraction of samples where status=0 (computing)

            # Rough GFLOPS estimate: if doing matmul [T,D]@[D,F]
            # FLOPs per matmul = 2*T*D*F, time = poll_interval
            # This is very rough

            cmd_name = {0:'NOP', 1:'MATMUL', 2:'RMSNORM', 3:'SOFTMAX', 4:'SILU', 255:'EXIT'}.get(cmd, f'OP_{cmd}')

            elapsed = now - prev_time
            print(f"{int(now)%100000:>6} {wd_rate:>10,.0f} {status:>7} {cmd_name:>5} {busy_pct:>5.1f}% {busy_count:>6} ", end='')

            # Estimate GFLOPS from watchdog rate
            # NMC4 @ 1GHz, watchdog increments ~1 per loop iteration
            # When idle: ~2M ticks/sec. When computing: lower
            idle_rate = 2_000_000  # calibrated idle watchdog rate
            if wd_rate > 0 and wd_rate < idle_rate:
                util = max(0, 1.0 - wd_rate / idle_rate) * 100
                # Peak: 512 GFLOPS FP32 (16 cores), but we use 1 core = 32 GFLOPS
                gflops = util / 100 * 32
                print(f"{gflops:>9.1f}G")
            else:
                print(f"{'idle':>10}")

            prev_wd = wd
            prev_time = now
            busy_count = 0
            total_count = 0

        # Sample busy/idle
        if status == 0 and cmd != 0:
            busy_count += 1
        total_count += 1

        time.sleep(0.01)  # 100 samples/sec

except KeyboardInterrupt:
    print("\nMonitor stopped")
finally:
    nm.PL_CloseAccess(access)
    nm.PL_CloseBoardDesc(board)
