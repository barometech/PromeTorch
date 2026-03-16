"""
NM Card Mini Monitor — Real-time hardware utilization dashboard.
Always-on-top, minimal footprint, doesn't interfere with training.

Run: python nmcard_monitor.py
"""
import ctypes, os, sys, time, threading, collections

# ============================================================
# NM Card DDR reader (background thread)
# ============================================================
NM_PATH = r'C:\Program Files\Module\NM_Card\libload\bin'
DDR = 0x00340000
STATUS = 30
WATCHDOG = 31

class CardReader:
    def __init__(self):
        os.environ['PATH'] = NM_PATH + ';' + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(NM_PATH)
        self.nm = ctypes.CDLL(os.path.join(NM_PATH, 'nm_card_load.dll'))
        self.nm.PL_SetTimeout.argtypes = [ctypes.c_uint32]
        self.nm.PL_SetTimeout(5000)
        self.nm.PL_GetBoardCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.nm.PL_GetBoardCount.restype = ctypes.c_int
        self.nm.PL_GetBoardDesc.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.nm.PL_GetBoardDesc.restype = ctypes.c_int
        self.nm.PL_GetAccess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int * 2), ctypes.POINTER(ctypes.c_void_p)]
        self.nm.PL_GetAccess.restype = ctypes.c_int
        self.nm.PL_ReadMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        self.nm.PL_ReadMemBlock.restype = ctypes.c_int
        self.nm.PL_CloseAccess.argtypes = [ctypes.c_void_p]
        self.nm.PL_CloseAccess.restype = ctypes.c_int
        self.nm.PL_CloseBoardDesc.argtypes = [ctypes.c_void_p]
        self.nm.PL_CloseBoardDesc.restype = ctypes.c_int

        self.board = None
        self.access = None
        self.connected = False

    def connect(self):
        try:
            count = ctypes.c_uint()
            self.nm.PL_GetBoardCount(ctypes.byref(count))
            if count.value == 0:
                return False
            self.board = ctypes.c_void_p()
            self.nm.PL_GetBoardDesc(0, ctypes.byref(self.board))
            cn = (ctypes.c_int * 2)(0, 0)
            self.access = ctypes.c_void_p()
            self.nm.PL_GetAccess(self.board, ctypes.byref(cn), ctypes.byref(self.access))
            self.connected = True
            return True
        except:
            return False

    def read(self):
        """Read card state. Returns (cmd, status, watchdog) or None."""
        if not self.connected:
            return None
        try:
            buf = (ctypes.c_uint * 32)()
            self.nm.PL_ReadMemBlock(self.access, buf, DDR, 32)
            return {
                'cmd': buf[0],
                'args': [buf[i] for i in range(1, 7)],
                'status': buf[STATUS],
                'watchdog': buf[WATCHDOG],
            }
        except:
            return None

    def close(self):
        try:
            if self.access:
                self.nm.PL_CloseAccess(self.access)
            if self.board:
                self.nm.PL_CloseBoardDesc(self.board)
        except:
            pass

# ============================================================
# Tkinter GUI
# ============================================================
try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:
    print("tkinter not available")
    sys.exit(1)

OP_NAMES = {0:'IDLE', 1:'MATMUL', 2:'RMSNORM', 3:'SOFTMAX', 4:'SILU',
            5:'ROPE', 6:'ATTN', 10:'ADD', 11:'MUL', 13:'GATE', 255:'EXIT'}

class MonitorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NM Card Mini")
        self.root.geometry("340x520")
        self.root.configure(bg='#0a0a0a')
        self.root.attributes('-topmost', True)
        self.root.resizable(False, False)

        # Data
        self.history = collections.deque(maxlen=120)  # 2 min @ 1Hz
        self.prev_wd = 0
        self.prev_time = time.time()
        self.busy_samples = 0
        self.total_samples = 0
        self.total_ops = 0
        self.util_pct = 0
        self.wd_rate = 0
        self.current_op = 'IDLE'
        self.card_alive = False

        # Card reader
        self.reader = CardReader()

        self._build_ui()
        self._start_polling()

    def _build_ui(self):
        r = self.root
        mono = ('Consolas', 10)
        mono_big = ('Consolas', 24, 'bold')
        mono_sm = ('Consolas', 9)

        # Title bar
        title = tk.Frame(r, bg='#1a1a1a', height=36)
        title.pack(fill='x')
        title.pack_propagate(False)
        tk.Label(title, text="  NM Card Mini  K1879VM8YA", bg='#1a1a1a', fg='#ff6b35',
                 font=('Consolas', 11, 'bold')).pack(side='left', padx=5)
        self.status_dot = tk.Label(title, text="●", bg='#1a1a1a', fg='#333', font=('Arial', 14))
        self.status_dot.pack(side='right', padx=10)

        # Big utilization number
        util_frame = tk.Frame(r, bg='#0a0a0a', height=80)
        util_frame.pack(fill='x', pady=(10, 0))
        util_frame.pack_propagate(False)
        self.util_label = tk.Label(util_frame, text="---%", bg='#0a0a0a', fg='#4caf50',
                                   font=mono_big)
        self.util_label.pack()
        tk.Label(util_frame, text="NMC4 Utilization", bg='#0a0a0a', fg='#666',
                 font=mono_sm).pack()

        # Graph canvas
        self.canvas = tk.Canvas(r, width=320, height=100, bg='#111', highlightthickness=1,
                                highlightbackground='#333')
        self.canvas.pack(padx=10, pady=5)

        # Stats grid
        stats = tk.Frame(r, bg='#0a0a0a')
        stats.pack(fill='x', padx=10, pady=5)

        def stat_row(parent, label, row):
            tk.Label(parent, text=label, bg='#0a0a0a', fg='#888', font=mono_sm,
                     anchor='w').grid(row=row, column=0, sticky='w', padx=(0, 10))
            val = tk.Label(parent, text="---", bg='#0a0a0a', fg='#ddd', font=mono, anchor='e')
            val.grid(row=row, column=1, sticky='e')
            return val

        self.lbl_op = stat_row(stats, "Current Op", 0)
        self.lbl_wdrate = stat_row(stats, "Watchdog/s", 1)
        self.lbl_ops = stat_row(stats, "Total Ops", 2)
        self.lbl_uptime = stat_row(stats, "Uptime", 3)
        self.lbl_cores = stat_row(stats, "Cores", 4)
        self.lbl_ddr = stat_row(stats, "DDR Used", 5)
        self.lbl_peak = stat_row(stats, "Peak GFLOPS", 6)
        self.lbl_temp = stat_row(stats, "Status", 7)
        stats.columnconfigure(1, weight=1)

        # Separator
        tk.Frame(r, bg='#333', height=1).pack(fill='x', padx=10, pady=5)

        # Hardware info
        hw = tk.Frame(r, bg='#0a0a0a')
        hw.pack(fill='x', padx=10)
        info = [
            "16x NMC4 @ 1GHz  |  5GB DDR3L",
            "PCIe 2.0 x4  |  25W TDP",
            "512 GFLOPS FP32 peak",
        ]
        for line in info:
            tk.Label(hw, text=line, bg='#0a0a0a', fg='#444', font=('Consolas', 8)).pack(anchor='w')

        # Bottom bar
        bottom = tk.Frame(r, bg='#1a1a1a', height=28)
        bottom.pack(fill='x', side='bottom')
        bottom.pack_propagate(False)
        tk.Label(bottom, text="PromeTorch", bg='#1a1a1a', fg='#555',
                 font=('Consolas', 8)).pack(side='left', padx=10)
        self.topmost_btn = tk.Button(bottom, text="📌", bg='#1a1a1a', fg='#888',
                                      font=('Arial', 10), bd=0, command=self._toggle_topmost)
        self.topmost_btn.pack(side='right', padx=5)

        self.start_time = time.time()

    def _toggle_topmost(self):
        current = self.root.attributes('-topmost')
        self.root.attributes('-topmost', not current)
        self.topmost_btn.configure(fg='#ff6b35' if not current else '#444')

    def _start_polling(self):
        # Connect in background
        def connect():
            if self.reader.connect():
                self.card_alive = True
            self.root.after(100, self._poll)
        threading.Thread(target=connect, daemon=True).start()

    def _poll(self):
        data = self.reader.read() if self.card_alive else None

        now = time.time()
        dt = now - self.prev_time

        if data:
            wd = data['watchdog']
            cmd = data['cmd']
            status = data['status']

            self.current_op = OP_NAMES.get(cmd, f'OP_{cmd}')

            # Count busy (status=0 and cmd!=0 means computing)
            if status == 0 and cmd != 0:
                self.busy_samples += 1
            self.total_samples += 1

            if cmd != 0 and status == 1:
                self.total_ops += 1

            if dt >= 1.0:
                self.wd_rate = (wd - self.prev_wd) / dt if self.prev_wd > 0 else 0

                # Utilization: when computing, watchdog increments slower
                # Calibrated idle rate ~2M/s for this dispatcher
                idle_rate = 2_000_000
                if self.wd_rate > 0:
                    self.util_pct = max(0, min(100, (1.0 - self.wd_rate / idle_rate) * 100))
                else:
                    self.util_pct = 0

                self.history.append(self.util_pct)
                self.prev_wd = wd
                self.prev_time = now
                self.busy_samples = 0
                self.total_samples = 0

            self.status_dot.configure(fg='#4caf50' if wd > 0 else '#f44336')
        else:
            if dt >= 1.0:
                self.history.append(0)
                self.util_pct = 0
                self.prev_time = now
            self.status_dot.configure(fg='#f44336')
            self.current_op = 'OFFLINE'

        self._update_ui()
        self.root.after(100, self._poll)  # 10 Hz polling

    def _update_ui(self):
        # Big number
        if self.card_alive:
            pct = self.util_pct
            color = '#4caf50' if pct < 50 else ('#ff9800' if pct < 80 else '#f44336')
            self.util_label.configure(text=f"{pct:.0f}%", fg=color)
        else:
            self.util_label.configure(text="---", fg='#666')

        # Stats
        self.lbl_op.configure(text=self.current_op,
                               fg='#4fc3f7' if self.current_op != 'IDLE' else '#666')
        self.lbl_wdrate.configure(text=f"{self.wd_rate:,.0f}")
        self.lbl_ops.configure(text=f"{self.total_ops:,}")
        uptime = time.time() - self.start_time
        self.lbl_uptime.configure(text=f"{int(uptime//60)}m {int(uptime%60)}s")
        self.lbl_cores.configure(text="1 / 16 active")
        self.lbl_peak.configure(text=f"{self.util_pct/100*32:.1f} / 32")
        self.lbl_temp.configure(text="OK" if self.card_alive else "OFFLINE",
                                fg='#4caf50' if self.card_alive else '#f44336')

        # Graph
        self._draw_graph()

    def _draw_graph(self):
        c = self.canvas
        c.delete('all')
        w, h = 320, 100

        # Grid lines
        for y_pct in [25, 50, 75]:
            y = h - (y_pct / 100 * h)
            c.create_line(0, y, w, y, fill='#222', dash=(2, 4))
            c.create_text(w - 2, y - 8, text=f"{y_pct}%", fill='#444',
                          font=('Consolas', 7), anchor='e')

        if len(self.history) < 2:
            return

        # Draw filled area
        points = []
        n = len(self.history)
        for i, val in enumerate(self.history):
            x = (i / max(n - 1, 1)) * w
            y = h - (val / 100 * h)
            points.append((x, y))

        # Fill
        fill_points = [(0, h)] + points + [(w, h)]
        flat = [coord for p in fill_points for coord in p]
        c.create_polygon(flat, fill='#1a3a1a', outline='')

        # Line
        if len(points) >= 2:
            line_flat = [coord for p in points for coord in p]
            c.create_line(line_flat, fill='#4caf50', width=2, smooth=True)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.reader.close()

if __name__ == '__main__':
    app = MonitorApp()
    app.run()
