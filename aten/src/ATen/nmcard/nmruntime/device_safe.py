"""
device_safe.py - Безопасная работа с NM Card Mini

ГЛАВНОЕ: dispatcher_safe.cpp имеет OP_EXIT команду.
Если карта зависла - записываем OP_EXIT и программа завершается сама!
"""

import ctypes
import os
import time

# Коды операций (должны совпадать с dispatcher_safe.cpp)
OP_NOP      = 0
OP_MATMUL   = 1
OP_RMSNORM  = 2
OP_SILU     = 3
OP_SOFTMAX  = 4
OP_ROPE     = 5
OP_ADD      = 6
OP_MUL      = 7
OP_ATTENTION = 8
OP_PING     = 100
OP_EXIT     = 255  # ВЫХОД!

# Статусы
STATUS_BUSY  = 0
STATUS_DONE  = 1
STATUS_ERROR = 2

# Memory layout
CMD_ADDR    = 0
ARG0_ADDR   = 1
ARG1_ADDR   = 2
ARG2_ADDR   = 3
ARG3_ADDR   = 4
ARG4_ADDR   = 5
ARG5_ADDR   = 6
STATUS_ADDR = 7
RESULT_ADDR = 8
DATA_ADDR   = 16

# DDR base
DDR_BASE = 0x00340000


class DeviceSafe:
    """Безопасный доступ к NM Card Mini с поддержкой OP_EXIT"""

    def __init__(self, board_index=0):
        # Добавить пути к DLL
        for path in [r'C:\Program Files\Module\NM_Card\libload\bin']:
            os.environ['PATH'] = path + ';' + os.environ.get('PATH', '')
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(path)

        # Загрузить DLL
        self.nm = ctypes.CDLL(r'C:\Program Files\Module\NM_Card\libload\bin\nm_card_load.dll')
        self._setup_functions()

        # Открыть карту
        self.board = ctypes.c_void_p()
        r = self.nm.PL_GetBoardDesc(board_index, ctypes.byref(self.board))
        if r != 0:
            raise RuntimeError(f"PL_GetBoardDesc failed: {r}")

        self.access = None
        self.dispatcher_loaded = False

    def _setup_functions(self):
        """Настройка типов функций"""
        nm = self.nm

        nm.PL_GetBoardDesc.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nm.PL_GetBoardDesc.restype = ctypes.c_int

        nm.PL_CloseBoardDesc.argtypes = [ctypes.c_void_p]
        nm.PL_CloseBoardDesc.restype = ctypes.c_int

        nm.PL_ResetBoard.argtypes = [ctypes.c_void_p]
        nm.PL_ResetBoard.restype = ctypes.c_int

        nm.PL_LoadInitCode.argtypes = [ctypes.c_void_p]
        nm.PL_LoadInitCode.restype = ctypes.c_int

        nm.PL_GetAccess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int * 2), ctypes.POINTER(ctypes.c_void_p)]
        nm.PL_GetAccess.restype = ctypes.c_int

        nm.PL_CloseAccess.argtypes = [ctypes.c_void_p]
        nm.PL_CloseAccess.restype = ctypes.c_int

        nm.PL_LoadProgramFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nm.PL_LoadProgramFile.restype = ctypes.c_int

        nm.PL_ReadMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        nm.PL_ReadMemBlock.restype = ctypes.c_int

        nm.PL_WriteMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        nm.PL_WriteMemBlock.restype = ctypes.c_int

        nm.PL_GetStatus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        nm.PL_GetStatus.restype = ctypes.c_int

        nm.PL_SetTimeout.argtypes = [ctypes.c_uint]
        nm.PL_SetTimeout.restype = ctypes.c_int

    def reset_and_init(self):
        """Сброс карты и загрузка init code"""
        r = self.nm.PL_ResetBoard(self.board)
        if r != 0:
            raise RuntimeError(f"PL_ResetBoard failed: {r}")

        r = self.nm.PL_LoadInitCode(self.board)
        if r != 0:
            raise RuntimeError(f"PL_LoadInitCode failed: {r}")

    def get_access(self, nm_id=0, cluster_id=0):
        """Получить доступ к ядру (работает БЕЗ reset!)"""
        if self.access:
            self.nm.PL_CloseAccess(self.access)

        core_no = (ctypes.c_int * 2)(nm_id, cluster_id)
        self.access = ctypes.c_void_p()
        r = self.nm.PL_GetAccess(self.board, ctypes.byref(core_no), ctypes.byref(self.access))
        if r != 0:
            raise RuntimeError(f"PL_GetAccess failed: {r}")

    def load_dispatcher(self, dispatcher_path=None):
        """Загрузить dispatcher на карту"""
        if dispatcher_path is None:
            dispatcher_path = r"C:\Users\paper\Desktop\nm_card_mini_as_TRAINER\nmc_programs\dispatcher_safe.abs"

        self.reset_and_init()
        self.get_access()

        r = self.nm.PL_LoadProgramFile(self.access, dispatcher_path.encode())
        if r != 0:
            raise RuntimeError(f"PL_LoadProgramFile failed: {r}")

        self.dispatcher_loaded = True

        # Подождать пока dispatcher инициализируется
        time.sleep(0.1)
        status = self.read_mem(STATUS_ADDR)
        if status != STATUS_DONE:
            raise RuntimeError(f"Dispatcher not ready, status={status}")

        print("Dispatcher loaded and ready!")

    def read_mem(self, offset, count=1):
        """Читать память карты"""
        if not self.access:
            self.get_access()

        buf = (ctypes.c_uint * count)()
        r = self.nm.PL_ReadMemBlock(self.access, buf, DDR_BASE + offset, count)
        if r != 0:
            raise RuntimeError(f"PL_ReadMemBlock failed: {r}")

        return buf[0] if count == 1 else list(buf)

    def write_mem(self, offset, data):
        """Писать в память карты"""
        if not self.access:
            self.get_access()

        if isinstance(data, int):
            data = [data]

        buf = (ctypes.c_uint * len(data))(*data)
        r = self.nm.PL_WriteMemBlock(self.access, buf, DDR_BASE + offset, len(data))
        if r != 0:
            raise RuntimeError(f"PL_WriteMemBlock failed: {r}")

    def send_command(self, cmd, args=None, timeout_ms=5000):
        """Отправить команду dispatcher-у и дождаться результата"""
        if args is None:
            args = []

        # Записать аргументы
        for i, arg in enumerate(args):
            self.write_mem(ARG0_ADDR + i, arg)

        # Отправить команду
        self.write_mem(CMD_ADDR, cmd)

        # Ждать завершения
        start = time.time()
        while True:
            status = self.read_mem(STATUS_ADDR)
            if status == STATUS_DONE:
                result = self.read_mem(RESULT_ADDR)
                return result
            elif status == STATUS_ERROR:
                raise RuntimeError(f"Command {cmd} failed with error")

            if (time.time() - start) * 1000 > timeout_ms:
                raise TimeoutError(f"Command {cmd} timeout")

            time.sleep(0.001)

    def ping(self, value=42):
        """Тест связи - возвращает value + 1"""
        result = self.send_command(OP_PING, [value])
        expected = value + 1
        if result != expected:
            raise RuntimeError(f"Ping failed: expected {expected}, got {result}")
        return result

    def shutdown(self):
        """
        БЕЗОПАСНОЕ ЗАВЕРШЕНИЕ!
        Отправляет OP_EXIT - dispatcher завершается корректно.
        После этого можно загрузить новую программу.
        """
        if not self.access:
            self.get_access()

        print("Sending OP_EXIT to dispatcher...")
        self.write_mem(CMD_ADDR, OP_EXIT)

        # Подождать завершения
        time.sleep(0.1)

        # Проверить результат
        result = self.read_mem(RESULT_ADDR)
        if result == 0xB1EB1E:
            print("Dispatcher terminated gracefully!")
        else:
            print(f"Dispatcher result: {hex(result)}")

        self.dispatcher_loaded = False

    def emergency_exit(self):
        """
        АВАРИЙНЫЙ ВЫХОД!
        Используй если карта зависла - записывает OP_EXIT напрямую.
        Работает даже без reset, если dispatcher правильный.
        """
        print("EMERGENCY: Writing OP_EXIT directly to memory...")
        try:
            self.get_access()  # Работает без reset!
            self.write_mem(CMD_ADDR, OP_EXIT)
            time.sleep(0.2)
            print("OP_EXIT sent. Check if card responds now.")
        except Exception as e:
            print(f"Emergency exit failed: {e}")
            print("Card may need REBOOT.")

    def close(self):
        """Закрыть все дескрипторы"""
        if self.dispatcher_loaded:
            try:
                self.shutdown()
            except:
                pass

        if self.access:
            self.nm.PL_CloseAccess(self.access)
            self.access = None

        if self.board:
            self.nm.PL_CloseBoardDesc(self.board)
            self.board = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================
# Пример использования
# ============================================

if __name__ == "__main__":
    print("=== DeviceSafe Test ===")

    try:
        with DeviceSafe() as dev:
            # Загрузить dispatcher
            dev.load_dispatcher()

            # Тест ping
            result = dev.ping(100)
            print(f"Ping test: sent 100, got {result}")

            # Завершить корректно
            dev.shutdown()

    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying emergency exit...")

        # Аварийный выход без reset
        dev = DeviceSafe()
        dev.emergency_exit()
        dev.close()
