/**
 * NMC Memory Worker - базовая программа для доступа к памяти
 *
 * Протокол:
 * 1. NMC выделяет буфер и отправляет адрес хосту (sync=1)
 * 2. Хост пишет данные по адресу
 * 3. NMC выполняет операцию (sync=2)
 * 4. Хост читает результат
 * 5. sync=99 для выхода
 */

#include <stdlib.h>
#include "nm6408load_nmc.h"

typedef unsigned int WORD32;

// Буфер в DDR - 64MB
#define BUFFER_SIZE (16 * 1024 * 1024)  // 16M words = 64MB
__attribute__((section(".data.ddr"))) WORD32 Buffer[BUFFER_SIZE];

// Второй буфер для результатов
__attribute__((section(".data.ddr"))) WORD32 ResultBuffer[BUFFER_SIZE];

int main()
{
    // Sync 1: Отправляем адреса буферов хосту
    // outArray = адрес входного буфера
    // outLen = размер
    ncl_hostSyncArray(1, Buffer, BUFFER_SIZE, NULL, NULL);

    // Sync 2: Отправляем адрес буфера результатов
    ncl_hostSyncArray(2, ResultBuffer, BUFFER_SIZE, NULL, NULL);

    // Главный цикл
    while (1)
    {
        int cmd = 0;
        void* inArray = NULL;
        unsigned int inLen = 0;

        // Ждём команду от хоста
        // cmd: 10 = copy, 11 = add_scalar, 12 = mul_scalar, 99 = exit
        cmd = ncl_hostSyncArray(3, NULL, 0, &inArray, &inLen);

        if (cmd == 99)
            break;

        if (cmd == 10) {
            // COPY: просто копируем Buffer -> ResultBuffer
            for (unsigned int i = 0; i < inLen && i < BUFFER_SIZE; i++) {
                ResultBuffer[i] = Buffer[i];
            }
        }
        else if (cmd == 11) {
            // ADD SCALAR: ResultBuffer[i] = Buffer[i] + scalar
            WORD32 scalar = *((WORD32*)inArray);
            for (unsigned int i = 0; i < inLen && i < BUFFER_SIZE; i++) {
                ResultBuffer[i] = Buffer[i] + scalar;
            }
        }
        else if (cmd == 12) {
            // MUL SCALAR: ResultBuffer[i] = Buffer[i] * scalar (float)
            float scalar = *((float*)inArray);
            float* fBuf = (float*)Buffer;
            float* fRes = (float*)ResultBuffer;
            for (unsigned int i = 0; i < inLen && i < BUFFER_SIZE; i++) {
                fRes[i] = fBuf[i] * scalar;
            }
        }
        else if (cmd == 20) {
            // VECTOR ADD: ResultBuffer[i] = Buffer[i] + Buffer[i + offset]
            unsigned int offset = inLen / 2;
            float* fBuf = (float*)Buffer;
            float* fRes = (float*)ResultBuffer;
            for (unsigned int i = 0; i < offset && i < BUFFER_SIZE; i++) {
                fRes[i] = fBuf[i] + fBuf[i + offset];
            }
        }

        // Sync 4: Сигнал что результат готов
        ncl_hostSync(4);
    }

    return 0;
}
